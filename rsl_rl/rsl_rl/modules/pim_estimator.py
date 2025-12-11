import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class PIMEstimator(nn.Module):
    def __init__(self,
                 history_length,
                 dim_nonperceptive_obs,
                 dim_perceptive_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=16,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(PIMEstimator, self).__init__()
        activation = get_activation(activation)

        self.history_length = history_length   # H = 5
        self.dim_nonperceptive_obs = dim_nonperceptive_obs   # dim(o^n_t) = 45
        self.dim_perceptive_obs = dim_perceptive_obs   # dim(o^p_t) = 96
        self.dim_latent = enc_hidden_dims[-1]   # dim(I_t) = 16
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # Perceptive Encoder
        enc_input_dim = self.history_length * self.dim_nonperceptive_obs + self.dim_perceptive_obs   # History (source): dim(o^n_t) * H + dim(o^p_t) = 45 * 5 + 96
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target Encoder
        tar_input_dim = self.dim_nonperceptive_obs + self.dim_perceptive_obs   # Current (target)：dim(o^n_t+1) + dim(o^p_t) = 45 + 96
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history, obs_perceptive):
        ''' obs_history: (batch_size, H*dim_nonperceptive_obs), 
            obs_perceptive: (batch_size, dim_perceptive_obs) '''
        vel, z = self.encode(obs_history, obs_perceptive)
        return vel.detach(), z.detach()

    def forward(self, obs_history, obs_perceptive):
        ''' obs_history: (batch_size, H*dim_nonperceptive_obs), 
            obs_perceptive: (batch_size, dim_perceptive_obs) '''
        parts = self.encoder(torch.cat([obs_history.detach(), obs_perceptive.detach()], dim=-1))
        vel, z = parts[..., :3], parts[..., 3:]  # vel, l_S
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    # Same as forward
    def encode(self, obs_history, obs_perceptive):   
        ''' obs_history: (batch_size, H*dim_nonperceptive_obs), 
            obs_perceptive: (batch_size, dim_perceptive_obs) '''
        parts = self.encoder(torch.cat([obs_history.detach(), obs_perceptive.detach()], dim=-1))
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, obs_perceptive, next_critic_obs, lr=None):
        ''' obs_history: (batch_size, H*dim_nonperceptive_obs), 
            obs_perceptive: (batch_size, dim_perceptive_obs) 
            next_critic_obs: (batch_size, dim_nonperceptive_obs + dim_privileged_obs) '''
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, self.dim_nonperceptive_obs: self.dim_nonperceptive_obs + 3].detach()   # ground truth (GT) vel
        next_obs = next_critic_obs.detach()[:, 3: self.dim_nonperceptive_obs + 3]   # next obs without cmd vel (include GT vel)

        z_s = self.encoder(obs_history, obs_perceptive)
        z_t = self.target(next_obs, obs_perceptive.detach())
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)   # normalize latent vectors
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone() # weight of prototype
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T   # similarity between z_s and prototypes
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)   # soft assignment
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()   # swap loss
        estimation_loss = F.mse_loss(pred_vel, vel)   # velocity prediction loss
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None