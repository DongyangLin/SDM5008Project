import torch
import torch.nn.functional as F

def G_a(actions: torch.Tensor):
    """
    actions: [B, 6]
    """
    idx = actions.new_tensor([1, 0, 3, 2, 5, 4], dtype=torch.long)
    sign = actions.new_tensor([-1, -1, 1, 1, 1, 1])
    return actions[:, idx] * sign


def G_o_n(obs: torch.Tensor, history_length=5):
    """
    obs: [B, H*D] or [B, D]
         [  command(3),
            base_ang_vel(3),
            proj_gravity(3),
            joint_pos(6),
            joint_vel(6),
            last_action(6)  ]
    """
    obs_length = int(obs.shape[-1] / history_length)
    # print(f"obs_length: {obs_length}")

    for i in range(history_length):
        start = i * obs_length
        end = (i + 1) * obs_length
        obs_single = obs[:, start:end]

        # command
        obs_single[..., 1] *= -1      # lin_vel_y
        obs_single[..., 2] *= -1      # ang_vel_z

        # base_ang_vel (x,y,z) → y,z flip
        obs_single[..., 4] *= -1
        obs_single[..., 5] *= -1

        # joint-related
        jp = obs_single[..., 6:12]
        jv = obs_single[..., 12:18]
        la = obs_single[..., 18:24]

        idx = torch.tensor([1, 0, 3, 2, 5, 4], device=obs.device)

        obs_single[..., 6:12] = jp[..., idx]
        obs_single[..., 12:18] = jv[..., idx]
        obs_single[..., 18:24] = G_a(la.reshape(-1, 6)).reshape_as(la)

        obs[:, start:end] = obs_single

    return obs

def G_o_p(height_obs: torch.Tensor, Nx: int = 12, Ny: int = 8):
    """
    height_obs: [B, Nx*Ny]
    """
    B = height_obs.shape[0]
    height_obs_reshaped = height_obs.view(B, Ny, Nx).transpose(1, 2)
    height_obs_flipped = torch.flip(height_obs_reshaped, dims=[2]).transpose(1, 2)
    height_obs_flipped = height_obs_flipped.view(B, Nx * Ny)
    return height_obs_flipped


def compute_symmetry_loss(actor_critic, obs_n, obs_p, critic_obs, policy_sym_coef, value_sym_coef):
    # flip inputs
    obs_n_f = G_o_n(obs_n)
    obs_p_f = G_o_p(obs_p)
    critic_f = G_o_n(critic_obs, history_length=1)

    # policy
    a = actor_critic.act(obs_n, obs_p)
    a_f = actor_critic.act(obs_n_f, obs_p_f)
    policy_sym_loss = F.mse_loss(G_a(a), a_f)

    # value
    v = actor_critic.evaluate(critic_obs, obs_p)
    v_f = actor_critic.evaluate(critic_f, obs_p_f)
    value_sym_loss = F.mse_loss(v, v_f)

    return policy_sym_coef * policy_sym_loss + value_sym_coef * value_sym_loss
