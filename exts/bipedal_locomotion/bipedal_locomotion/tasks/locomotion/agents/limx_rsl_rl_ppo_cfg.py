from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg, HIMActorCriticCfg

import os
robot_type = os.getenv("ROBOT_TYPE")  # 从环境变量获取机器人类型 / Get robot type from environment variable

# Isaac Lab original RSL-RL configuration
@configclass
class PFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24        # 每个环境每次收集的步数 / Steps collected per environment per iteration
    max_iterations = 15000        # 最大训练迭代次数 / Maximum training iterations
    save_interval = 500           # 模型保存间隔 / Model saving interval
    experiment_name = "pf_flat"   # 实验名称 / Experiment name
    empirical_normalization = False  # 不使用经验归一化 / Don't use empirical normalization

    # Actor-Critic网络配置
    # Actor-Critic network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # 初始动作噪声标准差 / Initial action noise std
        actor_hidden_dims=[512, 256, 128],  # Actor网络隐藏层维度 / Actor network hidden dimensions
        critic_hidden_dims=[512, 256, 128], # Critic网络隐藏层维度 / Critic network hidden dimensions
        activation="elu",                # 激活函数类型 / Activation function type
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 值函数损失系数 / Value function loss coefficient
        use_clipped_value_loss=True,  # 使用截断值函数损失 / Use clipped value function loss
        clip_param=0.2,               # PPO截断参数 / PPO clipping parameter
        entropy_coef=0.01,            # 熵正则化系数 / Entropy regularization coefficient
        num_learning_epochs=5,        # 每次迭代的学习轮数 / Learning epochs per iteration
        num_mini_batches=4,           # 小批次数量 / Number of mini-batches
        learning_rate=1.0e-3,         # 学习率 / Learning rate
        schedule="adaptive",          # 自适应学习率调度 / Adaptive learning rate schedule
        gamma=0.99,                   # 折扣因子 / Discount factor
        lam=0.95,                     # GAE lambda参数 / GAE lambda parameter
        desired_kl=0.01,              # 目标KL散度 / Target KL divergence
        max_grad_norm=1.0,            # 梯度裁剪阈值 / Gradient clipping threshold
    )


# PF_TRON1A平地训练配置 - 针对特定机器人型号的优化配置
# PF_TRON1A flat terrain training configuration - optimized for specific robot model
@configclass
class PF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000         # 较短的训练周期，适合平地环境 / Shorter training for flat terrain
    save_interval = 200           # 更频繁的保存 / More frequent saving
    experiment_name = "pf_tron_1a_flat"
    empirical_normalization = False   # False ！！！！！！！！！！！！！
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,   # 1.0, ！！！！！！！！！！！！！
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # 使用MLP版本的PPO算法，支持历史观测
    # Use MLP version of PPO algorithm with history observation support
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # 1.0e-3 ！！！！！！！！！！！！！
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,  # 1.0, ！！！！！！！！！！！！！
        obs_history_len=10,   # 观测历史长度 / Observation history length
    )

    # 编码器配置 - 用于处理历史观测信息
    # Encoder configuration - for processing history observation information
    encoder = EncoderCfg(
        output_detach=True,       # 输出分离，防止梯度回传 / Detach output to prevent gradient flow
        num_output_dim=3,         # 输出维度 / Output dimensions
        hidden_dims=[256, 128],   # 编码器隐藏层 / Encoder hidden dimensions
        activation="elu",         # 激活函数 / Activation function
        orthogonal_init=False,    # 不使用正交初始化 / Don't use orthogonal initialization
    )
    
# Stair Config !!!!!!!!!!
@configclass
class PF_TRON1AStairPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000         # longer iter, for stair terrains
    save_interval = 200           # 更频繁的保存 / More frequent saving
    experiment_name = "pf_tron_1a_flat"
    empirical_normalization = False   # False ！！！！！！！！！！！！！
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,   # 1.0, ！！！！！！！！！！！！！
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # 使用MLP版本的PPO算法，支持历史观测
    # Use MLP version of PPO algorithm with history observation support
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0, 
        obs_history_len=10,   # 观测历史长度 / Observation history length
    )

    # 编码器配置 - 用于处理历史观测信息
    # Encoder configuration - for processing history observation information
    encoder = EncoderCfg(
        output_detach=True,       # 输出分离，防止梯度回传 / Detach output to prevent gradient flow
        num_output_dim=3,         # 输出维度 / Output dimensions
        hidden_dims=[256, 128],   # 编码器隐藏层 / Encoder hidden dimensions
        activation="elu",         # 激活函数 / Activation function
        orthogonal_init=False,    # 不使用正交初始化 / Don't use orthogonal initialization
    )
    
# HIM Config
@configclass
class PF_HIM_PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # [Runner Settings]
    # 指定使用自定义的 HIM Runner (必须与 him_on_policy_runner.py 中的类名一致)
    runner_class_name = "HIMOnPolicyRunner" 
    
    num_steps_per_env = 100       # HIM 论文推荐 100 步  / HIM paper recommends 100 steps
    max_iterations = 20000        # 训练迭代次数 / Max training iterations
    save_interval = 200            # 保存间隔 / Save interval
    experiment_name = "pf_him_stair"
    empirical_normalization = False   # HIM 不使用经验归一化 / HIM does not use empirical normalization
    
    # [Policy Settings]
    # 使用自定义的 HIM 策略网络
    policy = RslRlPpoActorCriticCfg(
        class_name="HIMActorCritic", # 对应 him_actor_critic.py 中的类名
        init_noise_std=1.0,          # 初始噪声标准差 [cite: 468]
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # [Algorithm Settings]
    # 使用自定义的 HIM PPO 算法
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="HIMPPO",            # 对应 him_ppo.py 中的类名
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,             # 论文中 batch size 较大，这里根据显存调整
        learning_rate=1.0e-3,           # 学习率 [cite: 468]
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

#-----------------------------------------------------------------
@configclass
class SF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )
