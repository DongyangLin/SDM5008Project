"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--save_path", type=str, default=None, help="Path to save play logs (.npz/.csv).")

# 添加 RSL-RL cli 参数 / Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# 添加 AppLauncher cli 参数 / Append AppLauncher cli arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# 加载 omniverse 应用 / Load omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import sys
import time
import torch

from rsl_rl.runner import HIMOnPolicyRunner 

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import export_him_actor_critic_as_jit, export_him_actor_critic_as_onnx

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from utils.data_recorder import DataRecorder
from utils.camera import CameraController

def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""

    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 转换为单智能体实例（如果RL算法需要）/ Convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为 rsl-rl 包装环境 / Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    camera_controller = CameraController(env)

    # 加载先前训练的模型 / Load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # 获取训练好的策略以进行推理 / Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 导出策略到 onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_him_actor_critic_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        # print("Exported policy as jit script to: ", export_model_dir)

        export_him_actor_critic_as_onnx(
            ppo_runner.alg.actor_critic, export_model_dir,
        )

    # 重置环境 / Reset environment
    obs, extras = env.get_observations()

    # HIM：从 extras 中提取历史并展平 / HIM: extract history from extras and flatten
    obs_history = obs
    obs_history = obs_history.flatten(start_dim=1)

    step_idx = 0
    recorder = DataRecorder(log_dir, args_cli, lin_vel_start=33)
    # 模拟环境 / Simulate environment
    while simulation_app.is_running():

        # 以推理模式运行所有操作 / Run everything in inference mode
        with torch.inference_mode():
            camera_controller.update_camera_view()

            # 智能体步进 / Agent stepping
            actions = policy(obs_history)
            ret = env.step(actions)

            # 兼容性处理：检查返回值数量 / Compatibility handling: check number of return values
            if len(ret) == 5:
                obs, rew, terminated, truncated, extras = ret
            else:
                obs, rew, dones, extras = ret    # 假设是旧版或 Wrapper 后的 4 值 / Assume older version or after Wrapper with 4 values

            # HIM
            obs_history = obs
            obs_history = obs_history.flatten(start_dim=1)

            obs_pack = extras["observations"]["critic"]
            cmd_vel = obs_pack[..., 0:3]
            recorder.step(step_idx, obs_pack, cmd_vel, env)
            step_idx += 1

    # 关闭模拟器 / Close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # 运行主程序 / Run the main execution
    main()
    # 关闭模拟器应用 / Close simulator application
    simulation_app.close()
