"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--save_path", type=str, default=None, help="Path to save play logs (.npz/.csv).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import csv
import numpy as np
import torch

from rsl_rl.runner import PIMOnPolicyRunner 

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl.pim_exporter import export_pim_actor_critic_as_jit, export_pim_actor_critic_as_onnx
# from play import record_data


def record_data(step_idx, t0_wall, log_dir, args_cli, obs_pack, commands, logs):

    def _to_cpu_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        # list/tuple -> np
        return np.asarray(x)

    def _extract_cmd_vxvywz(cmd):
        """
        cmd can be shape (num_envs, K). We assume first 3 dims are vx, vy, wz if available.
        Adjust here if your task uses different ordering.
        """
        if cmd is None:
            return None
        cmd = _to_cpu_np(cmd)
        if cmd.ndim == 1:
            cmd = cmd[None, :]
        if cmd.shape[-1] >= 3:
            return cmd[..., 0], cmd[..., 1], cmd[..., 2]
        return None

    def _extract_base_lin_ang_vel(obs_pack):
        """
        Try to find actual base velocity from infos/obs dict.
        We attempt common keys. You may need to adapt candidate keys to your task.
        Returns vx, vy, wz arrays shape (num_envs,)
        """
        # common candidates for linear vel and angular vel (z)
        lin = obs_pack[..., 27:30]
        lin = obs_pack[..., 0:3]
        ang = obs_pack[..., 3:6]

        lin = _to_cpu_np(lin) if lin is not None else None
        ang = _to_cpu_np(ang) if ang is not None else None

        if lin is not None:
            if lin.ndim == 1:
                lin = lin[None, :]
            vx = lin[..., 0]
            vy = lin[..., 1] if lin.shape[-1] > 1 else np.zeros_like(vx)
        else:
            vx = vy = None

        if ang is not None:
            if ang.ndim == 1:
                ang = ang[None, :]
            # prefer z component if exists
            wz = ang[..., 2] if ang.shape[-1] > 2 else ang[..., -1]
        else:
            wz = None

        if vx is None or vy is None or wz is None:
            return None
        return vx, vy, wz

    def _extract_base_quat(obs_pack):
        """
        Try to find base orientation quaternion (x,y,z,w).
        """
        g = obs_pack[..., 6:9]
        g = _to_cpu_np(g)
        if g.ndim == 1:
            g = g[None, :]
        gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]

        # roll around x, pitch around y (standard aerospace convention)
        roll = np.arctan2(gy, gz)
        pitch = np.arctan2(-gx, np.sqrt(gy * gy + gz * gz))
        return roll, pitch

    # commanded velocities
    cmd_triplet = _extract_cmd_vxvywz(commands)
    if cmd_triplet is not None:
        cmd_vx, cmd_vy, cmd_wz = cmd_triplet
    else:
        cmd_vx = cmd_vy = cmd_wz = None

    # actual velocities
    act_triplet = _extract_base_lin_ang_vel(obs_pack)
    if act_triplet is not None:
        act_vx, act_vy, act_wz = act_triplet
    else:
        act_vx = act_vy = act_wz = None

    # roll/pitch from base quaternion
    roll, pitch = _extract_base_quat(obs_pack)

    # push to logs (store None-safe; if missing, store NaNs)
    def _mean_or_nan(x):
        if x is None:
            return np.nan
        x = _to_cpu_np(x).astype(np.float32)
        return float(np.nanmean(x))

    logs["step"].append(step_idx)
    logs["wall_time_s"].append(float(time.time() - t0_wall))

    logs["cmd_vx"].append(_mean_or_nan(cmd_vx))
    logs["cmd_vy"].append(_mean_or_nan(cmd_vy))
    logs["cmd_wz"].append(_mean_or_nan(cmd_wz))

    logs["act_vx"].append(_mean_or_nan(act_vx))
    logs["act_vy"].append(_mean_or_nan(act_vy))
    logs["act_wz"].append(_mean_or_nan(act_wz))

    logs["roll"].append(_mean_or_nan(roll))
    logs["pitch"].append(_mean_or_nan(pitch))
    logs["abs_roll"].append(np.abs(_mean_or_nan(roll)))
    logs["abs_pitch"].append(np.abs(_mean_or_nan(pitch)))

    if step_idx % 500 == 0:
        save_dir = os.path.join(log_dir, "play_logs")
        os.makedirs(save_dir, exist_ok=True)

        if args_cli.save_path is None:
            npz_path = os.path.join(save_dir, "play_log_mean.npz")
        else:
            npz_path = os.path.abspath(args_cli.save_path)
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)

        out = {}
        out["step"] = np.asarray(logs["step"], dtype=np.int64)
        out["wall_time_s"] = np.asarray(logs["wall_time_s"], dtype=np.float64)
        for k in ["cmd_vx","cmd_vy","cmd_wz","act_vx","act_vy","act_wz","roll","pitch","abs_roll","abs_pitch"]:
            out[k] = np.asarray(logs[k], dtype=np.float32)  # (T,)

        np.savez_compressed(npz_path, **out)
        print(f"[INFO] Saved play logs to: {npz_path}")

        csv_path = os.path.splitext(npz_path)[0] + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_step","wall_time_s","cmd_vx","cmd_vy","cmd_wz","act_vx","act_vy","act_wz","roll","pitch"])
            T = out["step"].shape[0]
            for t in range(T):
                writer.writerow([
                    int(out["step"][t]),
                    float(out["wall_time_s"][t]),
                    float(out["cmd_vx"][t]),
                    float(out["cmd_vy"][t]),
                    float(out["cmd_wz"][t]),
                    float(out["act_vx"][t]),
                    float(out["act_vy"][t]),
                    float(out["act_wz"][t]),
                    float(out["roll"][t]),
                    float(out["pitch"][t]),
                ])
        print(f"[INFO] Saved play logs CSV to: {csv_path}")

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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    ppo_runner = PIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 导出策略到jit / Export policy to jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_pim_actor_critic_as_jit(
        ppo_runner.alg.actor_critic, export_model_dir
    )
    print("Exported policy as jit script to: ", export_model_dir)

    if EXPORT_POLICY:
        # 导出策略到onnx / Export policy to onnx
        export_pim_actor_critic_as_onnx(
            ppo_runner.alg.actor_critic, export_model_dir,
        )
        print("Exported policy as onnx model to: ", export_model_dir)

        # export_mlp_as_onnx(
        #     ppo_runner.alg.actor_critic.actor, 
        #     export_model_dir, 
        #     "policy",
        #     ppo_runner.alg.actor_critic.num_actor_obs,
        # )
        # export_mlp_as_onnx(
        #     ppo_runner.alg.encoder,
        #     export_model_dir,
        #     "encoder",
        #     ppo_runner.alg.encoder.num_input_dim,
        # )
    # else:
        # reset environment
        obs, extras = env.get_observations()
        
        # PIM 关键：从 extras 中提取历史并展平, 提取感知观测
        obs_history = obs
        obs_history = obs_history.flatten(start_dim=1)
        obs_perceptive = extras["observations"]["perceptive"].squeeze(1)
        
        t0_wall = time.time()
        step_idx = 0

        logs = {
            "step": [],
            "wall_time_s": [],
            "cmd_vx": [],
            "cmd_vy": [],
            "cmd_wz": [],
            "act_vx": [],
            "act_vy": [],
            "act_wz": [],
            "roll": [],
            "pitch": [],
            "abs_roll": [],
            "abs_pitch": [],
        }
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs_history, obs_perceptive)
                ret = env.step(actions)
                # 兼容性处理：检查返回值数量
                if len(ret) == 5:
                    obs, rew, terminated, truncated, extras = ret
                else:
                    obs, rew, dones, extras = ret # 假设是旧版或 Wrapper 后的 4 值
                
                # PIM
                obs_history = obs
                obs_history = obs_history.flatten(start_dim=1)
                obs_perceptive = extras["observations"]["perceptive"].squeeze(1)
                # print(f"obs_perceptive: {obs_perceptive}")
                obs_pack = extras["observations"]["critic"]
                cmd_vel = obs_pack[..., 0:3]
                record_data(step_idx, t0_wall, log_dir, args_cli, obs_pack, cmd_vel, logs)
                step_idx += 1

        # close the simulator
        env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()