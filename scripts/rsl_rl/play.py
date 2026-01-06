import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# >>> ADDED: optional output path
parser.add_argument("--save_path", type=str, default=None, help="Path to save play logs (.npz/.csv).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import re
import gymnasium as gym
import os
import time
import csv
import numpy as np
import torch

from rsl_rl.runner import OnPolicyRunner, HIMOnPolicyRunner, PIMOnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


# >>> ADDED: helpers


def record_data(step_idx, t0_wall, log_dir, args_cli, obs_pack, commands, ppo_runner, logs, env):

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

    def _extract_feet_contact_forces(env_instance):
        try:
            base_env = env_instance.unwrapped
            sensor_name = "contact_forces"

            # 1. 获取传感器对象
            if sensor_name not in base_env.scene.sensors:
                return np.nan, np.nan

            contact_sensor = base_env.scene.sensors[sensor_name]

            # 2. 获取数据 (Num_Envs, Num_Bodies, 3)
            # 使用 net_forces_w 获取世界坐标系下的受力 (当前帧，无历史)
            forces_tensor = contact_sensor.data.net_forces_w
            forces_np = _to_cpu_np(forces_tensor)

            # 3. [关键步骤] 动态匹配 Body Names
            # contact_sensor.body_names 是一个列表，例如 ['g1_foot_L_Link', 'g1_foot_R_Link']
            # 我们需要找到哪个索引对应左脚，哪个对应右脚
            sensor_body_names = contact_sensor.body_names

            idx_L = -1
            idx_R = -1

            # 使用正则查找索引
            for i, name in enumerate(sensor_body_names):
                # 这里使用你提供的正则逻辑 ".*foot_[LR]_Link"
                if re.search(r".*foot_L_Link", name):
                    idx_L = i
                elif re.search(r".*foot_R_Link", name):
                    idx_R = i

            # 4. 提取数据
            val_l = np.nan
            val_r = np.nan

            # 如果找到了左脚索引
            if idx_L != -1:
                vec_l = forces_np[:, idx_L, :]  # (N, 3)
                val_l = np.mean(
                    np.linalg.norm(vec_l, axis=-1)
                )  # 求模长后取所有环境平均

            # 如果找到了右脚索引
            if idx_R != -1:
                vec_r = forces_np[:, idx_R, :]  # (N, 3)
                val_r = np.mean(np.linalg.norm(vec_r, axis=-1))

            return val_l, val_r

        except Exception as e:
            print(f"[Error] Extract forces failed: {e}")
            return np.nan, np.nan

    def _extract_non_foot_contact_forces(env_instance):
        """
        【修改版】
        获取当前通过 apply_external_force_torque 施加在机器人身上的主动外力（扰动）。
        来源：asset._external_force_b
        注意：这不再是“接触力”，而是你代码里写的“推力/踢力”。
        """
        try:
            base_env = env_instance.unwrapped

            # 1. 获取 Robot Asset
            # 通常名字是 "robot" 或 "g1"，需要根据你的 config 确认
            asset_name = "robot" 

            if asset_name not in base_env.scene.keys():
                # 尝试找找有没有叫 "g1" 的，或者打印 keys 帮你 debug
                # print(f"Available assets: {base_env.scene.keys()}")
                return np.nan

            robot_asset = base_env.scene[asset_name]

            # 2. [核心修改] 获取外部力 Buffer
            # 这个 Buffer 存储了本帧施加的扰动 (Num_Envs, Num_Bodies, 3)
            forces_tensor = robot_asset._external_force_b

            # 转为 Numpy
            if forces_tensor is None:
                return 0.0

            forces_np = _to_cpu_np(forces_tensor)

            # 3. 获取 Body Names 用于过滤
            # robot_asset.body_names 包含了所有连杆的名字
            asset_body_names = robot_asset.body_names

            # 4. 筛选非足部索引 (通常外力施加在 Base 上，所以这个筛选依然有效)
            target_indices = []

            for i, name in enumerate(asset_body_names):
                # 逻辑：排除脚部，统计剩下所有部位（主要是躯干）受到的推力
                if not re.search(r".*foot_[LR]_Link", name):
                    target_indices.append(i)

            # 5. 计算受力总和
            if len(target_indices) > 0:
                # 提取目标部位数据 (N, Num_Targets, 3)
                relevant_forces = forces_np[:, target_indices, :]

                # 计算每个刚体受力的模长
                forces_norm = np.linalg.norm(relevant_forces, axis=-1)

                # 求和：得到每个环境受到的总扰动强度
                total_disturbance = np.sum(forces_norm, axis=-1)

                # 返回平均值 (Scalar)
                return np.mean(total_disturbance)
            else:
                return 0.0

        except Exception as e:
            print(f"[Error] Extract external forces failed: {e}")
            return np.nan

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
    force_l, force_r = _extract_feet_contact_forces(env)
    external_force = _extract_non_foot_contact_forces(env)

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
    logs["contact_force_L"].append(_mean_or_nan(force_l))
    logs["contact_force_R"].append(_mean_or_nan(force_r))
    logs["external_force"].append(_mean_or_nan(external_force))

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
        for k in [
            "cmd_vx",
            "cmd_vy",
            "cmd_wz",
            "act_vx",
            "act_vy",
            "act_wz",
            "roll",
            "pitch",
            "abs_roll",
            "abs_pitch",
            "contact_force_L",
            "contact_force_R",
            "external_force",
        ]:
            out[k] = np.asarray(logs[k], dtype=np.float32)  # (T,)

        np.savez_compressed(npz_path, **out)
        print(f"[INFO] Saved play logs to: {npz_path}")

        csv_path = os.path.splitext(npz_path)[0] + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t_step",
                    "wall_time_s",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_wz",
                    "act_vx",
                    "act_vy",
                    "act_wz",
                    "roll",
                    "pitch",
                    "force_L",
                    "force_R",
                    "external_force",
                ]
            )
            T = out["step"].shape[0]
            for t in range(T):
                writer.writerow(
                    [
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
                        float(out["contact_force_L"][t]),  # <--- 写入左脚
                        float(out["contact_force_R"][t]),  # <--- 写入右脚
                        float(out["external_force"][t]),  # <--- 写入非足部受力
                    ]
                )
        print(f"[INFO] Saved play logs CSV to: {csv_path}")

def main():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

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

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_model_dir)
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            export_model_dir,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    # >>> ADDED: logging buffers
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
        "contact_force_L": [],
        "contact_force_R": [],
        "external_force": [],
    }

    # num_envs = int(getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1)))
    

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs, _, _, infos = env.step(actions)

            # update history/commands (existing)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands")

            # >>> ADDED: extract + record
            # pack candidates: prefer infos["observations"], fall back to infos itself, then obs_dict["observations"]
            obs_pack = infos["observations"]["critic"]

            record_data(step_idx, t0_wall, log_dir, args_cli, obs_pack, commands, ppo_runner, logs, env)

            step_idx += 1

    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    main()
    simulation_app.close()
