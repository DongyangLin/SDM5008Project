import os
import time
import csv
import re
import numpy as np
import torch
from collections import defaultdict


class DataRecorder:
    def __init__(self, log_dir, args_cli, lin_vel_start=0, ang_vel_start=3, g_start=6):
        """
        初始化数据记录器
        """
        self.log_dir = log_dir
        self.args_cli = args_cli

        # 观测空间切片配置
        self.lin_vel_start = lin_vel_start
        self.ang_vel_start = ang_vel_start
        self.g_start = g_start

        # 内部状态
        self.logs = defaultdict(list)
        self.t0_wall = time.time()  # 记录器初始化时间作为起始时间

    def step(self, step_idx, obs_pack, commands, env, experiment_name="Flat"):
        """
        每一步调用此函数来记录数据。会自动判断是否需要保存到磁盘。
        """
        # 1. 提取数据
        # Commanded velocities
        cmd_triplet = self._extract_cmd_vxvywz(commands)
        cmd_vx, cmd_vy, cmd_wz = (
            cmd_triplet if cmd_triplet is not None else (None, None, None)
        )

        # Actual velocities
        act_triplet = self._extract_base_lin_ang_vel(obs_pack)
        act_vx, act_vy, act_wz = (
            act_triplet if act_triplet is not None else (None, None, None)
        )

        # Orientation & Forces
        roll, pitch = self._extract_base_quat(obs_pack)
        force_l, force_r = self._extract_feet_contact_forces(env)
        external_force = self._extract_non_foot_contact_forces(env)

        # 2. 存入内存 logs
        self.logs["step"].append(step_idx)
        self.logs["wall_time_s"].append(float(time.time() - self.t0_wall))

        self.logs["cmd_vx"].append(self._mean_or_nan(cmd_vx))
        self.logs["cmd_vy"].append(self._mean_or_nan(cmd_vy))
        self.logs["cmd_wz"].append(self._mean_or_nan(cmd_wz))

        self.logs["act_vx"].append(self._mean_or_nan(act_vx))
        self.logs["act_vy"].append(self._mean_or_nan(act_vy))
        self.logs["act_wz"].append(self._mean_or_nan(act_wz))

        self.logs["roll"].append(self._mean_or_nan(roll))
        self.logs["pitch"].append(self._mean_or_nan(pitch))
        self.logs["abs_roll"].append(np.abs(self._mean_or_nan(roll)))
        self.logs["abs_pitch"].append(np.abs(self._mean_or_nan(pitch)))
        self.logs["contact_force_L"].append(self._mean_or_nan(force_l))
        self.logs["contact_force_R"].append(self._mean_or_nan(force_r))
        self.logs["external_force"].append(self._mean_or_nan(external_force))

        # 3. 定期保存 (每500步)
        if step_idx % 500 == 0:
            self.save_to_disk(experiment_name)

    def save_to_disk(self, experiment_name):
        """显式保存数据到磁盘 (NPZ 和 CSV)"""
        save_dir = os.path.join(self.log_dir, "play_logs")
        os.makedirs(save_dir, exist_ok=True)

        if self.args_cli.save_path is None:
            npz_path = os.path.join(save_dir, f"{experiment_name}_play_log_mean.npz")
        else:
            npz_path = os.path.abspath(self.args_cli.save_path)
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)

        # 准备 numpy 数组
        out = {}
        # 确保 step 和 time 存在
        if "step" not in self.logs or not self.logs["step"]:
            return  # 没数据不保存

        out["step"] = np.asarray(self.logs["step"], dtype=np.int64)
        out["wall_time_s"] = np.asarray(self.logs["wall_time_s"], dtype=np.float64)

        keys_to_save = [
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
        ]

        for k in keys_to_save:
            out[k] = np.asarray(self.logs[k], dtype=np.float32)

        # 保存 NPZ
        np.savez_compressed(npz_path, **out)
        print(f"[INFO] Saved play logs to: {npz_path}")

        # 保存 CSV
        csv_path = os.path.splitext(npz_path)[0] + ".csv"
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["t_step", "wall_time_s"] + [
                    k.replace("contact_", "").replace("external_", "")
                    for k in keys_to_save
                ]
                writer.writerow(header)

                T = out["step"].shape[0]
                for t in range(T):
                    row = [int(out["step"][t]), float(out["wall_time_s"][t])]
                    for k in keys_to_save:
                        row.append(float(out[k][t]))
                    writer.writerow(row)
            print(f"[INFO] Saved play logs CSV to: {csv_path}")
        except Exception as e:
            print(f"[Error] Failed to write CSV: {e}")

    # ================= 内部辅助方法 =================

    def _to_cpu_np(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _mean_or_nan(self, x):
        if x is None:
            return np.nan
        x = self._to_cpu_np(x).astype(np.float32)
        return float(np.nanmean(x))

    def _extract_cmd_vxvywz(self, cmd):
        if cmd is None:
            return None
        cmd = self._to_cpu_np(cmd)
        if cmd.ndim == 1:
            cmd = cmd[None, :]
        if cmd.shape[-1] >= 3:
            return cmd[..., 0], cmd[..., 1], cmd[..., 2]
        return None

    def _extract_base_lin_ang_vel(self, obs_pack):
        lin = obs_pack[..., self.lin_vel_start : self.lin_vel_start + 3]
        ang = obs_pack[..., self.ang_vel_start : self.ang_vel_start + 3]

        lin = self._to_cpu_np(lin) if lin is not None else None
        ang = self._to_cpu_np(ang) if ang is not None else None

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
            wz = ang[..., 2] if ang.shape[-1] > 2 else ang[..., -1]
        else:
            wz = None

        if vx is None or vy is None or wz is None:
            return None
        return vx, vy, wz

    def _extract_base_quat(self, obs_pack):
        g = obs_pack[..., self.g_start : self.g_start + 3]
        g = self._to_cpu_np(g)
        if g.ndim == 1:
            g = g[None, :]
        gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]

        roll = np.arctan2(gy, gz)
        pitch = np.arctan2(-gx, np.sqrt(gy * gy + gz * gz))
        return roll, pitch

    def _extract_feet_contact_forces(self, env_instance):
        try:
            base_env = env_instance.unwrapped
            sensor_name = "contact_forces"
            if sensor_name not in base_env.scene.sensors:
                return np.nan, np.nan

            contact_sensor = base_env.scene.sensors[sensor_name]
            forces_tensor = contact_sensor.data.net_forces_w
            forces_np = self._to_cpu_np(forces_tensor)
            sensor_body_names = contact_sensor.body_names

            idx_L, idx_R = -1, -1
            for i, name in enumerate(sensor_body_names):
                if re.search(r".*foot_L_Link", name):
                    idx_L = i
                elif re.search(r".*foot_R_Link", name):
                    idx_R = i

            val_l = (
                np.mean(np.linalg.norm(forces_np[:, idx_L, :], axis=-1))
                if idx_L != -1
                else np.nan
            )
            val_r = (
                np.mean(np.linalg.norm(forces_np[:, idx_R, :], axis=-1))
                if idx_R != -1
                else np.nan
            )
            return val_l, val_r
        except Exception as e:
            # 这里的print在大量循环中可能会刷屏，建议根据需要开启
            # print(f"[Error] Extract forces failed: {e}")
            return np.nan, np.nan

    def _extract_non_foot_contact_forces(self, env_instance):
        try:
            base_env = env_instance.unwrapped
            asset_name = "robot"
            if asset_name not in base_env.scene.keys():
                return np.nan
            robot_asset = base_env.scene[asset_name]

            forces_tensor = robot_asset._external_force_b
            if forces_tensor is None:
                return 0.0
            forces_np = self._to_cpu_np(forces_tensor)

            asset_body_names = robot_asset.body_names
            target_indices = [
                i
                for i, name in enumerate(asset_body_names)
                if not re.search(r".*foot_[LR]_Link", name)
            ]

            if len(target_indices) > 0:
                relevant_forces = forces_np[:, target_indices, :]
                forces_norm = np.linalg.norm(relevant_forces, axis=-1)
                return np.mean(np.sum(forces_norm, axis=-1))
            return 0.0
        except Exception as e:
            return np.nan
