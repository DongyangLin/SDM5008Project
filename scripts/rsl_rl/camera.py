import numpy as np

class CameraController:
    """Camera controller with smoothing (Steadicam effect)."""

    def __init__(self, env, distance=6.0, height=3.0, smoothing=0.05):
        self.env = env
        self.distance = distance
        self.height = height

        # 0.0 ~ 1.0 之间。数值越小，镜头越平滑（延迟感越强）；数值越大，反应越快但越抖。
        # 0.05 是一个比较适合行走机器人的“云台”感数值。
        self.smoothing = smoothing

        # 用于存储上一帧的相机位置，用于平滑计算
        self.last_eye = None
        self.last_target = None

    def update_camera_view(self):
        robot = self.env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w[0].cpu().numpy()
        quat = robot.data.root_quat_w[0].cpu().numpy()

        # --- 1. 计算理想的目标位置 (Desired Position) ---

        # 只提取 Yaw (水平朝向)，完全忽略机器人的 Roll 和 Pitch
        # 这样即使机器人摔倒或倾斜，镜头依然保持水平
        w, x, y, z = quat
        robot_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # 左前方 45 度
        offset_angle = robot_yaw + np.deg2rad(45)

        offset_vector = np.array([np.cos(offset_angle), np.sin(offset_angle), 0.0])

        # 理想的相机位置 (Desired Eye)
        desired_eye = pos + offset_vector * self.distance
        desired_eye[2] = pos[2] + self.height  # 依然基于机器人高度，但稍后会平滑

        # 理想的观察点 (Desired Target)
        desired_target = pos.copy()
        desired_target[2] += 0.5

        # --- 2. 平滑处理 (Stabilization) ---

        if self.last_eye is None:
            # 第一帧，直接赋值，避免镜头飞入
            self.last_eye = desired_eye
            self.last_target = desired_target

        # 使用线性插值 (Lerp) 来平滑移动
        # new_val = old_val * (1 - alpha) + target_val * alpha
        # 这种算法可以有效过滤掉机器人脚步的高频震动

        current_eye = (
            self.last_eye * (1 - self.smoothing) + desired_eye * self.smoothing
        )
        current_target = (
            self.last_target * (1 - self.smoothing) + desired_target * self.smoothing
        )

        # --- 3. 应用并保存 ---

        self.env.unwrapped.sim.set_camera_view(
            eye=current_eye,
            target=current_target,
            camera_prim_path="/OmniverseKit_Persp",
        )

        # 更新历史位置
        self.last_eye = current_eye
        self.last_target = current_target
