import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取CSV文件
# 请将 'your_data.csv' 替换为你实际的文件名
file_path = "/home/user/SDM5008/limxtron1lab-main/logs/rsl_rl/pf_him_stair/2025-12-16_Stable_Phase_3/play_logs/play_log_mean.csv"
data = pd.read_csv(file_path)

# 2. 数据预处理
# 如果 wall_time_s 是很大的时间戳，建议减去第一帧的时间，从0秒开始显示
# 如果想用 t_step 作为横坐标，可以将下行改为: time_axis = data['t_step']
time_axis = data["wall_time_s"] - data["wall_time_s"].iloc[0]

# 3. 创建画布 (3行1列的子图)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- 子图 1: X轴线性速度 (Vx) ---
ax1.plot(
    time_axis, data["cmd_vx"], "r--", label="Cmd Vx (指令)", linewidth=1.5
)  # 红色虚线
ax1.plot(
    time_axis, data["act_vx"], "b-", label="Act Vx (实际)", linewidth=1.5, alpha=0.8
)  # 蓝色实线
ax1.set_ylabel("Velocity X (m/s)", fontsize=12)
ax1.set_title("Velocity Tracking Performance", fontsize=14)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# --- 子图 2: Y轴线性速度 (Vy) ---
ax2.plot(time_axis, data["cmd_vy"], "r--", label="Cmd Vy (指令)", linewidth=1.5)
ax2.plot(
    time_axis, data["act_vy"], "g-", label="Act Vy (实际)", linewidth=1.5, alpha=0.8
)  # 绿色实线
ax2.set_ylabel("Velocity Y (m/s)", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

# --- 子图 3: Z轴角速度 (Wz) ---
ax3.plot(time_axis, data["cmd_wz"], "r--", label="Cmd Wz (指令)", linewidth=1.5)
ax3.plot(
    time_axis, data["act_wz"], "k-", label="Act Wz (实际)", linewidth=1.5, alpha=0.8
)  # 黑色实线
ax3.set_ylabel("Angular Velocity Z (rad/s)", fontsize=12)
ax3.set_xlabel("Time (s)", fontsize=12)
ax3.legend(loc="upper right")
ax3.grid(True, linestyle=":", alpha=0.6)

# 4. 调整布局并显示
plt.tight_layout()

# 如果需要保存图片，取消下面这行的注释
# plt.savefig('velocity_tracking.png', dpi=300)

plt.show()
