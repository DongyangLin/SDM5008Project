import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import colorsys
import matplotlib.colors as mcolors

# ================= 配置区域 =================
csv_file_path = (
    "logs/rsl_rl/pf_tron_1a_flat/2025-12-15_16-38-07/play_logs/play_log_mean.csv"
)
experiment_name = "flat"
# ===========================================

# ================= 全局绘图风格设置 (学术风格) =================
# 如果系统中没有 Times New Roman，会自动回退到默认衬线体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["axes.linewidth"] = 1.0  # 坐标轴线宽
plt.rcParams["grid.alpha"] = 0.3  # 网格透明度
plt.rcParams["grid.linestyle"] = "--"  # 网格线型
plt.rcParams["font.size"] = 12  # 基础字号

# 定义学术配色 (Hex codes)
COLOR_CMD = "#FA0101FF"  # 指令: 深灰色 (作为背景参考)
COLOR_VX = "#1973C2"  # Vx: 经典蓝
COLOR_VY = "#00B945"  # Vy: 鲜明绿
COLOR_WZ = "#FF9500"  # Wz: 砖红色 (原来是黑色，但在论文中红色更适合强调角速度)
COLOR_ROLL = "#00B945"  # Roll: 紫色
COLOR_PITCH = "#FF9500"  # Pitch: 橙色
COLOR_ZERO = "#1973C2"  # Zero Line: 浅红色 (珊瑚色)


def get_enhanced_color(hex_color, sat_factor=1.2, val_factor=0.85):
    """
    获取饱和度更高、且稍微深一点的颜色，用于绘制曲线。
    :param sat_factor: 饱和度倍数 (>1.0 为增加)
    :param val_factor: 亮度倍数 (<1.0 为变暗/变深)
    """
    rgb = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    # 增加饱和度 (最高不超过 1.0)
    s = min(1.0, s * sat_factor)
    # 稍微降低亮度 (让线条看起来更扎实，对比度更高)
    v = max(0.0, v * val_factor)

    return mcolors.to_hex(colorsys.hsv_to_rgb(h, s, v))


def tensorboard_smoothing(scalars, weight=0.6):
    """
    实现 TensorBoard 风格的平滑 (Exponential Moving Average).
    :param scalars: 原始数据列表或数组
    :param weight: 平滑系数 (0-1). 0为无平滑, 接近1为最大平滑.
    :return: 平滑后的 numpy 数组
    """
    if weight <= 0 or weight >= 1:
        return scalars

    last = scalars[0]  # 初始化
    smoothed = []
    for point in scalars:
        # EMA 公式: S_t = S_{t-1} * weight + Y_t * (1 - weight)
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def fix_roll_phase(roll_data):
    """
    修正Roll角的相位问题：
    1. 解缠绕，消除 +/- 3.14 的跳变
    2. 如果数据中心在 +/- 3.14 附近，则平移 PI，使其归零
    """
    # 1. 解缠绕 (处理 -3.14 <-> 3.14 的跳变)
    roll_unwrapped = np.unwrap(roll_data, discont=np.pi)

    # 2. 检查平均值是否偏移了 PI (约3.14)
    mean_val = np.mean(roll_unwrapped)

    # 如果平均值接近 PI 或 -PI (我们设定阈值为 2.0，超过即认为有偏移)
    if np.abs(mean_val) > 2.0:
        # 计算偏移了多少个 PI
        k = np.round(mean_val / np.pi)
        print(f"[数据修正] 检测到 Roll 均值偏移 {mean_val:.2f} (约 {k}*PI)")
        print(f"[数据修正] 正在执行平移操作，将中心归零...")
        roll_corrected = roll_unwrapped - k * np.pi
        return roll_corrected

    return roll_unwrapped


def analyze_robot_data(file_path, smoothing=0.8):
    file_dir = os.path.dirname(os.path.abspath(file_path))
    save_dir = os.path.join(file_dir, "images")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在读取文件: {file_path}")
    data = pd.read_csv(file_path)

    # =========================================================
    # [新增功能] 数据预处理：解缠绕 (Unwrap)
    # =========================================================
    # 这一步会消除 -3.14 到 3.14 的跳变，使曲线变得连续
    # disont=np.pi 表示如果两帧之间差值超过 pi，就认为发生了跳变
    data["roll"] = fix_roll_phase(data["roll"].values)
    data["pitch"] = np.unwrap(data["pitch"].values, discont=np.pi)

    # 时间轴归零
    time_axis = data["wall_time_s"] - data["wall_time_s"].iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    # 辅助函数：处理"原始+平滑"的双重绘制逻辑
    def plot_smooth_line(ax, x, y, color, label, smooth_factor):
        if smooth_factor > 0:
            # 1. 绘制原始数据 (背景，淡色，无标签)
            ax.plot(x, y, color=color, linestyle="-", linewidth=1.0, alpha=0.25)
            # 2. 计算平滑数据
            y_smooth = tensorboard_smoothing(y.values, weight=smooth_factor)
            # 3. 绘制平滑数据 (前景，深色，带标签)
            ax.plot(
                x,
                y_smooth,
                color=color,
                linestyle="-",
                linewidth=1.8,
                alpha=1.0,
                label=label,
            )
        else:
            # 不平滑：只画一条线
            ax.plot(
                x, y, color=color, linestyle="-", linewidth=1.5, alpha=0.9, label=label
            )

    # =========================================================
    # 图表 1: 速度跟踪对比 (Velocity Tracking)
    # =========================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- Vx ---
    # 指令不需要平滑，保持虚线
    ax1.plot(
        time_axis,
        data["cmd_vx"],
        color=COLOR_CMD,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    # 实际值应用平滑
    plot_smooth_line(ax1, time_axis, data["act_vx"], COLOR_VX, "Act Vx", smoothing)

    ax1.set_ylabel("Velocity X (m/s)")
    ax1.set_title(
        f"Velocity Tracking (Smoothing={smoothing})",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
    ax1.grid(True)

    # --- Vy ---
    ax2.plot(
        time_axis,
        data["cmd_vy"],
        color=COLOR_CMD,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    plot_smooth_line(ax2, time_axis, data["act_vy"], COLOR_VY, "Act Vy", smoothing)

    ax2.set_ylabel("Velocity Y (m/s)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
    ax2.grid(True)

    # --- Wz ---
    ax3.plot(
        time_axis,
        data["cmd_wz"],
        color=COLOR_CMD,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    plot_smooth_line(ax3, time_axis, data["act_wz"], COLOR_WZ, "Act Wz", smoothing)

    ax3.set_ylabel("Angular Rate Z (rad/s)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{experiment_name}_1_velocity_tracking.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # =========================================================
    # 图表 2: 跟踪误差分布 (带正态拟合曲线)
    # =========================================================

    # 1. 计算误差
    err_vx = data["cmd_vx"] - data["act_vx"]
    err_vy = data["cmd_vy"] - data["act_vy"]
    err_wz = data["cmd_wz"] - data["act_wz"]

    # 2. 计算 MSE
    mse_vx = np.mean(err_vx**2)
    mse_vy = np.mean(err_vy**2)
    mse_wz = np.mean(err_wz**2)

    # 3. 绘图
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    # 直方图通用参数
    hist_params = dict(
        bins=50, alpha=0.7, edgecolor="white", linewidth=0.5, density=False
    )

    def plot_fitted_gaussian(ax, data, base_color, mse_val, label_prefix):
        # A. 绘制直方图 (使用基础颜色)
        n, bins, patches = ax.hist(data, color=base_color, **hist_params)

        # B. 拟合正态分布
        mu, std = norm.fit(data)

        # C. 生成曲线坐标
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)  # 增加点数使曲线更平滑
        p = norm.pdf(x, mu, std)

        # D. 缩放 PDF 以匹配直方图高度
        bin_width = bins[1] - bins[0]
        p_scaled = p * len(data) * bin_width

        # E. 计算曲线颜色 (基于直方图颜色增强)
        curve_color = get_enhanced_color(base_color, sat_factor=1.2, val_factor=0.85)

        # F. 绘制拟合曲线
        # 使用实线或长虚线，线宽加粗
        label_text = f"Fit ($\mu={mu:.3f}, \sigma={std:.3f}$)"
        ax.plot(
            x,
            p_scaled,
            color=curve_color,
            linestyle="--",
            linewidth=2.5,
            label=label_text,
        )

        # 设置标题和图例
        ax.set_title(
            f"{label_prefix} (MSE: {mse_val:.1e})", fontsize=12, fontweight="bold"
        )
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Vx ---
    plot_fitted_gaussian(ax1, err_vx, COLOR_VX, mse_vx, "Error $v_x$")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Error $v_x$ (m/s)")
    ax1.set_title(
        f"Error Distribution $v_x$ (MSE: {mse_vx:.1e})", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Vy ---
    plot_fitted_gaussian(ax2, err_vy, COLOR_VY, mse_vy, "Error $v_y$")
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Error $v_y$ (m/s)")
    ax2.set_title(
        f"Error Distribution $v_y$ (MSE: {mse_vy:.1e})", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Wz ---
    plot_fitted_gaussian(ax3, err_wz, COLOR_WZ, mse_wz, "Error $\omega_z$")
    ax3.set_ylabel("Frequency")
    ax3.set_xlabel("Error $\omega_z$ (rad/s)")
    ax3.set_title(
        f"Error Distribution $\omega_z$ (MSE: {mse_wz:.1e})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{experiment_name}_2_error_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # =========================================================
    # 图表 3: 姿态震荡 (Oscillation) - [应用平滑]
    # =========================================================
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    roll_amp = data["roll"].max() - data["roll"].min()
    pitch_amp = data["pitch"].max() - data["pitch"].min()

    # --- Roll ---
    # 应用平滑绘制
    plot_smooth_line(ax1, time_axis, data["roll"], COLOR_ROLL, "Roll Angle", smoothing)

    # 0.0 基准线 (使用专门的 COLOR_ZERO 珊瑚色，更美观)
    ax1.axhline(
        0.0,
        color=COLOR_ZERO,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Zero Ref",
    )

    # 填充背景 (可选)
    ax1.fill_between(
        time_axis,
        data["roll"].min(),
        data["roll"].max(),
        color=COLOR_ROLL,
        alpha=0.03,
        linewidth=0,
    )

    ax1.set_ylabel("Roll (rad)")
    ax1.set_title(
        f"Body Orientation Analysis | Roll Range: {roll_amp:.3f} rad",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True)

    # --- Pitch ---
    plot_smooth_line(
        ax2, time_axis, data["pitch"], COLOR_PITCH, "Pitch Angle", smoothing
    )

    ax2.axhline(0.0, color=COLOR_ZERO, linestyle="--", label="Zero Ref", linewidth=1.5, alpha=0.8)
    ax2.fill_between(
        time_axis,
        data["pitch"].min(),
        data["pitch"].max(),
        color=COLOR_PITCH,
        alpha=0.03,
        linewidth=0,
    )

    ax2.set_ylabel("Pitch (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(
        f"Body Orientation Analysis | Pitch Range: {pitch_amp:.3f} rad",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{experiment_name}_3_oscillation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig3)
    
    # =================================================================
    # Fig 4: Gait Phase Analysis & External Force / 步态相位与外力分析
    # =================================================================
    # 修改：创建 2 行 1 列的图表，共享 X 轴
    # height_ratios=[2, 1] 让上面的步态图稍微高一点，或者 [1, 1] 等高，视喜好而定
    fig4, (ax_phase, ax_force) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # -------------------------------------------------------------------------
    # Subplot 1: Gait Phase Diagram (Contact Patterns) / 步态图
    # -------------------------------------------------------------------------

    # 1. 数据处理
    contact_threshold = 0.5
    is_contact_L = data["force_L"] > contact_threshold
    is_contact_R = data["force_R"] > contact_threshold

    # 2. 绘制步态图 (Gantt Chart Style)
    COLOR_L_FOOT = "#1f77b4"  # 蓝色
    COLOR_R_FOOT = "#ff7f0e"  # 橙色

    # --- 左脚 ---
    ax_phase.fill_between(
        time_axis,
        1.2,
        1.8,
        where=is_contact_L,
        color=COLOR_L_FOOT,
        alpha=0.8,
        label="Left Contact",
    )

    # --- 右脚 ---
    ax_phase.fill_between(
        time_axis,
        0.2,
        0.8,
        where=is_contact_R,
        color=COLOR_R_FOOT,
        alpha=0.8,
        label="Right Contact",
    )

    # 3. 装饰 Subplot 1
    ax_phase.set_yticks([0.5, 1.5])
    ax_phase.set_yticklabels(
        ["Right Foot", "Left Foot"], fontsize=12, fontweight="bold"
    )
    ax_phase.set_ylim(0, 2.0)
    ax_phase.set_title(
        f"Gait Phase & External Disturbance Analysis",
        fontsize=14,
        fontweight="bold",
    )
    ax_phase.grid(True, axis="x", linestyle="--", alpha=0.5)

    # 统计占空比
    total_steps = len(time_axis)
    duty_L = np.sum(is_contact_L) / total_steps * 100
    duty_R = np.sum(is_contact_R) / total_steps * 100

    ax_phase.text(
        time_axis.min(), 1.85, f"L Duty: {duty_L:.1f}%", color=COLOR_L_FOOT, fontsize=10
    )
    ax_phase.text(
        time_axis.min(), 0.85, f"R Duty: {duty_R:.1f}%", color=COLOR_R_FOOT, fontsize=10
    )

    # -------------------------------------------------------------------------
    # Subplot 2: External Force / 外力变化曲线
    # -------------------------------------------------------------------------

    COLOR_EXT_FORCE = "#d62728"  # 红色，代表警告/扰动

    # 绘制曲线
    # ax_force.plot(
    #     time_axis,
    #     data["external_force"],
    #     color=COLOR_EXT_FORCE,
    #     linewidth=1.5,
    #     label="Disturbance Force",
    # )
    
    plot_smooth_line(
        ax_force,
        time_axis,
        data["external_force"],
        COLOR_EXT_FORCE,
        "Disturbance Force (Smoothed)",
        0.0,
    )

    # # 填充颜色，让脉冲更明显
    # ax_force.fill_between(
    #     time_axis, data["external_force"], 0, color=COLOR_EXT_FORCE, alpha=0.2
    # )

    # 装饰 Subplot 2
    ax_force.set_ylabel("Ext Force (N)", fontsize=10, fontweight="bold")
    ax_force.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
    ax_force.grid(True, linestyle="--", alpha=0.5)
    ax_force.legend(loc="upper right")

    # 自动调整 Y 轴范围，留一点余量
    force_max = data["external_force"].max()
    if force_max > 0:
        ax_force.set_ylim(-1.0, force_max * 1.2)  # 稍微留点顶空
    else:
        ax_force.set_ylim(-1.0, 10.0)  # 默认范围

    # -------------------------------------------------------------------------
    # 保存与清理
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{experiment_name}_4_gait_phase_with_force.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig4)

    print(f"图表绘制完成。Smoothing系数: {smoothing}")


if __name__ == "__main__":
    if os.path.exists(csv_file_path):
        analyze_robot_data(csv_file_path)
    else:
        print("未找到文件")
