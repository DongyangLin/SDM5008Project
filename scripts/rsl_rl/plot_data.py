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
COLOR_BLUE = "#1973C2"
COLOR_GREEN = "#00B945"
COLOR_ORANGE = "#FF9500"
COLOR_RED = "#FA0101FF"
COLOR_PURPLE = "#9B26AF"


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
        color=COLOR_RED,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    # 实际值应用平滑
    plot_smooth_line(ax1, time_axis, data["act_vx"], COLOR_BLUE, "Act Vx", smoothing)

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
        color=COLOR_RED,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    plot_smooth_line(ax2, time_axis, data["act_vy"], COLOR_GREEN, "Act Vy", smoothing)

    ax2.set_ylabel("Velocity Y (m/s)")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray")
    ax2.grid(True)

    # --- Wz ---
    ax3.plot(
        time_axis,
        data["cmd_wz"],
        color=COLOR_RED,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )
    plot_smooth_line(ax3, time_axis, data["act_wz"], COLOR_ORANGE, "Act Wz", smoothing)

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
    plot_fitted_gaussian(ax1, err_vx, COLOR_BLUE, mse_vx, "Error $v_x$")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Error $v_x$ (m/s)")
    ax1.set_title(
        f"Error Distribution $v_x$ (MSE: {mse_vx:.1e})", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Vy ---
    plot_fitted_gaussian(ax2, err_vy, COLOR_GREEN, mse_vy, "Error $v_y$")
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Error $v_y$ (m/s)")
    ax2.set_title(
        f"Error Distribution $v_y$ (MSE: {mse_vy:.1e})", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # --- Wz ---
    plot_fitted_gaussian(ax3, err_wz, COLOR_ORANGE, mse_wz, "Error $\omega_z$")
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
    plot_smooth_line(ax1, time_axis, data["roll"], COLOR_GREEN, "Roll Angle", smoothing)

    # 0.0 基准线 (使用专门的 COLOR_BLUE 珊瑚色，更美观)
    ax1.axhline(
        0.0,
        color=COLOR_BLUE,
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
        color=COLOR_GREEN,
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
        ax2, time_axis, data["pitch"], COLOR_ORANGE, "Pitch Angle", smoothing
    )

    ax2.axhline(0.0, color=COLOR_BLUE, linestyle="--", label="Zero Ref", linewidth=1.5, alpha=0.8)
    ax2.fill_between(
        time_axis,
        data["pitch"].min(),
        data["pitch"].max(),
        color=COLOR_ORANGE,
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
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # -------------------------------------------------------------------------
    # Subplot 1: Gait Phase Diagram (Contact Patterns) / 步态图
    # -------------------------------------------------------------------------

    # 1. 数据处理
    contact_threshold = 0.5
    is_contact_L = data["force_L"] > contact_threshold
    is_contact_R = data["force_R"] > contact_threshold

    # 计算上升沿事件 (非接触 -> 接触) 的时间点
    left_on_idx = np.where(np.diff(is_contact_L.astype(int)) == 1)[0] + 1
    right_on_idx = np.where(np.diff(is_contact_R.astype(int)) == 1)[0] + 1

    left_on_times = (
        time_axis.iloc[left_on_idx].values if len(left_on_idx) > 0 else np.array([])
    )
    right_on_times = (
        time_axis.iloc[right_on_idx].values if len(right_on_idx) > 0 else np.array([])
    )

    # 若左脚事件不足两次则无法定义周期
    phase_times = []
    phase_deg = []
    if len(left_on_times) > 1 and len(right_on_times) > 0:
        # 对每个左脚周期 [L_i, L_{i+1}) 寻找区间内的第一个右脚触地事件
        for i in range(len(left_on_times) - 1):
            t0 = left_on_times[i]
            t1 = left_on_times[i + 1]
            period = t1 - t0
            if period <= 0:
                continue
            # 在区间内的右脚事件索引
            mask = (right_on_times >= t0) & (right_on_times < t1)
            if np.any(mask):
                t_right = right_on_times[np.where(mask)[0][0]]
                frac = (t_right - t0) / period
                phase_times.append((t0 + t1) / 2.0)
                phase_deg.append(frac * 360.0)
            else:
                # 如果区间内没有右脚事件，记录 nan 以便绘图间断
                phase_times.append((t0 + t1) / 2.0)
                phase_deg.append(np.nan)

    if len(phase_times) > 0:
        ax_phase.plot(
            phase_times,
            phase_deg,
            "-o",
            color=COLOR_BLUE,
            markersize=4,
            linewidth=1.5,
            label="Phase (deg)",
        )
        # 绘制 180° 参考线（理想交替对应 180°）
        ax_phase.axhline(
            180.0,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="180° (alternation)",
        )
        mean_phase = np.nanmean(phase_deg)
        std_phase = np.nanstd(phase_deg)
        ax_phase.text(
            phase_times[0],
            350,
            f"Mean: {mean_phase:.1f}°, STD: {std_phase:.1f}°",
            fontsize=10,
            fontweight="bold",
        )
    else:
        ax_phase.text(0.5, 0.5, "No valid cycles/events to compute phase", ha="center")

    ax_phase.set_ylim(-10, 370)
    ax_phase.set_yticks([0, 90, 180, 270, 360])
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_title(
        "Gait Phase Difference (Right relative to Left)", fontweight="bold"
    )
    ax_phase.grid(True, linestyle="--", alpha=0.4)
    ax_phase.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Subplot 2: External Force / 外力变化曲线
    # -------------------------------------------------------------------------

    plot_smooth_line(
        ax_force,
        time_axis,
        data["external_force"],
        COLOR_RED,
        "Disturbance Force",
        0.0,
    )

    # 装饰 Subplot 2
    ax_force.set_ylabel("Ext Force (N)", fontsize=10)
    ax_force.set_xlabel("Time (s)", fontsize=10)
    ax_force.grid(True, linestyle="--", alpha=0.5)
    ax_force.legend(loc="upper right")
    ax_force.set_title("External Disturbance Force over Time", fontweight="bold")

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
