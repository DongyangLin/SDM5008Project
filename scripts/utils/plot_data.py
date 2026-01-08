import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import colorsys
import matplotlib.colors as mcolors

# 配置 / Config
csv_file_path = (
    # "/logs/rsl_rl/pf_tron_1a_flat/2025-12-15_16-38-07/play_logs/Flat_play_log_mean.csv"
    # "logs/rsl_rl/pf_pim_stair/2025-12-17_09-56-22/play_logs/play_log_rough.csv"
    "logs/rsl_rl/pf_him_stair/2025-12-16_Stable_Phase_3/play_logs/play_log_slope.csv"
)
experiment_name = "him_slope"

# 绘图风格（学术） / Plot style (academic)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["font.size"] = 12

# 颜色 (Hex) / Colors (Hex)
COLOR_BLUE = "#1973C2"
COLOR_GREEN = "#00B945"
COLOR_ORANGE = "#FF9500"
COLOR_RED = "#FA0101FF"
COLOR_PURPLE = "#9B26AF"


def get_enhanced_color(hex_color, sat_factor=1.2, val_factor=0.85):
    """获取增强颜色（饱和度/亮度调整）。
    Get enhanced color (saturation/value adjust).
    Args:
        sat_factor: saturation multiplier (>1 increases)
        val_factor: value multiplier (<1 darkens)
    """
    rgb = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(*rgb)

    # 调整饱和度和亮度 / adjust saturation/value
    s = min(1.0, s * sat_factor)
    v = max(0.0, v * val_factor)

    return mcolors.to_hex(colorsys.hsv_to_rgb(h, s, v))


def tensorboard_smoothing(scalars, weight=0.6):
    """TensorBoard 风格的指数移动平均平滑。
    TensorBoard-style EMA smoothing.
    Args:
        scalars: input list/array
        weight: smoothing weight in (0,1)
    Returns:
        numpy array of smoothed values
    """
    if weight <= 0 or weight >= 1:
        return scalars

    last = scalars[0]
    smoothed = []
    for point in scalars:
        # EMA: S_t = weight*S_{t-1} + (1-weight)*Y_t
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def fix_roll_phase(roll_data):
    """处理角度解缠绕并在需要时平移 PI 使中心归零。
    Fix roll phase wrapping and center shift.
    """
    # 解缠绕 / unwrap angles
    roll_unwrapped = np.unwrap(roll_data, discont=np.pi)
    mean_val = np.mean(roll_unwrapped)
    if np.abs(mean_val) > 2.0:
        k = np.round(mean_val / np.pi)
        print(f"[Fix] Roll mean offset {mean_val:.2f}, shift {-k}*PI")
        return roll_unwrapped - k * np.pi
    return roll_unwrapped


def analyze_robot_data(file_path, smoothing=0.8):
    file_dir = os.path.dirname(os.path.abspath(file_path))
    save_dir = os.path.join(file_dir, "images")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Reading file: {file_path}")
    # 读取数据 / Read data：取前3000行以避免过大文件
    data = pd.read_csv(file_path, nrows=3001)

    # =========================================================
    # 数据预处理：解缠绕 / Data preprocess: unwrap angles
    # 移除 -pi/pi 跳变以便连续绘图 / Remove -pi/pi jumps for continuous plots
    # =========================================================
    data["roll"] = fix_roll_phase(data["roll"].values)
    data["pitch"] = np.unwrap(data["pitch"].values, discont=np.pi)

    # 时间轴归零 / zero time axis
    time_axis = data["wall_time_s"] - data["wall_time_s"].iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    # 辅助：绘制原始与平滑 / helper: plot raw + smoothed
    def plot_smooth_line(ax, x, y, color, label, smooth_factor):
        if smooth_factor > 0:
            # 先画原始（淡），再画平滑 / draw raw (faint) then smoothed
            ax.plot(x, y, color=color, linestyle="-", linewidth=1.0, alpha=0.25)
            # 计算平滑值 / compute smoothed values
            y_smooth = tensorboard_smoothing(y.values, weight=smooth_factor)

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
    # 速度跟踪 / Chart 1: Velocity Tracking
    # =========================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- Vx ---
    # 指令虚线，实际值平滑 / Cmd as dashed, act smoothed
    ax1.plot(
        time_axis,
        data["cmd_vx"],
        color=COLOR_RED,
        linestyle="--",
        linewidth=1.5,
        label="Cmd (Ref)",
    )

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
    # 误差分布（正态拟合） / Chart 2: Error Distributions (Gaussian fit)
    # =========================================================

    # 计算误差 / compute errors
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
        # 绘制直方图并拟合正态 / Draw histogram and fitted Gaussian
        n, bins, patches = ax.hist(data, color=base_color, **hist_params)
        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x, mu, std)
        bin_width = bins[1] - bins[0]
        p_scaled = p * len(data) * bin_width
        curve_color = get_enhanced_color(base_color, sat_factor=1.2, val_factor=0.85)
        label_text = f"Fit ($\mu={mu:.3f}, \sigma={std:.3f}$)"
        ax.plot(x, p_scaled, color=curve_color, linestyle="--", linewidth=2.5, label=label_text)
        ax.set_title(f"{label_prefix} (MSE: {mse_val:.1e})", fontsize=12, fontweight="bold")
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
    # 姿态震荡 / Chart 3: Attitude Oscillation (Roll/Pitch)
    # =========================================================
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    roll_amp = data["roll"].max() - data["roll"].min()
    pitch_amp = data["pitch"].max() - data["pitch"].min()

    # --- Roll ---
    # 绘制平滑的 Roll / plot smoothed roll
    plot_smooth_line(ax1, time_axis, data["roll"], COLOR_GREEN, "Roll Angle", smoothing)
    # 0 基准线 / zero reference line
    ax1.axhline(
        0.0,
        color=COLOR_BLUE,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Zero Ref",
    )

    # 可选背景填充 / optional background fill
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
    # 步态相位与外力分析 / Fig 4: Gait Phase Analysis & External Force
    # =================================================================
    # fig4, (ax_phase, ax_force) = plt.subplots(
    #     2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    # )

    # # -------------------------------------------------------------------------
    # # 步态图 / Subplot 1: Gait Phase Diagram (Contact Patterns)
    # # -------------------------------------------------------------------------
    # # 接触检测 / Data: contact detection
    # contact_threshold = 0.5
    # is_contact_L = data["force_L"] > contact_threshold
    # is_contact_R = data["force_R"] > contact_threshold

    # # 上升沿索引（非接触->接触） / rising-edge indices (no->yes contact)
    # left_on_idx = np.where(np.diff(is_contact_L.astype(int)) == 1)[0] + 1
    # right_on_idx = np.where(np.diff(is_contact_R.astype(int)) == 1)[0] + 1

    # left_on_times = (
    #     time_axis.iloc[left_on_idx].values if len(left_on_idx) > 0 else np.array([])
    # )
    # right_on_times = (
    #     time_axis.iloc[right_on_idx].values if len(right_on_idx) > 0 else np.array([])
    # )

    # # 需要至少2次左脚触地以定义周期 / need >=2 left events to define cycles
    # phase_times = []
    # phase_deg = []
    # if len(left_on_times) > 1 and len(right_on_times) > 0:
    #     # 对每个左脚周期，找首个右脚触地 / For each left cycle [L_i, L_{i+1}), find first right on-event
    #     for i in range(len(left_on_times) - 1):
    #         t0 = left_on_times[i]
    #         t1 = left_on_times[i + 1]
    #         period = t1 - t0
    #         if period <= 0:
    #             continue
    #         # 区间内的右脚事件掩码 / mask of right events within cycle
    #         mask = (right_on_times >= t0) & (right_on_times < t1)
    #         if np.any(mask):
    #             t_right = right_on_times[np.where(mask)[0][0]]
    #             frac = (t_right - t0) / period
    #             phase_times.append((t0 + t1) / 2.0)
    #             phase_deg.append(frac * 360.0)
    #         else:
    #             # 无事件则记录 NaN / record NaN if no right event in this cycle
    #             phase_times.append((t0 + t1) / 2.0)
    #             phase_deg.append(np.nan)

    # if len(phase_times) > 0:
    #     ax_phase.plot(
    #         phase_times,
    #         phase_deg,
    #         "-o",
    #         color=COLOR_BLUE,
    #         markersize=4,
    #         linewidth=1.5,
    #         label="Phase (deg)",
    #     )

    #     ax_phase.axhline(
    #         180.0,
    #         color="gray",
    #         linestyle="--",
    #         linewidth=1.0,
    #         alpha=0.7,
    #         label="180° (alternation)",
    #     )
    #     mean_phase = np.nanmean(phase_deg)
    #     std_phase = np.nanstd(phase_deg)
    #     ax_phase.text(
    #         phase_times[0],
    #         350,
    #         f"Mean: {mean_phase:.1f}°, STD: {std_phase:.1f}°",
    #         fontsize=10,
    #         fontweight="bold",
    #     )
    # else:
    #     ax_phase.text(0.5, 0.5, "No valid cycles/events to compute phase", ha="center")

    # ax_phase.set_ylim(-10, 370)
    # ax_phase.set_yticks([0, 90, 180, 270, 360])
    # ax_phase.set_ylabel("Phase (degrees)")
    # ax_phase.set_title(
    #     "Gait Phase Difference (Right relative to Left)", fontweight="bold"
    # )
    # ax_phase.grid(True, linestyle="--", alpha=0.4)
    # ax_phase.legend(loc="upper right")

    # # -------------------------------------------------------------------------
    # # 外力曲线 / Subplot 2: External Force
    # # -------------------------------------------------------------------------

    # plot_smooth_line(
    #     ax_force,
    #     time_axis,
    #     data["force"],
    #     COLOR_RED,
    #     "Disturbance Force",
    #     0.0,
    # )
    
    # ax_force.set_ylabel("Ext Force (N)", fontsize=10)
    # ax_force.set_xlabel("Time (s)", fontsize=10)
    # ax_force.grid(True, linestyle="--", alpha=0.5)
    # ax_force.legend(loc="upper right")
    # ax_force.set_title("External Disturbance Force over Time", fontweight="bold")

    # force_max = data["force"].max()
    # if force_max > 0:
    #     ax_force.set_ylim(-1.0, force_max * 1.2)
    # else:
    #     ax_force.set_ylim(-1.0, 10.0)

    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(save_dir, f"{experiment_name}_4_gait_phase_with_force.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close(fig4)

    # print(f"Plots saved. Smoothing: {smoothing}")


if __name__ == "__main__":
    if os.path.exists(csv_file_path):
        analyze_robot_data(csv_file_path)
    else:
        print("CSV file not found")
