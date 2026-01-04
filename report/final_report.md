
# **SDM5008 Final Project Report**

---

**Course:** SDM5008 – Advanced Robot Control
**Project Title:** RL-based Locomotion Control for a Point-Foot Biped in Isaac Lab
**Platform:** NVIDIA Isaac Lab (Isaac Sim 4.5.0 + Isaac Lab 2.1.0)

**Submitted By:**
*   **Name:** `Zhang Haoran`
*   **Student ID:** `12413024`
*   **Email:** `[12413024@mail.susutech.edu.cn]`

**Date of Submission:** `[January 8, 2026]`

---

### **Abstract**

This project investigates reinforcement learning–based locomotion control for a point-foot biped robot using NVIDIA Isaac Lab. Two control architectures, namely the **Hybrid Internal Model (HIM)** and the **Perceptive Internal Model (PIM)**, are implemented, trained, and evaluated under identical simulation conditions. The policies are trained using **Proximal Policy Optimization (PPO)** with GPU-parallelized simulation. We evaluate performance on flat-ground velocity tracking, disturbance rejection, and terrain generalization. Experimental results demonstrate that both HIM and PIM achieve stable locomotion, while exhibiting distinct trade-offs in tracking accuracy, robustness, and recovery behavior under disturbances.

**Keywords:** 

---

### **1. Introduction**

Bipedal locomotion remains a fundamental challenge in legged robotics due to hybrid dynamics, underactuation, and strong sensitivity to disturbances. Recent advances in reinforcement learning (RL), combined with high-fidelity simulators such as NVIDIA Isaac Lab, have enabled the training of robust locomotion policies through large-scale parallel simulation.

In this project, we focus on a simplified **point-foot biped** model and study RL-based velocity tracking control. The objectives are threefold: (i) to gain a deep understanding of the Isaac Lab software architecture, (ii) to design reward functions and observation spaces that yield stable and robust walking, and (iii) to compare two policy structures—HIM and PIM—under identical evaluation protocols.

---

### **2. Isaac Lab Framework Overview**

#### **2.1 Scene Configuration**

The scene is configured using Isaac Lab’s modular configuration system. The point-foot biped is loaded as a USD articulation asset, with:

- Rigid body definitions for the base and legs,

- Actuated hip and knee joints,

- A floating base with 6-DoF dynamics.

The simulation runs with GPU acceleration, allowing thousands of parallel environments for efficient data collection.

#### **2.2 Observation Manager**

The observation vector includes:

- Base linear and angular velocity (in the body frame),

- Projected gravity vector,

- Joint positions and velocities,

- Previous action,

- Commanded velocity $(v_x, v_y, \omega_z)$.

Gaussian noise is injected into selected observations to improve robustness and sim-to-real generalization.

#### **2.3 Action Manager**

Actions are defined as joint position targets for a PD controller. The RL policy outputs normalized action commands, which are scaled and applied as desired joint offsets relative to a nominal pose.

#### **2.4 Reward Manager**

The reward function is composed of weighted terms, including:

- Velocity tracking reward,

- Orientation and height stability penalties,

- Energy and action smoothness penalties,

- Contact and foot-slip regularization.

Careful tuning of reward weights is crucial for achieving stable gaits.

---

### **3. Control Architectures: HIM and PIM**

#### **3.1 Hybrid Internal Model (HIM)**

HIM adopts a hierarchical structure, where:

- A high-level module processes command and global state information,

- A low-level module generates joint-level actions.

This structure encourages temporal abstraction and can improve robustness under large disturbances.

#### **3.2 Perceptive Internal Model (PIM)**

PIM uses a single end-to-end policy that directly maps observations to actions. The architecture is simpler and often yields faster convergence, but may be less structured in handling complex disturbances.

Both policies are trained using PPO with identical hyperparameters for a fair comparison.

---

### **4. Training Setup**

- **Algorithm:** Proximal Policy Optimization (PPO)

- **Parallel Environments:** O(10^3)

- **Simulation Time Step:** Δt=0.005s

- **Policy Update Frequency:** Fixed horizon rollouts

- **Domain Randomization:** Mass, friction, and external force perturbations

Training progress is monitored using reward curves and episode length statistics.

---

### **5. Experimental Results**

#### **5.1 Flat Ground Velocity Tracking**

Both HIM and PIM successfully track commanded velocities over long horizons. Quantitatively:

- PIM achieves lower steady-state velocity tracking error,

- HIM exhibits smoother base motion and reduced oscillations.

#### **5.2 Disturbance Rejection**

External impulses are applied to the robot base during walking:

- HIM tolerates larger impulse magnitudes without falling,

- PIM recovers faster for small disturbances but fails earlier under large pushes.

#### **5.3 Terrain Traversal**

On mixed terrains (slopes and steps):

- HIM shows better adaptability during terrain transitions,

- PIM maintains higher speed on flat segments but is more sensitive to sudden height changes.

---

### **6. Discussion**

The comparative study highlights a trade-off between performance and robustness. HIM benefits from hierarchical structure, which improves stability and disturbance rejection, while PIM excels in simplicity and tracking accuracy under nominal conditions. These observations suggest that architectural priors play an important role in RL-based locomotion.

---

### **7. Conclusion**

This project demonstrates the effectiveness of reinforcement learning for biped locomotion in Isaac Lab. By implementing and comparing HIM and PIM, we gain insights into how policy structure affects robustness and performance. Future work may include sim-to-real transfer, more expressive motions, and integration with model-based safety constraints.

### **References**

[1] Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv, 2017.
[2] NVIDIA Isaac Lab Documentation.
[3] Relevant course and tutorial materials.