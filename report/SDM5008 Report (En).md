
# **SDM5008 Final Project Report** 

--- 

**Course:** SDM5008 – Advanced Robot Control 
**Project Title:** RL-based Locomotion Control for a Point-Foot Biped in Isaac Lab 
**Platform:** NVIDIA Isaac Lab (Isaac Sim 4.5.0 + Isaac Lab 2.1.0) 

**Submitted By:** 
* **Name:** `Zikun Zhuang` `Zhiyu Wang` 
* **Student ID:** `12532840` `12532838` 
* **Email:** `[12532840@mail.sustech.edu.cn]` `[12532838@mail.sustech.edu.cn]` 

**Date of Submission:** `[January 8, 2026]`

---

## Project Overview

This project is built on the NVIDIA Isaac Lab simulation platform and aims to provide an efficient reinforcement learning–based locomotion control framework for the TRON1 bipedal robot developed by LimX Dynamics. The core contributions of this project include:

1. **Highly Robust Locomotion Control**: Achieves stable walking and accurate velocity tracking for the TRON1 robot on both flat ground and unstructured, complex terrains.
2. **Reproduction and Integration of State-of-the-Art Algorithms**: Successfully reproduces and deploys advanced reinforcement learning algorithms such as [**HIM (Hybrid Internal Model)**](https://arxiv.org/abs/2411.14386) [^1] and [**PIM (Perceptive Internal Model)**](https://arxiv.org/abs/2312.11460) [^2] on the TRON1 platform using Isaac Sim.
3. **Systematic Performance Evaluation**: Conducts comprehensive comparative experiments to analyze the performance differences among HIM, PIM, and the baseline **Encoder-MLP** algorithm across various task scenarios, providing detailed empirical evidence to support algorithm selection for bipedal robots.

The core implementation of this project is based on LimX Dynamics’ open-source repository: [**TRON1 Reinforcement Learning on Isaac Lab**](https://github.com/limxdynamics/tron1-rl-isaaclab).

**Keywords:** Isaac Lab, TRON1, Bipedal Locomotion, Reinforcement Learning, PPO, HIM, PIM, Robust Control.


---

## 1. Code Review & Architecture

This project is built upon NVIDIA Isaac Lab’s `ManagerBasedRLEnv` and adopts a highly modular, configuration-driven design. The entire reinforcement learning environment is decoupled into four core managers: Scene, Observation, Reward, and Action.

### 1.1 Scene Configuration

**Files**: `limx_base_env_cfg.py`, `pointfoot_cfg.py`, `terrains_cfg.py`

The scene configuration module is responsible for defining the physical entities and environmental properties of the simulation. In this project, this is mainly reflected in the `PFSceneCfg` class and its referenced asset configurations.

#### 1.1.1 USD Asset Loading and Physical Properties

In `exts/bipedal_locomotion/bipedal_locomotion/assets/config/pointfoot_cfg.py`, `POINTFOOT_CFG` defines the robot’s asset properties.

* **USD Path**: The code loads the Universal Scene Description (USD) file located at `../usd/PF_TRON1A/PF_TRON1A.usd` via `sim_utils.UsdFileCfg`, which serves as the foundation for the robot’s geometric and physical models.
* **Rigid Body and Joint Properties**: The configuration explicitly defines rigid body properties (`RigidBodyPropertiesCfg`), such as enabling self-collisions (`enabled_self_collisions=True`) and solver iteration counts, to ensure the stability of the physical simulation.
* **Initial State**: `init_state` specifies the default joint angles (`joint_pos`) and base position at robot spawn time, providing a consistent reset state for reinforcement learning.

#### 1.1.2 Scene Integration

In the `PFSceneCfg` class within `limx_base_env_cfg.py`, the robot asset is integrated into the interactive scene. This class also configures terrain (`TerrainImporterCfg`), lighting (`DomeLightCfg`), and sensors (e.g., `ContactSensorCfg` for foot contact detection).

- **Robot Asset**:

  - Loaded via `ArticulationCfg` using the USD file (`PF_TRON1A.usd`).

  - **Joint Properties**:

    | **Joint Name** | **Damping** | **Stiffness** | **Initial Pos** | **Remarks** |
    | -------------- | ----------- | ------------- | --------------- | ----------- |
    | `abad_L_Joint` | 2.5         | 40.0          | 0.0             | Left hip abduction joint |
    | `abad_R_Joint` | 2.5         | 40.0          | 0.0             | Right hip abduction joint |
    | `hip_L_Joint`  | 2.5         | 40.0          | 0.0             | Left hip joint |
    | `hip_R_Joint`  | 2.5         | 40.0          | 0.0             | Right hip joint |
    | `knee_L_Joint` | 2.5         | 40.0          | 0.0             | Left knee joint |
    | `knee_R_Joint` | 2.5         | 40.0          | 0.0             | Right knee joint |
    | `foot_L_Joint` | N/A*        | N/A*          | 0.0             | - |
    | `foot_R_Joint` | N/A*        | N/A*          | 0.0             | - |

  - **Physical Properties**:

    | **Property Name** | **Value** | **Description** |
    | ----------------- | --------- | --------------- |
    | `enabled_self_collisions` | `True` | Enables self-collision detection to prevent mesh interpenetration |
    | `linear_damping`          | `0.0`  | Linear damping coefficient |
    | `angular_damping`         | `0.0`  | Angular damping coefficient |
    | `max_linear_velocity`     | `1000.0` | Maximum linear velocity limit |
    | `max_angular_velocity`    | `1000.0` | Maximum angular velocity limit |
    | `solver_position_iter`    | `4`    | Position solver iteration count |
    | `solver_velocity_iter`    | `4`    | Velocity solver iteration count |

- **Terrain**:

  - Imported via `TerrainImporterCfg`.
  - Supports multiple terrain generators: `BLIND_ROUGH_TERRAINS_CFG` (waves, grids, rough noise) and `STAIRS_TERRAINS_CFG` (pyramid stairs).

- **Sensors**:

  - **Contact Sensor**: Defined in `PFSceneCfg`, used to detect foot contact forces, with a sampling frequency aligned with the physics simulation timestep.
  - **Ray Caster (Height Scanner)**: Enabled only in HIM/PIM configurations, used to scan terrain height as privileged input to the Critic.




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


---

## **References**

[^1]:Long, J., Ren, J., Shi, M., Wang, Z., Huang, T., Luo, P., & Pang, J. (2024). **Learning Humanoid Locomotion with Perceptive Internal Model**. *arXiv preprint arXiv:2411.14386*. https://arxiv.org/abs/2411.14386
[^2]:Long, J., Wang, Z., Li, Q., Gao, J., Cao, L., & Pang, J. (2024). **Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response**. *The Twelfth International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2312.11460
[^3]:Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv, 2017.
[^4]:NVIDIA Isaac Lab Documentation.
[^5]:Relevant course and tutorial materials.