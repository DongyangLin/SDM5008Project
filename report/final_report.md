
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

**Tips for Success:**
*   **Write for a Smart Audience:** Assume the reader is technical but not an expert on your specific project.
*   **Be Clear and Concise:** Avoid jargon without explanation. Use simple, direct language.
*   **Visuals are Key:** Use diagrams, charts, and screenshots to break up text and improve understanding.
*   **Proofread:** Spelling and grammar errors reduce credibility. Read it aloud to catch mistakes.
*   **Be Honest:** Discussing challenges and limitations shows critical thinking and strengthens your report.