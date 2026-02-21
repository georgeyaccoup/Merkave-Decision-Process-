# Reinforcement Learning Learning Repo (Bandits + Dynamic Programming for Robotics)

This repository is a learning-focused set of small Reinforcement Learning projects designed to help you move from **reading theory** to **implementing complete solutions** with clear, step-by-step reasoning.

It includes:
- **Dynamic Programming (DP)** on a **robotics navigation MDP** (Value Iteration + Policy Iteration).
- **Multi-Armed Bandits** experiments (Greedy / Epsilon-Greedy) using an RL-Glue style setup.

---

## Contents (in this one repo)

### 1) Robotics DP Tutorial — Grid Navigation as an MDP
A terminal-guided tutorial script that teaches you how to build and solve an MDP using DP.

**Robotics story:** A mobile robot navigates a warehouse-like grid to reach a charging station. Motion is noisy (slip), so the transition model is probabilistic.

**You will learn:**
- How to define an MDP model: **S, A, P, R, γ**
- How to build a transition table `P[s][a]`
- How to compute:
  - **Value Iteration** → `V*` then greedy policy `π*`
  - **Policy Iteration** → (Policy Evaluation + Policy Improvement) until stable

**Typical output:**
- A grid of values `V(s)` (higher value = “better to be here”)
- An arrow grid policy (U/R/D/L) showing what the robot should do in each cell

---

### 2) Multi-Armed Bandits — Exploration vs Exploitation
A bandit implementation that demonstrates why exploration matters.

**You will learn:**
- How a bandit differs from an MDP (no next-state dynamics, only reward feedback)
- How to implement:
  - **Greedy** action selection
  - **Epsilon-greedy** exploration
- How to run many trials/steps and compare average reward and best-action selection

> Note: The bandits code uses an RL-Glue style structure and imports RL-Glue / environment modules. Make sure those dependencies (or the course-provided files) are available when running.

---

## Requirements

### DP Robotics tutorial
- Python 3.8+  
- No special packages required (pure Python)

### Bandits experiments
- Python 3.8+
- Common packages:
  ```bash
  pip install numpy matplotlib tqdm
