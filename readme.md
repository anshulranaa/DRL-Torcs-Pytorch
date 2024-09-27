# Deep Reinforcement Learning for Autonomous Driving in TORCS

This repository contains the implementation of a deep reinforcement learning (DRL) project focused on autonomous driving using The Open Racing Car Simulator (TORCS). We have analyzed and compared several DRL algorithms, including Deep Q-Networks (DQN), Deep Deterministic Policy Gradient (DDPG), Proximal Policy Optimization (PPO), and a novel hybrid algorithm combining DDPG and PPO.

## Project Overview

Autonomous driving is a critical area of research, and the use of DRL in simulation environments like TORCS offers a safe and efficient way to train AI agents to navigate complex road environments. In this project, we explore how DRL algorithms can be applied to improve the performance of autonomous vehicles in terms of distance covered, and we also test the robustness of our models by excluding brake sensory inputs.

### Key Algorithms Implemented:
- **Deep Q-Networks (DQN)**: Used as a baseline model to compare against more advanced DRL methods.
- **Deep Deterministic Policy Gradient (DDPG)**: A continuous control algorithm suited for steering and acceleration.
- **Proximal Policy Optimization (PPO)**: Known for its stability and sample efficiency.
- **Hybrid DDPG-PPO**: Combines the strengths of DDPG and PPO to achieve improved performance and stability.

## Problem Statement

Autonomous vehicles require advanced AI systems capable of navigating challenging environments while ensuring safety and efficiency. Real-world testing of such systems is both risky and costly, making simulation environments like TORCS an attractive alternative for training and evaluation. This project implements and compares different DRL algorithms to optimize vehicle control in TORCS, with a particular focus on maximizing the distance covered on the track.

## Approach

The project was structured as follows:

1. **Environment**: We use the `gym_torcs` module in Python to interact with the TORCS environment, providing sensory inputs such as track position, speed, and steering angles.
2. **Actions**: The agent controls the car using three actions: acceleration, braking, and steering.
3. **Reward Function**: The reward is defined by the TORCS environment and is based on the vehicle’s velocity along the track.
4. **Training**: Each algorithm is trained for 50 episodes due to resource constraints, and we compare performance across two scenarios: with and without brake input.

### Sensory Inputs
- Angle of the car relative to the track
- Position on the track
- Distance between the car and track edges
- Speed (x, y, z directions)
- Wheel spin velocity

### Evaluation Metrics
- **Distance Covered**: The primary metric for comparing the algorithms.
- **Robustness Test**: We also tested the models by removing the brake input, simulating real-world scenarios where brakes may fail.

## Results

| Model                | Distance Covered (Track) | Distance Covered (Without Brakes) |
|----------------------|--------------------------|-----------------------------------|
| DQN (Baseline)        | 33.03m                   | 33.11m                            |
| DDPG                 | 40.02m                   | 36.53m                            |
| PPO                  | 81.25m                   | 60.47m                            |
| DDPG-PPO Hybrid      | 45.45m                   | 42.48m                            |

From the results, PPO outperformed the other models in both scenarios due to its stability and efficient learning process. Our hybrid DDPG-PPO algorithm showed promising results by outperforming DQN and DDPG.

## Conclusions

Our project demonstrates the effectiveness of DRL techniques in autonomous driving simulations. While DDPG and PPO individually provide significant improvements over traditional methods, our hybrid DDPG-PPO algorithm presents a promising avenue for future research. The exclusion of the brake input showed that the models could still perform well, simulating real-world challenges such as brake failure.

### Future Work
- Increase the number of training episodes for better model convergence.
- Explore additional scenarios like different racetracks, weather conditions, and traffic situations to further test the robustness of the algorithms.

## Repository Contents
- **`dqn.py`**: Implementation of the Deep Q-Network.
- **`ddpg.py`**: Implementation of the Deep Deterministic Policy Gradient.
- **`ppo.py`**: Implementation of the Proximal Policy Optimization.
- **`ddpg_ppo.py`**: Implementation of the Hybrid DDPG-PPO algorithm.

