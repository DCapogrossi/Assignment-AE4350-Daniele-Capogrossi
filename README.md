# Reinforcement Learning for Minigolf  
**AE4350 – Bio-Inspired Intelligence for Aerospace Applications**  
**Student:** Daniele Capogrossi  
**Student ID:** 6355412  

## 📘 Overview  
This project implements a Reinforcement Learning (RL) agent trained to play a 2D minigolf game in a simulated environment. The environment features realistic elements such as water hazards, walls, and continuous position tracking. The agent learns using Q-Learning with function approximation via tile coding, and is further extended with SARSA(λ) and eligibility traces.

## 🧠 Algorithms Implemented
- **Q-Learning** (off-policy TD control)
- **SARSA(λ)** (on-policy TD control with eligibility traces)
- **Tile Coding** (linear function approximation for continuous states)
- **ε-greedy exploration** with exponential decay
- **Reward shaping** for performance tuning

## 🕹 Environment Features
- Continuous 2D state space
- Discrete action space:
  - 36 directions (10° increments)
  - 14 non-uniform force magnitudes
- Water hazards and walls
- Reward shaping and penalties
- Episode termination on goal or hazard collision
