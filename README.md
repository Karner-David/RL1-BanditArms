# Multi-Armed Bandit Exploration Strategies

This project implements and compares several action-selection strategies for solving the **K-armed Gaussian bandit** problem. It is designed as an educational reinforcement learning project, walking through classic exploration-vs-exploitation approaches.

## 📌 Algorithms Implemented

Each agent chooses among `K` arms (bandit levers), trying to maximize rewards over time.

- **Epsilon-Greedy**: Chooses the best known action most of the time, but explores randomly with probability ε.
- **Optimistic Greedy**: Starts with high initial estimates to encourage exploration without randomness.
- **Softmax (Boltzmann Exploration)**: Selects actions probabilistically based on value estimates using a temperature parameter τ.

## 📁 Project Structure

```
bandit_rl/
├── agents.py         # Agent classes (EpsilonGreedyAgent, OptimisticGreedyAgent, SoftmaxAgent)
├── bandit.py         # GaussianBandit environment
├── experiment.py     # run_experiment function and evaluation metrics
├── plot.py           # plot_results function for visualizing performance
├── main.py           # Example runner script to compare agents
└── README.md         # This file
```

## 📊 Visual Outputs

Generated after running `main.py`:
- ![Average Reward](avg_reward.png)
- ![Optimal Action Percentage](pct_opt.png)
- ![Cumulative Regret](regret.png)

## 📚 Things I learned

This project helped solidify key concepts in reinforcement learning:
- Exploration: this is important because you want the model to explore possible options that may be better than the current options, however too much exploration leads to underfitting
- Exploitation: this is important because you want the model to find the best option from the available options, but too much of this may lead to overfitting and not exploring an option that may be better.
- The different Algorithms to find the best arm to pull and the logic supporting them.
---

