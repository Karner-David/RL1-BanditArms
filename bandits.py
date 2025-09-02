#!/usr/bin/env python3
"""
Project 1 — Multi-Armed Bandits
Implements a stationary 10-arm Gaussian bandit and three agents:
  1) EpsilonGreedy (sample-average updates)
  2) OptimisticGreedy (greedy with optimistic Q0, α via sample-average)
  3) SoftmaxAgent (Boltzmann exploration)
Produces plots with 95% CIs for:
  - average reward
  - % optimal action
  - cumulative regret
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------ Bandit ------------------
class GaussianBandit:
    def __init__(self, K=10, sigma=1.0):
        # hidden true means
        self.means = np.random.normal(0, 1, K)
        self.sigma = sigma
        self.opt_action = np.argmax(self.means)
        self.opt_mean = np.max(self.means)
        self.K = K

    def step(self, a):
        """Return reward sampled from arm a."""
        return np.random.normal(self.means[a], self.sigma)


# ------------------ Agents ------------------
class BaseAgent:
    def __init__(self, K):
        self.K = K
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.K)
        self.N = np.zeros(self.K)

    def select_action(self, t):
        raise NotImplementedError

    def update(self, a, r):
        """Sample-average update"""
        self.N[a] += 1
        lr = 1 / self.N[a]
        self.Q[a] += (lr * (r - self.Q[a]))

class EpsilonGreedy(BaseAgent):
    def __init__(self, K, epsilon=0.1):
        super().__init__(K)
        self.eps = epsilon

    def select_action(self, t):
        # TODO: epsilon-greedy selection
        rand_val = np.random.rand()
        if rand_val < self.eps:
            return np.random.randint(0, self.K)
        else:
            # have to get the arm with the highest Q but have to make it random so it doesn't bias lower indexed arms
            max_Q = np.max(self.Q)
            max_indxs = np.where(self.Q == max_Q)[0]
            return np.random.choice(max_indxs)

class OptimisticGreedy(BaseAgent):
    def __init__(self, K, q0=3.0):
        super().__init__(K)
        self.Q[:] = q0

    def select_action(self, t):
        # TODO: pick greedy action (argmax Q)
        max_Q = np.max(self.Q)
        max_indxs = np.where(self.Q == max_Q)[0]
        return np.random.choice(max_indxs)

class SoftmaxAgent(BaseAgent):
    def __init__(self, K, tau=0.1):
        super().__init__(K)
        self.tau = tau

    def select_action(self, t):
        # TODO: implement Boltzmann exploration
        logits = self.Q / self.tau
        max_logit = np.max(logits)
        logits = logits - max_logit
        probs = np.exp(logits) / np.sum(np.exp(logits))
        rand = np.random.uniform()
        for i, prob in enumerate(probs):
            rand -= prob
            if rand <= 0:
                return i
        return self.K - 1

# ------------------ Experiment ------------------
def run_experiment(agent_class, agent_kwargs, num_runs=2000, T=1000, K=10):
    rewards = np.zeros((num_runs, T))
    optimal = np.zeros((num_runs, T))
    regret = np.zeros((num_runs, T))

    for run in range(num_runs):
        bandit = GaussianBandit(K=K)
        agent = agent_class(K, **agent_kwargs)
        agent.reset()

        cum_regret = 0
        for t in range(T):
            action = agent.select_action(t)
            reward = bandit.step(action)
            agent.update(action, reward)

            rewards[run, t] = reward
            optimal[run, t] = 1 if action == bandit.opt_action else 0
            cum_regret += bandit.opt_mean - bandit.means[action]
            regret[run, t] = cum_regret

    def mean_ci(x, axis=0):
        m = x.mean(axis=axis)
        se = x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
        lo = m - 1.96 * se
        hi = m + 1.96 * se
        return m, lo, hi
    
    avg_reward = mean_ci(rewards)
    avg_optimal = mean_ci(optimal)
    avg_regret = mean_ci(regret)
    
    return avg_reward, avg_optimal, avg_regret

def plot_results(xs, results, outdir="plots"):
    """
    xs: 1D array of length T (e.g., np.arange(T))
    results: list of dicts, one per agent, each with:
      {
        "label": "ε-greedy (ε=0.1)",
        "avg_reward": (mean, lo, hi),   # each length T
        "avg_optimal": (mean, lo, hi),  # each length T
        "avg_regret": (mean, lo, hi),   # each length T
      }
    outdir: folder to save PNGs
    """
    os.makedirs(outdir, exist_ok=True)

    def _panel(metric_key, ylabel, title, filename, clamp01=False):
        plt.figure(figsize=(8, 5))
        for res in results:
            mean, lo, hi = res[metric_key]
            if clamp01:
                lo = np.clip(lo, 0.0, 1.0)
                hi = np.clip(hi, 0.0, 1.0)
            plt.plot(xs, mean, label=res["label"])
            plt.fill_between(xs, lo, hi, alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=160)
        plt.close()

    _panel("avg_reward",  "Average reward",
           "Average Reward vs. Time",          "avg_reward.png")

    _panel("avg_optimal", "Probability of optimal action",
           "Probability of Choosing the Optimal Arm", "pct_opt.png",
           clamp01=True)

    _panel("avg_regret",  "Cumulative regret",
           "Cumulative Regret vs. Time",       "regret.png")

# ------------------ Main ------------------
if __name__ == "__main__":
    # Example: run epsilon-greedy

    # TODO: plot
    T = 1000
    xs = np.arange(T)

    results = []

    # ε-greedy
    avg_reward, avg_optimal, avg_regret = run_experiment(EpsilonGreedy, {"epsilon": 0.1}, T=T)
    results.append({
        "label": "ε-greedy (ε=0.1)",
        "avg_reward": avg_reward,
        "avg_optimal": avg_optimal,
        "avg_regret": avg_regret,
    })

    # Optimistic greedy
    avg_reward, avg_optimal, avg_regret = run_experiment(OptimisticGreedy, {"q0": 1.0}, T=T)
    results.append({
        "label": "Optimistic (q0=1.0)",
        "avg_reward": avg_reward,
        "avg_optimal": avg_optimal,
        "avg_regret": avg_regret,
    })

    # Softmax
    avg_reward, avg_optimal, avg_regret = run_experiment(SoftmaxAgent, {"tau": 0.1}, T=T)
    results.append({
        "label": "Softmax (τ=0.1)",
        "avg_reward": avg_reward,
        "avg_optimal": avg_optimal,
        "avg_regret": avg_regret,
    })

    plot_results(xs, results, outdir="plots")
