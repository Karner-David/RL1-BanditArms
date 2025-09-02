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
    def __init__(self, K, q0=1.0):
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
            # TODO: agent picks action, bandit returns reward, agent updates
            pass

    # TODO: compute averages and CIs
    return rewards, optimal, regret

def plot_results(xs, results):
    # TODO: implement matplotlib plots with CI shading
    pass

# ------------------ Main ------------------
if __name__ == "__main__":
    # Example: run epsilon-greedy
    rewards, optimal, regret = run_experiment(EpsilonGreedy, {"epsilon":0.1})
    # TODO: plot
