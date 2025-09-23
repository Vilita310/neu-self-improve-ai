# bandit_hw.py
# Minimal solution: k-armed bandit framework, Figure 2.2 reproduction, and gradient bandit (4 settings)

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Environment
# ---------------------------

class Env:
    """Abstract environment interface."""
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError

class Bandit(Env):
    """
    Stationary k-armed bandit.
    True values q* ~ N(0,1); reward ~ N(q*[a], 1).
    """
    def __init__(self, k=10):
        self.k = k
        self.reset()

    def reset(self):
        self.q_true = np.random.normal(0.0, 1.0, size=self.k)
        return None

    def step(self, action: int):
        reward = np.random.normal(self.q_true[action], 1.0)
        return None, reward, False, {}

# ---------------------------
# Agents
# ---------------------------

class Agent:
    """Abstract agent interface."""
    def select_action(self):
        raise NotImplementedError
    def update(self, action, reward, next_state=None):
        raise NotImplementedError

class EpsilonGreedyAgent(Agent):
    """
    ε-greedy with sample-average updates.
    """
    def __init__(self, k=10, epsilon=0.1, q_init=0.0):
        self.k = k
        self.epsilon = epsilon
        self.q_est = np.full(k, q_init, dtype=float)
        self.action_count = np.zeros(k, dtype=int)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return int(np.argmax(self.q_est))

    def update(self, action, reward, next_state=None):
        self.action_count[action] += 1
        step = 1.0 / self.action_count[action]  # sample-average
        self.q_est[action] += step * (reward - self.q_est[action])

class GradientBanditAgent(Agent):
    """
    Gradient bandit with softmax over preferences H.
    Optional baseline = running average reward.
    """
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.H = np.zeros(k, dtype=float)
        self.pi = np.ones(k, dtype=float) / k
        self.t = 0
        self.avg_reward = 0.0

    def _update_policy(self):
        # softmax with a simple max-shift for numerical stability
        m = np.max(self.H)
        expH = np.exp(self.H - m)
        self.pi = expH / np.sum(expH)

    def select_action(self):
        self._update_policy()
        return int(np.random.choice(self.k, p=self.pi))

    def update(self, action, reward, next_state=None):
        self.t += 1
        b = 0.0
        if self.baseline:
            # running average as baseline
            self.avg_reward += (reward - self.avg_reward) / self.t
            b = self.avg_reward
        self._update_policy()
        onehot = np.zeros(self.k)
        onehot[action] = 1.0
        # preference update
        self.H += self.alpha * (reward - b) * (onehot - self.pi)

# ---------------------------
# Experiment
# ---------------------------

def simulate(agent_ctor, runs=2000, steps=1000, env_ctor=Bandit, **agent_kwargs):
    """
    Run multiple independent bandit problems and average metrics.
    Returns:
        mean_rewards: (steps,)
        mean_optimal: (steps,) fraction of optimal action chosen
    """
    rewards = np.zeros((runs, steps), dtype=float)
    optimal = np.zeros((runs, steps), dtype=float)

    for r in range(runs):
        env = env_ctor()
        env.reset()
        agent = agent_ctor(**agent_kwargs)
        optimal_action = int(np.argmax(env.q_true))

        for t in range(steps):
            a = agent.select_action()
            _, rew, _, _ = env.step(a)
            agent.update(a, rew, None)
            rewards[r, t] = rew
            optimal[r, t] = 1.0 if a == optimal_action else 0.0

    return rewards.mean(axis=0), optimal.mean(axis=0)

def main():
    # Fixed setup per assignment
    runs = 2000
    steps = 1000
    seed = 0
    np.random.seed(seed)

    # Figure 2.2: ε-greedy (sample-average updates)
    _, opt_eps0   = simulate(EpsilonGreedyAgent, runs=runs, steps=steps, epsilon=0.0)
    _, opt_eps001 = simulate(EpsilonGreedyAgent, runs=runs, steps=steps, epsilon=0.01)
    _, opt_eps01  = simulate(EpsilonGreedyAgent, runs=runs, steps=steps, epsilon=0.1)

    # Gradient bandit (four required settings)
    _, opt_g_nb_a01 = simulate(GradientBanditAgent, runs=runs, steps=steps, alpha=0.1, baseline=False)
    _, opt_g_nb_a04 = simulate(GradientBanditAgent, runs=runs, steps=steps, alpha=0.4, baseline=False)
    _, opt_g_b_a01  = simulate(GradientBanditAgent, runs=runs, steps=steps, alpha=0.1, baseline=True)
    _, opt_g_b_a04  = simulate(GradientBanditAgent, runs=runs, steps=steps, alpha=0.4, baseline=True)

    # Plot as percentage to match the book axis
    pct = lambda y: y * 100.0

    plt.figure(figsize=(12, 8))
    plt.plot(pct(opt_eps0),   label="ε-greedy ε=0")
    plt.plot(pct(opt_eps001), label="ε-greedy ε=0.01")
    plt.plot(pct(opt_eps01),  label="ε-greedy ε=0.1")
    plt.plot(pct(opt_g_nb_a01), label="Gradient α=0.1 (no baseline)")
    plt.plot(pct(opt_g_nb_a04), label="Gradient α=0.4 (no baseline)")
    plt.plot(pct(opt_g_b_a01),  label="Gradient α=0.1 (with baseline)")
    plt.plot(pct(opt_g_b_a04),  label="Gradient α=0.4 (with baseline)")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Figure 2.2 + Gradient Bandit (4 settings)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_2_2_with_gradient.png", dpi=160)
    plt.show()

if __name__ == "__main__":
    main()
