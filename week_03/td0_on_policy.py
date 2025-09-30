# td0_on_policy.py
# TD(0) on-policy control = SARSA(0) with epsilon-greedy behavior.

from __future__ import annotations
import numpy as np
from windy_gridworld import WindyGridworld, ACTIONS

def eps_greedy(Q, s, eps, rng):
    if rng.rand() < eps:
        return rng.randint(len(ACTIONS))
    return int(np.argmax(Q[s]))

def sarsa_control(env: WindyGridworld, episodes=5000, eps=0.1, alpha=0.1, gamma=1.0, seed=0):
    rng = np.random.RandomState(seed)
    Q = {s: np.zeros(len(ACTIONS)) for s in env.states()}
    for _ in range(episodes):
        s = env.reset()
        a = eps_greedy(Q, s, eps, rng)
        done = False
        while not done:
            ns, r, done = env.step(s, a)
            if done:
                Q[s][a] += alpha * (r - Q[s][a])
                break
            na = eps_greedy(Q, ns, eps, rng)
            td_target = r + gamma * Q[ns][na]
            Q[s][a] += alpha * (td_target - Q[s][a])
            s, a = ns, na
    pi = {s: int(np.argmax(Q[s])) for s in Q}
    return Q, pi
