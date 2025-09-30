# mc_on_policy.py
# First-visit MC on-policy control with epsilon-soft policies for Windy Gridworld.

from __future__ import annotations
import numpy as np
from windy_gridworld import WindyGridworld, ACTIONS

def epsilon_greedy(Q, s, eps):
    a_star = np.argmax(Q[s])
    probs = np.ones(len(ACTIONS)) * (eps / len(ACTIONS))
    probs[a_star] += 1.0 - eps
    return probs

def run_episode(env: WindyGridworld, Q, eps, gamma, rng):
    s = env.reset()
    episode = []
    done = False
    while not done:
        probs = epsilon_greedy(Q, s, eps)
        a = rng.choice(len(ACTIONS), p=probs)
        ns, r, done = env.step(s, a)
        episode.append((s, a, r))
        s = ns
    # compute returns
    G = 0.0
    returns = []
    for t in reversed(range(len(episode))):
        s,a,r = episode[t]
        G = r + gamma * G
        returns.append((s,a,G))
    returns.reverse()
    # first-visit updates
    seen = set()
    for t,(s,a,Gt) in enumerate(returns):
        if (s,a) in seen: 
            continue
        seen.add((s,a))
        Q[s][a] += 0.1 * (Gt - Q[s][a])  # constant-alpha MC
    return episode

def mc_on_policy_control(env: WindyGridworld, episodes=5000, eps=0.1, gamma=1.0, seed=0):
    rng = np.random.RandomState(seed)
    Q = {s: np.zeros(len(ACTIONS)) for s in env.states()}
    for _ in range(episodes):
        run_episode(env, Q, eps, gamma, rng)
    # derive greedy policy
    pi = {s: np.argmax(Q[s]) for s in Q}
    return Q, pi
