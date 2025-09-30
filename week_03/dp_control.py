# dp_control.py
# Dynamic Programming control (policy iteration) for Windy Gridworld.
# We compute expected transitions by enumerating the three wind outcomes (0.1, 0.8, 0.1).

from __future__ import annotations
import numpy as np
from windy_gridworld import WindyGridworld, WGConfig, DELTAS, ACTIONS

def _phi(env: WindyGridworld, s, a, wind_override):
    """Deterministic transform applying action and a specific wind value."""
    if s == env.goal:
        return s
    r, c = s
    dr, dc = DELTAS[a]
    tr, tc = r + dr, c + dc
    tr = tr - wind_override
    tr = max(0, min(env.H-1, tr))
    tc = max(0, min(env.W-1, tc))
    ns = (tr, tc)
    if ns in env.walls:
        ns = s
    return ns

def policy_evaluation(env: WindyGridworld, V, pi, gamma=1.0, theta=1e-6):
    """Evaluate a deterministic or stochastic policy pi(s)->prob(a|s) using model expectations."""
    H, W = env.H, env.W
    while True:
        delta = 0.0
        V_old = V.copy()
        for r in range(H):
            for c in range(W):
                s = (r,c)
                if s == env.goal or s in env.walls:
                    V[r,c] = 0.0
                    continue
                v = 0.0
                for a, pa in enumerate(pi[r,c]):
                    if pa == 0.0:
                        continue
                    # Enumerate wind outcomes: w+1 (0.1), w (0.8), 0 (0.1)
                    w = env.cfg.wind_by_col[c]
                    outcomes = [(w+1, 0.1), (w, 0.8), (0, 0.1)]
                    for wv, pw in outcomes:
                        ns = _phi(env, s, a, wv)
                        rwd = 0.0 if ns == env.goal else -1.0
                        v += pa * pw * (rwd + gamma * V_old[ns[0], ns[1]])
                delta = max(delta, abs(v - V[r,c]))
                V[r,c] = v
        if delta < theta:
            break
    return V

def policy_improvement(env: WindyGridworld, V, gamma=1.0):
    H, W = env.H, env.W
    pi = np.zeros((H,W,len(ACTIONS)), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            s = (r,c)
            if s == env.goal or s in env.walls:
                pi[r,c,:] = 0.0
                continue
            q = np.zeros(len(ACTIONS))
            w = env.cfg.wind_by_col[c]
            outcomes = [(w+1, 0.1), (w, 0.8), (0, 0.1)]
            for a in ACTIONS:
                val = 0.0
                for wv, pw in outcomes:
                    ns = _phi(env, s, a, wv)
                    rwd = 0.0 if ns == env.goal else -1.0
                    val += pw * (rwd + gamma * V[ns[0], ns[1]])
                q[a] = val
            # greedy
            best = np.argwhere(q == q.max()).flatten()
            new_pi = np.zeros(len(ACTIONS))
            new_pi[best] = 1.0 / len(best)
            pi[r,c,:] = new_pi
    return pi

def policy_iteration(env: WindyGridworld, gamma=1.0, theta=1e-6, max_iter=200):
    H, W = env.H, env.W
    V = np.zeros((H,W), dtype=np.float64)
    # Start with equiprobable policy
    pi = np.ones((H,W,len(ACTIONS)), dtype=np.float64) / len(ACTIONS)
    for _ in range(max_iter):
        V = policy_evaluation(env, V, pi, gamma, theta)
        new_pi = policy_improvement(env, V, gamma)
        if np.allclose(new_pi, pi):
            break
        pi = new_pi
    return V, pi
