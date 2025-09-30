# gridworld_base.py
# Classic 4x4 Gridworld from Sutton & Barto (2nd ed.), Fig 4.1 setup.
# Two terminal states (0,0) and (3,3). Rewards: -1 per step, 0 on entering terminal.
# Policy: equiprobable random actions. Gamma = 1.0.
# This module exposes utilities for iterative policy evaluation.

from __future__ import annotations
import numpy as np

ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # R, L, D, U

class Gridworld4x4:
    def __init__(self):
        self.H = 4
        self.W = 4
        self.terminals = {(0,0), (3,3)}
        self.gamma = 1.0

    def step(self, s, a):
        """Deterministic transition by clamping to grid; reward -1 unless terminal next."""
        if s in self.terminals:
            return s, 0.0
        r, c = s
        dr, dc = ACTIONS[a]
        nr, nc = max(0, min(self.H-1, r+dr)), max(0, min(self.W-1, c+dc))
        ns = (nr, nc)
        if ns in self.terminals:
            return ns, 0.0
        return ns, -1.0

    def states(self):
        return [(r,c) for r in range(self.H) for c in range(self.W)]

def iterative_policy_evaluation(env: Gridworld4x4, theta=1e-4):
    """Evaluate equiprobable random policy on the 4x4 gridworld.
    Returns an array V of shape (H,W).
    """
    V = np.zeros((env.H, env.W), dtype=np.float64)
    while True:
        delta = 0.0
        V_old = V.copy()
        for r in range(env.H):
            for c in range(env.W):
                if (r,c) in env.terminals:
                    continue
                v = 0.0
                for a in range(4):
                    ns, rwd = env.step((r,c), a)
                    v += 0.25 * (rwd + env.gamma * V_old[ns[0], ns[1]])
                V[r,c] = v
                delta = max(delta, abs(V[r,c]-V_old[r,c]))
        if delta < theta:
            break
    return V
