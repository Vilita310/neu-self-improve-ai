# windy_gridworld.py
# Stochastic Windy Gridworld as specified in the assignment.
# State s=(r,c). Actions: up, down, left, right. Wind by column with stochastic W_c.
# Reward: -1 per step, 0 on reaching goal. Episode ends at goal.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
DELTAS = {
    UP:   (-1, 0),
    DOWN: ( 1, 0),
    LEFT: ( 0,-1),
    RIGHT:( 0, 1),
}

@dataclass
class WGConfig:
    H: int
    W: int
    wind_by_col: list  # length W, nonnegative ints
    start: tuple       # (r,c)
    goal: tuple        # (r,c)
    walls: set | None = None
    gamma: float = 1.0
    stochastic: bool = True

class WindyGridworld:
    def __init__(self, cfg: WGConfig, seed: int | None = None):
        assert len(cfg.wind_by_col) == cfg.W
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

    @property
    def H(self): return self.cfg.H
    @property
    def W(self): return self.cfg.W
    @property
    def walls(self): return self.cfg.walls or set()
    @property
    def start(self): return self.cfg.start
    @property
    def goal(self): return self.cfg.goal

    def reset(self):
        return self.start

    def sample_wind(self, c):
        w = self.cfg.wind_by_col[c]
        if not self.cfg.stochastic:
            return w
        u = self.rng.rand()
        if u < 0.1:
            return w + 1
        elif u < 0.9:
            return w
        else:
            return 0

    def step(self, s, a):
        """Return next_state, reward, done."""
        if s == self.goal:
            return s, 0.0, True
        r, c = s
        dr, dc = DELTAS[a]
        tr, tc = r + dr, c + dc  # after action
        wc = self.sample_wind(c)
        tr = tr - wc              # apply wind (push up)
        # clip to bounds
        tr = max(0, min(self.H-1, tr))
        tc = max(0, min(self.W-1, tc))
        ns = (tr, tc)
        # bump into wall -> stay
        if ns in self.walls:
            ns = s
        reward = 0.0 if ns == self.goal else -1.0
        done = (ns == self.goal)
        return ns, reward, done

    def states(self):
        S = []
        for r in range(self.H):
            for c in range(self.W):
                if (r,c) not in self.walls:
                    S.append((r,c))
        return S
