# code/mcts_uct.py
# Minimal, readable MCTS with UCT.
# Student submission — kept straightforward on purpose.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import random

@dataclass
class NodeStats:
    N: int = 0                                 # N(s)
    Nsa: Dict[Any, int] = field(default_factory=dict)     # N(s,a)
    Qsa: Dict[Any, float] = field(default_factory=dict)   # Q(s,a)
    children: Dict[Any, Any] = field(default_factory=dict) # a -> child_key

class MCTS:
    def __init__(self, env, c: float = math.sqrt(2), rollout_depth: int = 20, gamma: float = 1.0, rng: Optional[random.Random]=None):
        """
        env must provide:
          - legal_actions(state) -> list[action]
          - step(state, a) -> (next_state, reward, done), reward from the actor's perspective
          - terminal_value(state) -> (done: bool, value_from_player_to_move)
          - hashable_state(state) -> Any
        """
        self.env = env
        self.c = float(c)
        self.rollout_depth = int(rollout_depth)
        self.gamma = float(gamma)
        self.rng = rng or random.Random(0)
        self.table: Dict[Any, NodeStats] = {}

    def plan(self, root_state: Any, iters: int = 1000, temperature: Optional[float] = None) -> Any:
        root_key = self.env.hashable_state(root_state)
        if root_key not in self.table:
            self.table[root_key] = NodeStats()

        for _ in range(int(iters)):
            self._simulate(root_state)

        stats = self.table[root_key]
        legal = self.env.legal_actions(root_state)
        if not legal:
            return None

        visits = [(a, stats.Nsa.get(a, 0)) for a in legal]
        if not visits:
            return None

        if temperature is None or temperature <= 1e-8:
            return max(visits, key=lambda x: x[1])[0]
        else:
            # sample by N^(1/T)
            weights = [(a, max(1, n)**(1.0/temperature)) for a, n in visits]
            Z = sum(w for _, w in weights) or 1.0
            r = self.rng.random() * Z
            cum = 0.0
            for a, w in weights:
                cum += w
                if r <= cum:
                    return a
            return weights[-1][0]

    def _simulate(self, state: Any) -> float:
        path = []  # [(s_key, a)]
        cur = state

        # Selection / Expansion
        depth = 0
        while True:
            done, val = self.env.terminal_value(cur)
            if done:
                value = val
                break

            s_key = self.env.hashable_state(cur)
            stats = self.table.setdefault(s_key, NodeStats())
            actions = self.env.legal_actions(cur)

            untried = [a for a in actions if a not in stats.children]
            if untried:
                a = self.rng.choice(untried)
                nxt, reward, done2 = self.env.step(cur, a)
                stats.children[a] = self.env.hashable_state(nxt)
                stats.Nsa.setdefault(a, 0)
                stats.Qsa.setdefault(a, 0.0)
                path.append((s_key, a))
                if done2:
                    value = reward
                else:
                    value = self._rollout(nxt)
                break
            else:
                a = self._uct_action(s_key, actions)
                nxt, reward, done2 = self.env.step(cur, a)
                path.append((s_key, a))
                cur = nxt
                depth += 1
                if done2:
                    value = reward
                    break

        # Backprop
        self._backup(path, value)
        return value

    def _uct_action(self, s_key: Any, actions: List[Any]) -> Any:
        stats = self.table[s_key]
        Ns = max(1, stats.N)
        best, best_val = None, -1e18
        for a in actions:
            Na = stats.Nsa.get(a, 0)
            Q = stats.Qsa.get(a, 0.0)
            ucb = Q + self.c * math.sqrt(math.log(Ns + 1) / (Na + 1e-9))
            if ucb > best_val:
                best, best_val = a, ucb
        return best

    def _rollout(self, state: Any) -> float:
        g, disc = 0.0, 1.0
        cur = state
        for _ in range(self.rollout_depth):
            done, val = self.env.terminal_value(cur)
            if done:
                g += disc * val
                break
            actions = self.env.legal_actions(cur)
            if not actions:
                break
            a = random.choice(actions)
            nxt, reward, done2 = self.env.step(cur, a)
            g += disc * reward
            disc *= self.gamma
            cur = nxt
            if done2:
                break
        return g

    def _backup(self, path, value: float):
        # incremental mean over edges in path
        for s_key, a in reversed(path):
            stats = self.table[s_key]
            stats.N += 1
            nsa = stats.Nsa.get(a, 0) + 1
            qsa = stats.Qsa.get(a, 0.0)
            stats.Qsa[a] = qsa + (value - qsa) / nsa
            stats.Nsa[a] = nsa
