# code/envs/transit.py
# Tiny "realistic" task: find a route from src to dst over a toy transit graph.
# MCTS treats each state as (current_stop, goal_stop). Actions are next legs.

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json

class TransitEnv:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)  # {stop: [{"to": stop, "time": minutes}]}

    def initial_state(self, src_dst: Tuple[str, str]):
        src, dst = src_dst
        return (src, dst, 0)  # (current, goal, elapsed_minutes)

    def legal_actions(self, state) -> List[Tuple[str, int]]:
        cur, dst, t = state
        return [(edge["to"], edge["time"]) for edge in self.graph.get(cur, [])]

    def step(self, state, action: Tuple[str, int]):
        cur, dst, t = state
        nxt_stop, cost = action
        nxt_state = (nxt_stop, dst, t + cost)
        done, val = self.terminal_value(nxt_state)
        # reward from the actor's perspective: negative travel time to encourage speed
        reward = 0.0
        if done:
            reward = 0.0 if nxt_stop != dst else 1.0  # small terminal reward for success
        else:
            reward = -cost / 60.0  # scale minutes to hours
        return nxt_state, reward, done

    def terminal_value(self, state):
        cur, dst, t = state
        if cur == dst:
            return True, 1.0
        # depth or time guard could be added by caller in rollout depth
        return False, 0.0

    def hashable_state(self, state): return state
