# code/envs/tictactoe.py
from __future__ import annotations
from typing import List, Tuple, Any

EMPTY, X, O = 0, 1, -1

def other(p): return -p

class TicTacToe:
    def initial_state(self):
        return (X, (0,)*9)  # (player_to_move, board tuple)

    def legal_actions(self, state) -> List[int]:
        _, board = state
        return [i for i,v in enumerate(board) if v == EMPTY]

    def step(self, state, action: int) -> Tuple[Any, float, bool]:
        player, board = state
        assert board[action] == EMPTY
        b = list(board); b[action] = player; b = tuple(b)
        w = self._winner(b)
        if w == player:   return ((other(player), b), 1.0, True)
        if w == other(player): return ((other(player), b), -1.0, True)
        if all(v != EMPTY for v in b): return ((other(player), b), 0.0, True)
        return ((other(player), b), 0.0, False)

    def terminal_value(self, state):
        player, board = state
        w = self._winner(board)
        if w == player: return True, 1.0
        if w == other(player): return True, -1.0
        if all(v != EMPTY for v in board): return True, 0.0
        return False, 0.0

    def hashable_state(self, state): return state

    def _winner(self, b):
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        for i,j,k in lines:
            s = b[i]+b[j]+b[k]
            if s == 3: return X
            if s == -3: return O
        return 0
