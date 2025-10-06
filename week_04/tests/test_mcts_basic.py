# tests/test_mcts_basic.py
# Tiny smoke tests (pytest). These aren't exhaustive, just to catch silly regressions.

import math, random
from src.mcts_uct import MCTS
from src.envs.tictactoe import TicTacToe, EMPTY, X, O

def test_tictactoe_not_crash():
    env = TicTacToe()
    mcts = MCTS(env, c=math.sqrt(2), rollout_depth=8, gamma=1.0, rng=random.Random(0))
    s = env.initial_state()
    a = mcts.plan(s, iters=200)
    assert a in env.legal_actions(s)

def test_tictactoe_center_reasonable():
    env = TicTacToe()
    mcts = MCTS(env, c=1.2, rollout_depth=8, gamma=1.0, rng=random.Random(0))
    s = env.initial_state()
    a = mcts.plan(s, iters=600)
    assert a in range(9)


if __name__ == "__main__":
    # Allow running as: python tests/test_mcts_basic.py
    import sys, os, pytest
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    raise SystemExit(pytest.main([__file__]))
