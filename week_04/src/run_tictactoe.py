# src/run_tictactoe.py
# Runner for Tic-Tac-Toe vs MCTS.
# Supports two ways of running:
#   (A) Recommended: python -m src.run_tictactoe --iters 2000 --c 1.414
#   (B) Fallback:    python src/run_tictactoe.py  (uses a small sys.path tweak)

import argparse, math, random, sys, os

# Fallback path tweak if executed as a file: ensure package root on sys.path
if __name__ == "__main__":
    PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)

from src.mcts_uct import MCTS
from src.envs.tictactoe import TicTacToe, X, O, EMPTY

def pretty(board):
    mk = {EMPTY: ".", X: "X", O: "O"}
    return "\n".join(" ".join(mk[board[r*3+c]] for c in range(3)) for r in range(3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--c", type=float, default=math.sqrt(2))
    ap.add_argument("--temp", type=float, default=None)
    args = ap.parse_args()

    env = TicTacToe()
    mcts = MCTS(env, c=args.c, rollout_depth=16, gamma=1.0, rng=random.Random(42))
    state = env.initial_state()
    print("You are O. MCTS plays X. Enter moves as 0..8 (top-left to bottom-right).")

    while True:
        player, board = state
        print("\nBoard:\n" + pretty(board))
        done, val = env.terminal_value(state)
        if done:
            if val > 0: print("Player-to-move wins.")
            elif val < 0: print("Player-to-move loses.")
            else: print("Draw.")
            break

        legal = env.legal_actions(state)
        if player == X:
            a = mcts.plan(state, iters=args.iters, temperature=args.temp)
            print(f"MCTS chooses: {a}")
        else:
            while True:
                try:
                    a = int(input(f"Your move {legal}: "))
                except EOFError:
                    print("\nBye."); return
                if a in legal: break
                print("Illegal move.")
        state, reward, done = env.step(state, a)

if __name__ == "__main__":
    main()
