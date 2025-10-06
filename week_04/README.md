# Week 04 — MCTS-UCT (plus a small LLM+MCTS plan)

This week’s assignment had two parts:
1) Implement MCTS-UCT and validate it on a simple environment.
2) Pick a recent paper (post-2022) that uses MCTS with LLMs and either replicate a result or apply it to a realistic task.  
   I picked **LLM-MCTS (NeurIPS 2023)** and sketched a small, realistic task (toy public-transit planning).

## What’s here

- `src/mcts_uct.py` — plain MCTS with the UCT rule. No magic, just the four classic steps.
- `src/envs/tictactoe.py` — Tic-Tac-Toe for quick sanity checks.
- `src/run_tictactoe.py` — tiny CLI to play against the MCTS agent.
- `src/envs/transit.py` + `data/transit_graph.json` — a small “realistic” task: choose connections across a toy bus/rail graph to get from A to B.
- `src/llm_model_interface.py` — where I would plug in LLM priors (action priors / value prior). Left as a thin, clear stub.
- `tests/test_mcts_basic.py` — super small tests to catch obvious regressions.
- `report/paper_plan.md` — short plan for the LLM-MCTS paper and how I’d evaluate it on the transit task.
- `report/method_notes.md` — my short notes on UCT, parameters, and things I tried.
- `report/references.bib` — BibTeX for the chosen paper.
- `scripts/run_tictactoe.sh` — one-liner to run a 2k-sim game.

## How to run
```bash
cd week_04

# 1) Play Tic-Tac-Toe vs MCTS (MCTS is X, you are O). Press 0..8 for moves.
python -m src.run_tictactoe --iters 2000 --c 1.414

# 2) (Optional) Quick dry-run of transit search from A to F
python - <<'PY'
from src.envs.transit import TransitEnv
from src.mcts_uct import MCTS
env = TransitEnv("data/transit_graph.json")
s = env.initial_state(("A","F"))
mcts = MCTS(env, c=1.2, rollout_depth=12, gamma=0.99)
a = mcts.plan(s, iters=1500)
print("First suggested leg from A toward F:", a)
PY
```

## Running tests

```bash
cd week_04
pytest -q
```

> Optional (not recommended): run the game script directly

```bash
cd week_04
python src/run_tictactoe.py --iters 2000 --c 1.414
```

## Notes (student-style)

* I default to `c = sqrt(2)` but `1.0~2.0` all felt fine on Tic-Tac-Toe. More sims > fancy tricks.
* Rollouts are uniform random; for transit I cap depth and use small discount. Obvious room for a learned or heuristic rollout policy.
* For the paper part, I’d add **PUCT** and prior pseudo-counts once I have the LLM priors ready (see comments in `src/mcts_uct.py`).

## References

* Zhao, Z., Lee, W. S., & Hsu, D. (2023). *Large Language Models as Commonsense Knowledge for Large-Scale Task Planning.* **NeurIPS 2023**.
  PDF: [https://proceedings.neurips.cc/paper_files/paper/2023/file/65a39213d7d0e1eb5d192aa77e77eeb7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/65a39213d7d0e1eb5d192aa77e77eeb7-Paper-Conference.pdf)
  arXiv: [https://arxiv.org/abs/2305.14078](https://arxiv.org/abs/2305.14078)
  OpenReview: [https://openreview.net/forum?id=Wjp1AYB8lH](https://openreview.net/forum?id=Wjp1AYB8lH)
