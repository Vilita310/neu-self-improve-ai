# Method Notes (short)

- **UCT rule**: `Q + c * sqrt(log N / (n + eps))`. I used `c ≈ sqrt(2)` as the default.
- **Rollouts**: uniform random for now. Depth limits matter a lot; too shallow = noisy; too deep = slow.
- **Backups**: simple incremental mean of edge returns. I didn't alternate signs (env returns are already from the actor’s perspective).
- **Output**: picked action by max visit count. Temperature sampling is there if we want variety during training.
- **What I tried**: mainly c=1.2~1.6, iteration counts from 200 to 3k. More sims is better (no surprise).


Ref: Zhao, Lee, & Hsu (NeurIPS 2023) — LLM‑MCTS.
