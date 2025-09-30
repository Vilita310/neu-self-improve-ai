# Windy Gridworld & Figure 4.1 (Sutton & Barto)

This folder contains exactly what's required:
- **Replicate Figure 4.1** (4×4 gridworld, equiprobable random policy).
- **Implement Windy Gridworld** (stochastic wind as specified).
- **Implement control algorithms over Windy Gridworld**:
  - DP control (policy iteration)
  - MC on‑policy control (first‑visit, ε‑soft)
  - MC off‑policy control (ordinary IS)
  - TD(0) on‑policy control (SARSA)
  - TD(0) off‑policy control with **unweighted** IS
  - TD(0) off‑policy control with **weighted** IS

Minimal and direct—no extras beyond the requirement.

## Quick Start

```bash
# Python 3.10+ recommended
cd week_windy_gridworld
python run_all.py
```

Expected outputs:
- A 4×4 value table approximating **Fig 4.1**.
- A few one‑line prints for each control method over Windy Gridworld (action at start, etc.).

## Files

- `gridworld_base.py` — Classic 4×4 gridworld and iterative policy evaluation to reproduce **Fig 4.1**.
- `replicate_figure_4_1.py` — Standalone script that prints the value table.
- `windy_gridworld.py` — Stochastic Windy Gridworld environment exactly per spec.
- `dp_control.py` — Policy iteration using expected model (enumerates wind outcomes).
- `mc_on_policy.py` — First‑visit MC control with ε‑soft policy.
- `mc_off_policy.py` — Off‑policy MC with ordinary importance sampling.
- `td0_on_policy.py` — SARSA(0) on‑policy control.
- `td0_off_policy_is.py` — Off‑policy TD(0) via importance‑sampled SARSA (ordinary & weighted).
- `run_all.py` — Simple runner to exercise everything.
