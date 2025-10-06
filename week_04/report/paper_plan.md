# Paper Plan — LLM‑MCTS (NeurIPS 2023)

**Chosen paper**  
Zirui Zhao, Wee Sun Lee, David Hsu. **Large Language Models as Commonsense Knowledge for Large-Scale Task Planning.** *NeurIPS 2023.*  
PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/65a39213d7d0e1eb5d192aa77e77eeb7-Paper-Conference.pdf  
arXiv: https://arxiv.org/abs/2305.14078  
OpenReview: https://openreview.net/forum?id=Wjp1AYB8lH

## Why this paper
It integrates an LLM as (i) a commonsense **world model** and (ii) **policy priors**, while MCTS verifies and composes multi‑step plans. This matches the assignment’s requirement to combine MCTS with LLMs on a realistic task.

## Task (realistic, not Gym)
A small **public‑transit planning** simulator on a fixed graph (`data/transit_graph.json`). Objective: go from A to F with short ETA and few transfers. No external web calls at runtime.

## Method I will implement
- Start with the provided **MCTS‑UCT**.
- Add **PUCT** and **prior pseudo‑counts** to incorporate LLM priors:
  - `action_priors(state)` → initialize edge priors; either (a) PUCT: `Q + c * P(s,a) * sqrt(N) / (1 + N(s,a))`, or (b) add Dirichlet‑like pseudo‑counts to `N(s,a)`.
  - `value_prior(state)` → bootstrap Q estimates for faster convergence.
- For an initial ablation, mimic LLM hints with small heuristics (e.g., prefer edges that reduce time). Then, if allowed, replace with an actual LLM call.

## Evaluation
- **Metrics**: success rate, mean steps, and (optionally) elapsed simulated minutes.
- **Baselines**: Greedy heuristic; MCTS‑UCT w/o priors; **MCTS+LLM priors (PUCT)**.
- **Ablations**: number of simulations, exploration constant `c`, temperature at root, strength of priors.

## Expected outcome
With informative priors, the planner should expand fewer off‑route branches and reach goals with fewer simulations—qualitatively consistent with findings reported by Zhao et al. (2023). See References below.

## References
- Zhao, Z., Lee, W. S., & Hsu, D. (2023). *Large Language Models as Commonsense Knowledge for Large-Scale Task Planning.* **NeurIPS 2023**.  
  PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/65a39213d7d0e1eb5d192aa77e77eeb7-Paper-Conference.pdf  
  arXiv: https://arxiv.org/abs/2305.14078  
  OpenReview: https://openreview.net/forum?id=Wjp1AYB8lH
