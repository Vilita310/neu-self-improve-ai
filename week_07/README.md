# RAGEN + A*PO on WebShop (Week 07)

Implementing **RAGEN** (trajectory-level agent RL) from scratch and replacing PPO/GRPO with **A*PO**.  
We evaluate on a lightweight WebShop-style environment and report results per the assignment.

> **Minimal PyTorch only**: single-process training (no `verl`, `deepspeed`, etc.). Optional GPU works; CPU is fine for sanity runs.

---

## 1) How the system works (RAGEN + A*PO)

**RAGEN** runs a two-phase loop over **full multi-turn trajectories** (state → thinking → actions → reward).  
**A*PO** replaces PPO/GRPO via a **critic-free** two-stage recipe:

### Stage-1 (Offline V*) — pre-compute optimal soft value per prompt
For each prompt \(x\), sample responses from a **frozen** reference policy \(\pi_{\text{ref}}\), score with the task reward \(r(x,y)\), and estimate:
\[
V^*(x) = \beta \,\log\,\mathbb{E}_{y\sim \pi_{\text{ref}}(\cdot|x)}\!\big[\exp(r(x,y)/\beta)\big]
\]
We **cache** \(V^*(x)\) once; no on-policy value learning is needed.

### Stage-2 (Online policy) — single rollout per prompt, KL-regularized update
For each prompt, generate one response from the **current** policy \(\pi\).  
Form an **advantage** with KL anchoring:
\[
A(x,y) = r(x,y)+\beta \cdot \log \frac{\pi_{\text{ref}}(y|x)}{\pi(y|x)}-V^*(x)
\]
and optimize the simple regression-style objective (no clipping, **no critic**):
\[
\min_{\pi}\; -\log \pi(y|x)\cdot A(x,y) + \lambda\cdot \mathrm{KL}\!\big(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
\]

---

## 2) Code flow (who calls what)

High-level launcher is `ragen/train_ragen_apo.py`:

1) **Phase-1 rollouts**  
   - `train_ragen_apo.py → ragen_loop.py`: run an episode  
     - `policy_fn(obs)` yields *think + action* (text inference wrapped to tensors)
     - `env.step(action)` returns next obs / reward / done  
     - store full trajectory (obs/thought/action/reward/done)
2) **A*PO updates**  
   - `train_ragen_apo.py → stage1_vstar.py`: from cached or fresh samples of \(\pi_{\text{ref}}\), compute \(V^*(x)\)  
   - `train_ragen_apo.py → stage2_policy_opt.py`: compute length-normalized log-probs, KL-anchored advantage \(A(x,y)\) and do one optimization step (with gradient clipping & stability guards)

This is **RAGEN (outer loop) + A*PO (inner objective)**. We remove PPO/GRPO critic by using the analytic \(V^*\) and KL anchoring.

---

## 3) Setup

```bash
python -m venv .venv && source .venv/bin/activate
# CPU wheels (if GPU, install CUDA build from pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.36.0" "tqdm>=4.66.0" "numpy>=1.24.0"
```

---

## 4) Quick start (5–10 min, CPU OK)

### 4.1 Sanity-check the environment
```bash
python envs/webshop_env.py
# Try actions:
#   click 1
#   search shoes
#   buy 1
```

### 4.2 Train (RAGEN + A*PO)
```bash
python ragen/train_ragen_apo.py   --env_path data/webshop_mock.json   --epochs 3   --batch_size 4   --device cpu     # use 'cuda' if available
```
Outputs:
- `models/best_agent.pth`
- `results/fixed_training_results.json` (training stats)

### 4.3 Evaluate
```bash
python eval/evaluate_webshop.py   --env_path data/webshop_mock.json   --policy_path models/best_agent.pth   --episodes 100   --out results/evaluation_results.json
```

### 4.4 Submission table
We include `results/webshop/summary.csv` (built from `evaluation_results.json`).  
If you need to rebuild, use your helper (e.g., `tools/make_summary.py`):

```bash
python tools/make_summary.py   --eval_json results/evaluation_results.json   --out_csv  results/webshop/summary.csv
```

> Per assignment, only our method’s row is required. Random/ref baselines may be added later if you run them.

---

## 5) What to submit (Nov 3)

- ✅ Code (minimal single-process PyTorch)
- ✅ `results/webshop/summary.csv` (from `evaluation_results.json`)
- ✅ `reports/slides_webshop.pptx` (**2 pages**: system overview + results table)
- ✅ `final_failure_analysis.md` (failure cases & diagnosis)
- (Optional) `reports/webshop_run.jsonl` (per-episode audit)

The instructor may ask any teammate to make a quick code change; keep the following files clean & well-commented:  
`ragen/train_ragen_apo.py`, `ragen/ragen_loop.py`, `ragen/stage1_vstar.py`, `ragen/stage2_policy_opt.py`, `eval/evaluate_webshop.py`.

---

## 6) Notes & troubleshooting

- **Minimalism**: no `verl`/`deepspeed` or other RL frameworks
- **Stability**: guards for empty generations; length-normalized sequence log-probs; clamped advantages; grad clipping (`max_norm=0.5`)
- **Determinism**: set seeds if you need strict repeatability
- **Runtime**: CPU is fine; a single GPU speeds up Stage-2

---

## 7) Acknowledgments

- WebShop: Yao et al. (2022)  
- RAGEN: Multi-turn RL for LLM agents  
- A*PO: Optimal-value regression with KL anchoring; two-stage, critic-free
