# MaAS Adapter for FEVER Fact Verification

## Overview
This project adapts the Multi-agent Architecture Search (MaAS) framework to the FEVER (Fact Extraction and VERification) benchmark. The system dynamically searches for the optimal agent topology to verify claims against textual evidence.

## Benchmark
- **Name**: FEVER (Fact Extraction and VERification)
- **Source**: `paper_dev.jsonl` (Development Split)
- **Task**: 3-way Classification (SUPPORTS, REFUTES, NOT ENOUGH INFO)

## Adaptation Details
1.  **Data Loader**: Custom adapter implemented to parse FEVER JSONL format.
2.  **Operators**: Added `web_retrieve` and `evidence_selector` operators to the search space.
3.  **Evaluator**: Modified exact-match evaluation to support textual label matching.

## Key Findings

### Failure Analysis (Root Cause)
The system struggles with short, negated claims. The architecture search often converges on a simple `Retrieve -> Verify` loop for these short queries, missing a necessary `Logic Decomposition` step. This leads to retrieval bias where the model finds documents about the entities but fails to identify the negation.

### Success Analysis (Architecture Discovery)
For complex, long claims involving multiple entities, MaAS successfully discovered a multi-agent structure:
`Planner` -> `Parallel Search` -> `Aggregator` -> `Verifier`.
This dynamic structure allows the system to gather distinct pieces of evidence before making a final verdict.

## File Structure
- `maas_fever_adapter/`: Core adapter code.
- `outputs/fever/`: Execution logs and analysis results.
- `optimize.py`: Entry point for the framework.
