# code/llm_model_interface.py
# Where LLM priors would plug in. Left simple on purpose so it's easy to swap your own calls.

from typing import Any, Dict, List, Tuple

class LLMWorldModel:
    def action_priors(self, state: Any, legal_actions: List[Any]) -> Dict[Any, float]:
        if not legal_actions: return {}
        p = 1.0 / len(legal_actions)
        return {a: p for a in legal_actions}

    def value_prior(self, state: Any) -> float:
        # Heuristic fallback: unknown -> neutral
        return 0.0
