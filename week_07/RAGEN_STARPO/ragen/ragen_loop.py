#ragen_loop.py: will use the Actor part to get actions.
import sys
import os
import random
from typing import Callable, List, Dict, Any, Tuple

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from envs.webshop_env import WebShopEnv

PolicyFunction = Callable[[str], Tuple[str, str]]

# ==========================================================
# PART B: RAGEN Loop
# ==========================================================

def heuristic_policy(observation: str) -> Tuple[str, str]:
    # ... (this function remains the same as before)
    obs_lower = observation.lower()
    thought = ""
    action = ""
    if "you see 5 items" in obs_lower:
        thought = "I'm on the main page. The target is 'red running shoe'. I should search for 'red shoes'."
        action = "search red shoes"
    elif "results for" in obs_lower:
        thought = "I see search results. I will click the first item, 'Red Running Shoes' (ID 1)."
        action = "click 1" 
    elif "you opened" in obs_lower:
        thought = "I have an item page open. I will buy this item."
        action = "buy 1" 
    else:
        thought = "I am not sure where I am. I will search for 'shoes' to reset."
        action = "search shoes"
    return thought, action


def run_ragen_loop(env: WebShopEnv, policy_fn: PolicyFunction) -> tuple[List[Dict[str, Any]], float]:
    obs = env.reset()
    done = False
    total_reward = 0
    trajectory = []
    
    print(f"--- Starting New RAGEN Rollout ---")
    print(f"Initial Observation: {obs}\n")
    
    max_steps = 10 
    step_count = 0
    
    while not done and step_count < max_steps:
        
        # --- PHASE 1: THINK (Policy) ---
        thought, action = policy_fn(obs)
        
        # --- PHASE 2: ACT (Environment) ---
        next_obs, reward, done = env.step(action)
        
        # --- PHASE 3: OBSERVE (Store) ---
        step_data = {
            "observation": obs,
            "thought": thought,
            "action": action,
            "reward": reward,
            "next_observation": next_obs,
            "done": done,
            # --- THIS IS THE CRITICAL ADDITION ---
            # It stores the tensors (log_prob, value, etc.) from the agent
            "policy_data": getattr(policy_fn, 'last_data', {}) 
            # ------------------------------------
        }
        trajectory.append(step_data)
        
        obs = next_obs
        total_reward += reward
        step_count += 1
        
        print(f"  Step:    {step_count}")
        print(f"  Thought: {thought}")
        print(f"  Action:  {action}")
        print(f"  Result:  {obs}")
        print("-" * 20)
        
    print(f"--- Episode Finished ---")
    print(f"Final Reward: {total_reward}")
    
    return trajectory, total_reward


# ==========================================================
# Test run for Part B (remains the same)
# ==========================================================
if __name__ == "__main__":
    print("=" * 40)
    print("  RUNNING RAGEN LOOP (PART B TEST)")
    print("=" * 40)
    
    env = WebShopEnv()
    policy_to_use = heuristic_policy
    trajectory, final_reward = run_ragen_loop(env, policy_to_use)
    
    print(f"\nRollout complete. Total steps: {len(trajectory)}. Final score: {final_reward}")
    
    print("\n=" * 40)
    print("  RAGEN LOOP (PART B) TEST COMPLETE")
    print("=" * 40)