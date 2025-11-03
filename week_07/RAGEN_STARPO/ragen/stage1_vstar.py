#stage1_vstar.py will use the Critic part to get its "guesses" (values)
import torch

def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the advantages and value targets using Generalized Advantage Estimation (GAE).
    
    This is the "V-star" part of your stage1. It looks at the entire
    trajectory and calculates how "surprising" the outcome was at each step.

    Args:
        rewards: A tensor of rewards from each step of the trajectory. Shape: [T]
        values: A tensor of value (critic) estimates for each state. Shape: [T+1]
                (Includes the value of the *final* next_state)
        dones: A tensor of done flags (1.0 if done, 0.0 if not). Shape: [T]
        gamma: The discount factor (e.g., 0.99).
        gae_lambda: The GAE lambda parameter (e.g., 0.95).

    Returns:
        A tuple of (advantages, value_targets):
        - advantages: The GAE advantage estimate for each step. Shape: [T]
        - value_targets: The "V-star" target for the critic to train on. Shape: [T]
    """
    
    # We need to reverse the Tensors to loop backward from the end of the episode
    rewards = torch.flip(rewards, dims=(0,))
    values = torch.flip(values, dims=(0,)) # This is [T+1]
    dones = torch.flip(dones, dims=(0,))
    
    # T (trajectory length)
    T = len(rewards)
    
    advantages = []
    gae_advantage = 0.0
    
    # Get the value of the *last* state, which is the first item in the flipped tensor
    # This V(s_T+1) is 0 if the episode was done, otherwise it's the critic's estimate
    next_value = values[0] 
    
    # Loop backwards through the trajectory (from T-1 down to 0)
    for t in range(T):
        # The 'values' tensor has T+1 items.
        # values[t+1] is V(s_t)
        # values[t] is V(s_t+1)
        # We use values[t+1] here
        current_value = values[t+1] 
        
        # 1. Calculate the TD Error (the "surprise")
        #    delta = r_t + gamma * V(s_t+1) * (not done) - V(s_t)
        #    (1.0 - dones[t]) is the 'not done' mask
        td_error = rewards[t] + gamma * next_value * (1.0 - dones[t]) - current_value
        
        # 2. Calculate the GAE Advantage
        #    A_t = delta_t + gamma * lambda * A_t+1 * (not done)
        gae_advantage = td_error + gamma * gae_lambda * gae_advantage * (1.0 - dones[t])
        
        # Prepend to our list (to re-reverse the order)
        advantages.insert(0, gae_advantage)
        
        # Update next_value for the next loop iteration
        next_value = current_value

    # Convert advantages list to a tensor
    advantages_tensor = torch.stack(advantages)
    
    # 3. Calculate Value Targets ("V-star")
    #    The value target is just the advantage plus the original value estimate
    #    V_target = A_t + V(s_t)
    #    We use values[1:] to get the V(s_t) for t=0..T-1 (flipped)
    #    and then re-flip the result to match the original order.
    value_targets_reversed = advantages_tensor + torch.flip(values[1:], dims=(0,))
    
    return advantages_tensor, value_targets_reversed

# ==========================================================
# Test run
# ==========================================================
if __name__ == "__main__":
    """
    A simple test to make sure the GAE calculation works.
    
    To run from your project root:
    python ragen/stage1_vstar.py
    """
    print("--- Testing GAE (stage1_vstar.py) ---")
    
    rewards_t = torch.tensor([0.0, 0.0, 1.0])
    dones_t = torch.tensor([0.0, 0.0, 1.0])
    values_t = torch.tensor([0.1, 0.2, 0.5, 0.0])
    gamma = 0.99
    gae_lambda = 0.95
    
    advantages, value_targets = compute_gae_advantages(
        rewards_t, values_t, dones_t, gamma, gae_lambda
    )
    
    print(f"Rewards:    {rewards_t}")
    print(f"Values:     {values_t[:-1]}") 
    print(f"Advantages: {advantages}")
    print(f"V-Targets:  {value_targets}")
    print("\n--- Test Complete ---")