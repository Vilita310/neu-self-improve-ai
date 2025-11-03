#stage2_policy_opt.py will calculate the final loss to update both parts.
import torch
import torch.nn.functional as F

def compute_ppo_loss(
    logits_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    old_log_probs_batch: torch.Tensor,
    advantages_batch: torch.Tensor,
    value_targets_batch: torch.Tensor,
    values_batch: torch.Tensor,
    clip_epsilon: float,
    value_loss_coeff: float,
    entropy_coeff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the A*PO (PPO) loss.
    
    This is the core of the "Update Stage" (Phase 2).

    Args:
        logits_batch: New logits from the policy network. Shape: [T, A]
        actions_batch: Actions taken during rollout. Shape: [T]
        old_log_probs_batch: Log-probs of actions from rollout. Shape: [T]
        advantages_batch: Advantages from stage1. Shape: [T]
        value_targets_batch: V-star targets from stage1. Shape: [T]
        values_batch: Value estimates from rollout. Shape: [T]
        clip_epsilon: The PPO clipping parameter (e.g., 0.2).
        value_loss_coeff: Weight for the value loss (e.g., 0.5).
        entropy_coeff: Weight for the entropy bonus (e.g., 0.01).

    Returns:
        A tuple of (total_loss, policy_loss, value_loss, entropy_loss)
    """
    
    # 1. --- Policy Loss (A*PO) ---
    
    # Get new log-probabilities and entropy
    dist = torch.distributions.Categorical(logits=logits_batch)
    new_log_probs = dist.log_prob(actions_batch)
    entropy = dist.entropy().mean()

    # Calculate the ratio: pi(a|s)_new / pi(a|s)_old
    # Using exp(log_prob_new - log_prob_old)
    ratio = torch.exp(new_log_probs - old_log_probs_batch)

    # Calculate the two parts of the PPO objective
    surr1 = ratio * advantages_batch
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch

    # The policy loss is the minimum of the two, negated (we want to maximize)
    policy_loss = -torch.min(surr1, surr2).mean()

    # 2. --- Value Loss (Critic) ---
    
    # This is the "V-star" target.
    # We use a simple Mean Squared Error (MSE) loss.
    #value_loss = F.mse_loss(values_batch, value_targets_batch)
    #lux change 
    value_loss = F.mse_loss(values_batch.squeeze(), value_targets_batch.squeeze())

    # 3. --- Total Loss ---
    total_loss = (
        policy_loss + 
        value_loss_coeff * value_loss - 
        entropy_coeff * entropy
    )
    
    return total_loss, policy_loss, value_loss, entropy

# ==========================================================
# Test run
# ==========================================================
if __name__ == "__main__":
    """
    A simple test to make sure the loss calculation works.

    To run from your project root:
    python ragen/stage2_policy_opt.py
    """
    print("--- Testing PPO Loss (stage2_policy_opt.py) ---")

    # Simulate a 3-step trajectory (T=3) with 4 actions (A=4)
    T, A = 3, 4
    
    # Data from rollout
    logits_t = torch.randn(T, A) # New policy logits
    actions_t = torch.tensor([0, 1, 0]) # Actions taken
    old_log_probs_t = torch.tensor([-1.5, -1.2, -1.8]) # Old log_probs
    values_t = torch.tensor([0.1, 0.2, 0.5]) # Old V(s)
    
    # Data from stage1
    advantages_t = torch.tensor([0.76, 0.5, 0.5]) # Advantages
    value_targets_t = torch.tensor([0.96, 0.9, 1.0]) # V-star targets

    # Hyperparameters
    clip_eps = 0.2
    vf_coeff = 0.5
    ent_coeff = 0.01

    loss, p_loss, v_loss, ent = compute_ppo_loss(
        logits_t, actions_t, old_log_probs_t,
        advantages_t, value_targets_t, values_t,
        clip_eps, vf_coeff, ent_coeff
    )
    
    print(f"Total Loss:   {loss.item()}")
    print(f"Policy Loss:  {p_loss.item()}")
    print(f"Value Loss:   {v_loss.item()}")
    print(f"Entropy:      {ent.item()}")
    print("\n--- Test Complete ---")