import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Any, Tuple
import json

# Imports (same as before)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.webshop_env import WebShopEnv  # Use the FIXED environment
from ragen.ragen_loop import run_ragen_loop
from ragen.stage1_vstar import compute_gae_advantages
from ragen.stage2_policy_opt import compute_ppo_loss


# ============================================
# IMPROVED AGENT with Better Architecture
# ============================================
class ImprovedA2CAgent(nn.Module):
    """
    Simplified, more effective architecture with:
    1. Better input processing
    2. Separate actor/critic heads
    3. Layer normalization for stable training
    """
    def __init__(self, vocab_size: int, action_space_size: int, 
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super(ImprovedA2CAgent, self).__init__()
        
        # Smaller embedding - we don't need huge capacity
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Single LSTM layer is sufficient
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True,
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Separate processing for actor and critic
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_space_size)
        )
        
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state_indices: torch.Tensor):
        # Embed and process
        embedded = self.embedding(state_indices)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        features = self.layer_norm(hidden[-1])
        
        # Get policy and value
        action_logits = self.actor_fc(features)
        state_value = self.critic_fc(features)
        
        return action_logits, state_value


# ============================================
# DYNAMIC ACTION SPACE
# ============================================
class DynamicActionSpace:
    """
    Actions adapt to the environment state.
    Uses categorical search + parametric click/buy.
    """
    def __init__(self):
        # Core search strategies
        self.search_templates = [
            "search {category}",  # Category search
            "search {color} {category}",  # Color + category
            "search {modifier} {category}",  # Modifier + category
        ]
        
        # Common search terms
        self.categories = [
            "shoes", "running", "sneakers", "boots", "sandals",
            "electronics", "headphones", "watch", "charger",
            "apparel", "hoodie", "jeans", "jacket",
            "fitness", "yoga", "dumbbells",
            "home", "pillow", "lamp"
        ]
        
        self.colors = ["red", "blue", "black", "white", "green", "yellow"]
        self.modifiers = ["premium", "pro", "classic", "modern"]
        
        # Build action space
        self.actions = self._build_actions()
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
    
    def _build_actions(self):
        actions = []
        
        # Add search actions
        for cat in self.categories:
            actions.append(f"search {cat}")
        
        for color in self.colors:
            for cat in ["shoes", "hoodie", "headphones"]:
                actions.append(f"search {color} {cat}")
        
        # Add click/buy for IDs 1-100 (covers more products)
        for i in range(1, 101):
            actions.append(f"click {i}")
            actions.append(f"buy {i}")
        
        return actions
    
    def get_action_str(self, idx: int) -> str:
        return self.idx_to_action.get(idx, "search shoes")
    
    def size(self) -> int:
        return len(self.actions)


# ============================================
# TOKENIZER (Same as before)
# ============================================
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.idx = 2

    def build_vocab(self, texts: List[str]):
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = self.idx
                    self.idx += 1
    
    def tokenize(self, text: str, max_len: int) -> torch.Tensor:
        tokens = [
            self.vocab.get(word.lower(), self.vocab['<UNK>']) 
            for word in text.split()
        ]
        
        if len(tokens) < max_len:
            tokens.extend([self.vocab['<PAD>']] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


# ============================================
# BATCH ROLLOUT COLLECTION
# ============================================
def collect_batch_rollouts(env_fn, policy_fn, num_episodes: int, device):
    """
    Collect multiple episodes before updating.
    KEY FIX: Train on more data!
    """
    all_trajectories = []
    all_rewards = []
    
    for _ in range(num_episodes):
        env = env_fn()
        trajectory, total_reward = run_ragen_loop(env, policy_fn)
        all_trajectories.extend(trajectory)
        all_rewards.append(total_reward)
    
    # Convert to tensors
    obs_batch = []
    actions_batch = []
    log_probs_batch = []
    values_batch = []
    rewards_list = []
    dones_list = []
    
    for step in all_trajectories:
        data = step['policy_data']
        obs_batch.append(data['obs_tensor'])
        actions_batch.append(data['action_idx_tensor'])
        log_probs_batch.append(data['log_prob'])
        values_batch.append(data['value'])
        rewards_list.append(step['reward'])
        dones_list.append(1.0 if step['done'] else 0.0)
    
    return {
        'obs': torch.cat(obs_batch, dim=0).to(device),
        'actions': torch.cat(actions_batch, dim=0).to(device),
        'old_log_probs': torch.cat(log_probs_batch, dim=0).to(device),
        'values': torch.stack(values_batch, dim=0).to(device),
        'rewards': torch.tensor(rewards_list, dtype=torch.float32, device=device),
        'dones': torch.tensor(dones_list, dtype=torch.float32, device=device),
        'episode_rewards': all_rewards
    }


# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def train_ragen_apo_fixed():
    """
    FIXED training with:
    1. Batch collection (16 episodes per update)
    2. Dense rewards from fixed environment
    3. Better action space
    4. Higher exploration
    """
    print("=" * 60)
    print("FIXED RAGEN + A*PO Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # FIXED HYPERPARAMETERS
    num_epochs = 200  # More epochs
    episodes_per_update = 16  # BATCH COLLECTION!
    eval_every = 20
    max_obs_len = 100  # Longer obs for better state representation
    
    # RL parameters
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_loss_coeff = 0.5
    entropy_coeff = 0.1  # HIGHER for more exploration
    lr = 3e-4
    num_update_epochs = 4
    
    print(f"\nKey Fixes:")
    print(f"  ✓ Batch size: {episodes_per_update} episodes per update")
    print(f"  ✓ Dense reward shaping in environment")
    print(f"  ✓ Randomized targets for generalization")
    print(f"  ✓ Expanded action space (100 products)")
    print(f"  ✓ Higher entropy ({entropy_coeff}) for exploration")
    print("=" * 60)
    
    # Initialize
    action_space = DynamicActionSpace()
    
    # Build vocabulary from diverse examples
    tokenizer = SimpleTokenizer()
    sample_texts = [
        "User wants to buy red running shoes",
        "Found 10 results",
        "Viewing Nike Pro Sneakers Price $89",
        "Purchased correctly",
        "search blue headphones",
        "click 5",
        "GOOD SEARCH HIGHLY RELEVANT"
    ]
    tokenizer.build_vocab(sample_texts)
    
    print(f"\nVocab size: {len(tokenizer.vocab)}")
    print(f"Action space: {action_space.size()}")
    
    # Create agent
    agent = ImprovedA2CAgent(
        vocab_size=len(tokenizer.vocab),
        action_space_size=action_space.size(),
        embedding_dim=64,
        hidden_dim=128
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # Policy function
    def agent_policy_fn(observation: str):
        obs_tensor = tokenizer.tokenize(observation, max_obs_len).to(device)
        
        with torch.no_grad():
            logits, value = agent(obs_tensor)
            dist = Categorical(logits=logits)
            action_idx_tensor = dist.sample()
            log_prob = dist.log_prob(action_idx_tensor)
        
        action_str = action_space.get_action_str(action_idx_tensor.item())
        thought = f"Choosing: {action_str}"
        
        agent_policy_fn.last_data = {
            "obs_tensor": obs_tensor,
            "action_idx_tensor": action_idx_tensor,
            "log_prob": log_prob,
            "value": value.squeeze()
        }
        
        return thought, action_str
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    training_history = []
    best_reward = -float('inf')
    
    for epoch in range(num_epochs):
        # BATCH COLLECTION
        batch_data = collect_batch_rollouts(
            env_fn=lambda: WebShopEnv(),
            policy_fn=agent_policy_fn,
            num_episodes=episodes_per_update,
            device=device
        )
        
        avg_episode_reward = sum(batch_data['episode_rewards']) / len(batch_data['episode_rewards'])
        
        # Get final value for bootstrapping
        with torch.no_grad():
            last_obs = tokenizer.tokenize("done", max_obs_len).to(device)
            _, last_value = agent(last_obs)
        
        values_with_bootstrap = torch.cat([
            batch_data['values'], 
            last_value.squeeze(0)
        ])
        
        # Compute advantages
        advantages, value_targets = compute_gae_advantages(
            batch_data['rewards'],
            values_with_bootstrap,
            batch_data['dones'],
            gamma,
            gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple update epochs
        epoch_losses = []
        for _ in range(num_update_epochs):
            logits, new_values = agent(batch_data['obs'])
            
            total_loss, p_loss, v_loss, ent = compute_ppo_loss(
                logits,
                batch_data['actions'],
                batch_data['old_log_probs'],
                advantages,
                value_targets,
                new_values.squeeze(),
                clip_epsilon,
                value_loss_coeff,
                entropy_coeff
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Track progress
        training_history.append({
            'epoch': epoch + 1,
            'avg_reward': avg_episode_reward,
            'loss': avg_loss,
            'total_steps': len(batch_data['rewards'])
        })
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Avg Reward: {avg_episode_reward:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Steps: {len(batch_data['rewards'])}")
        
        # Save best model
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            os.makedirs('models', exist_ok=True)
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'vocab': tokenizer.vocab,
                'action_space': action_space.actions,
                'reward': best_reward
            }, 'models/best_agent.pth')
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Best Reward: {best_reward:.3f}")
    print("=" * 60)
    
    # Save final results
    os.makedirs('results', exist_ok=True)
    with open('results/fixed_training_results.json', 'w') as f:
        json.dump({
            'training_history': training_history,
            'best_reward': best_reward,
            'hyperparameters': {
                'episodes_per_update': episodes_per_update,
                'num_epochs': num_epochs,
                'lr': lr,
                'entropy_coeff': entropy_coeff
            }
        }, f, indent=2)
    
    return agent, tokenizer, action_space


if __name__ == "__main__":
    agent, tokenizer, action_space = train_ragen_apo_fixed()