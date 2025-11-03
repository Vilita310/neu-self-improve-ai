import sys
import os
import torch
from torch.distributions import Categorical
import json
from typing import Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.webshop_env import WebShopEnv
from ragen.ragen_loop import run_ragen_loop
from ragen.train_ragen_apo import ImprovedA2CAgent, SimpleTokenizer, DynamicActionSpace

# Add these aliases
A2C_Agent = ImprovedA2CAgent
ActionSpace = DynamicActionSpace


def load_trained_model(model_path='models/trained_agent.pth', device='cpu'):
    """Load trained agent from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = checkpoint['vocab']
    tokenizer.idx = max(checkpoint['vocab'].values()) + 1
    
    # Reconstruct action space
    action_space = ActionSpace()
    action_space.actions = checkpoint['action_space']
    action_space.action_to_idx = {a: i for i, a in enumerate(action_space.actions)}
    action_space.idx_to_action = {i: a for i, a in enumerate(action_space.actions)}
    
    # Reconstruct agent
    agent = A2C_Agent(
        vocab_size=len(tokenizer.vocab),
        action_space_size=action_space.size(),
        embedding_dim=64,   # MATCH TRAINING
        hidden_dim=128      # MATCH TRAINING
    ).to(device)
    
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    return agent, tokenizer, action_space


def evaluate_comprehensive(agent, tokenizer, action_space, device, num_episodes=100, max_obs_len=50):
    """Run comprehensive evaluation."""
    
    def policy_fn(observation: str) -> Tuple[str, str]:
        obs_tensor = tokenizer.tokenize(observation, max_obs_len).to(device)
        
        with torch.no_grad():
            logits, value = agent(obs_tensor)
            dist = Categorical(logits=logits)
            action_idx = dist.sample().item()
        
        action_str = action_space.get_action_str(action_idx)
        thought = f"Agent decides to: {action_str}"
        
        policy_fn.last_data = {
            "obs_tensor": obs_tensor,
            "action_idx_tensor": torch.tensor([action_idx]).to(device),
            "log_prob": dist.log_prob(torch.tensor([action_idx]).to(device)),
            "value": value.squeeze()
        }
        
        return thought, action_str
    
    # Run evaluation episodes
    successes = 0
    failures = 0
    total_reward = 0
    total_steps = 0
    failure_cases = []
    
    print(f"\n{'='*60}")
    print(f"Running {num_episodes} evaluation episodes...")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        env = WebShopEnv()
        trajectory, reward = run_ragen_loop(env, policy_fn)
        
        total_reward += reward
        total_steps += len(trajectory)
        
        if reward > 0:
            successes += 1
        else:
            failures += 1
            # Store failure case
            if len(failure_cases) < 10:  # Keep first 10 failures
                failure_cases.append({
                    'episode': episode + 1,
                    'trajectory': [
                        {'action': step['action'], 
                         'observation': step['next_observation'][:100]}
                        for step in trajectory
                    ],
                    'final_reward': reward
                })
        
        # Progress indicator
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes...")
    
    # Calculate metrics
    results = {
        'num_episodes': num_episodes,
        'success_rate': successes / num_episodes,
        'failure_rate': failures / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'avg_steps': total_steps / num_episodes,
        'total_successes': successes,
        'total_failures': failures,
        'failure_cases': failure_cases
    }
    
    return results


def print_results(results):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Episodes:     {results['num_episodes']}")
    print(f"Success Rate:       {results['success_rate']:.1%}")
    print(f"Failure Rate:       {results['failure_rate']:.1%}")
    print(f"Avg Reward:         {results['avg_reward']:.3f}")
    print(f"Avg Steps:          {results['avg_steps']:.1f}")
    print(f"Total Successes:    {results['total_successes']}")
    print(f"Total Failures:     {results['total_failures']}")
    
    print(f"\n{'='*60}")
    print("FAILURE CASE EXAMPLES (First 5)")
    print(f"{'='*60}")
    
    for i, case in enumerate(results['failure_cases'][:5], 1):
        print(f"\nFailure Case {i} (Episode {case['episode']}):")
        for step in case['trajectory']:
            print(f"  Action: {step['action']}")
            print(f"  Result: {step['observation']}")
        print(f"  Final Reward: {case['final_reward']}")
    
    print(f"\n{'='*60}\n")


def save_results(results, output_file='results/evaluation_results.json'):
    """Save results to JSON."""
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/best_agent.pth'
    num_episodes = 100
    
    print(f"Device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Load model
    try:
        agent, tokenizer, action_space = load_trained_model(model_path, device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using: python ragen/train_ragen_apo.py")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate_comprehensive(
        agent, tokenizer, action_space, device, 
        num_episodes=num_episodes
    )
    
    # Print and save results
    print_results(results)
    save_results(results)