
"""A*-PO Algorithm Implementation (Simplified for CPU)"""

def a_star_po_loss(log_ratio, reward, v_star, beta=1.0):
    """Compute A*-PO regression loss: (beta * log_ratio - (r - V*))^2"""
    return (beta * log_ratio - (reward - v_star)) ** 2

def offline_estimate_vstar(samples):
    """Estimate value function V* by averaging rewards."""
    return sum(s["reward"] for s in samples) / len(samples)
