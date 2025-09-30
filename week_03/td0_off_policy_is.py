# td0_off_policy_is.py
# Off-policy TD(0) control via importance-sampled SARSA.
# Implements both ordinary (unweighted) and weighted IS variants.
# Target policy: greedy wrt Q. Behavior: epsilon-greedy wrt Q with eps_b.

from __future__ import annotations
import numpy as np
from windy_gridworld import WindyGridworld, ACTIONS

def greedy(Q, s): return int(np.argmax(Q[s]))

def behavior_probs(Q, s, eps_b):
    a_star = np.argmax(Q[s])
    probs = np.ones(len(ACTIONS)) * (eps_b / len(ACTIONS))
    probs[a_star] += 1.0 - eps_b
    return probs

def td0_off_policy_is(env: WindyGridworld, episodes=8000, eps_b=0.2, alpha=0.1, gamma=1.0, seed=42, weighted=False):
    rng = np.random.RandomState(seed)
    Q = {s: np.zeros(len(ACTIONS)) for s in env.states()}
    for _ in range(episodes):
        s = env.reset()
        done = False
        # choose a from behavior
        bprobs = behavior_probs(Q, s, eps_b)
        a = rng.choice(len(ACTIONS), p=bprobs)
        while not done:
            ns, r, done = env.step(s, a)
            # importance ratio for (s,a)
            pi_a = 1.0 if a == greedy(Q, s) else 0.0
            b_a = bprobs[a]
            rho = 0.0 if b_a == 0 else (pi_a / b_a)
            if done:
                target = r
            else:
                # choose next a from behavior for SARSA
                bprobs_next = behavior_probs(Q, ns, eps_b)
                na = rng.choice(len(ACTIONS), p=bprobs_next)
                target = r + gamma * Q[ns][na]
            err = target - Q[s][a]
            if weighted:
                Q[s][a] += (alpha * rho) * err
            else:
                Q[s][a] += alpha * rho * err
            if done:
                break
            s, a, bprobs = ns, na, bprobs_next
    pi = {s: greedy(Q, s) for s in Q}
    return Q, pi
