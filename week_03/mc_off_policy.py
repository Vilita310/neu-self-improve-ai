# mc_off_policy.py
# Off-policy MC control using ordinary importance sampling.
# Behavior policy: epsilon-soft; Target policy: greedy wrt Q.

from __future__ import annotations
import numpy as np
from windy_gridworld import WindyGridworld, ACTIONS

def greedy_action(Q, s):
    return int(np.argmax(Q[s]))

def eps_soft_probs(Q, s, eps):
    a_star = np.argmax(Q[s])
    probs = np.ones(len(ACTIONS)) * (eps / len(ACTIONS))
    probs[a_star] += 1.0 - eps
    return probs

def mc_off_policy_control(env: WindyGridworld, episodes=10000, eps_b=0.2, gamma=1.0, seed=1):
    rng = np.random.RandomState(seed)
    Q = {s: np.zeros(len(ACTIONS)) for s in env.states()}
    for _ in range(episodes):
        # generate with behavior b (epsilon-soft wrt current Q)
        s = env.reset()
        traj = []
        done = False
        while not done:
            b = eps_soft_probs(Q, s, eps_b)
            a = rng.choice(len(ACTIONS), p=b)
            ns, r, done = env.step(s, a)
            traj.append((s,a,r,b[a]))
            s = ns
        # ordinary importance sampling update (per-episode, backward)
        G = 0.0
        W = 1.0
        for t in reversed(range(len(traj))):
            s,a,r,bprob = traj[t]
            G = r + gamma * G
            # target is greedy wrt Q
            pi_a = 1.0 if a == greedy_action(Q, s) else 0.0
            if bprob == 0.0:
                break
            W *= (pi_a / bprob)
            Q[s][a] += (W) * (G - Q[s][a]) * 0.01  # small step for stability
            if pi_a == 0.0:
                break  # importance weight becomes zero beyond this point
    pi = {s: greedy_action(Q, s) for s in Q}
    return Q, pi
