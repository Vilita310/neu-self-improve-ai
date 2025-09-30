# run_all.py
# Simple runner to:
# 1) Replicate Fig 4.1 values (prints matrix).
# 2) Build a Windy Gridworld and run: DP control, MC on/off, TD on/off (unweighted & weighted).

import numpy as np
from gridworld_base import Gridworld4x4, iterative_policy_evaluation
from windy_gridworld import WindyGridworld, WGConfig
from dp_control import policy_iteration
from mc_on_policy import mc_on_policy_control
from mc_off_policy import mc_off_policy_control
from td0_on_policy import sarsa_control
from td0_off_policy_is import td0_off_policy_is

def make_default_env(seed=0):
    cfg = WGConfig(
        H=7, W=10,
        wind_by_col=[0,0,0,1,2,1,0,0,1,0],
        start=(3,0),
        goal=(3,7),
        walls=set(),
        gamma=1.0,
        stochastic=True
    )
    return WindyGridworld(cfg, seed=seed)

if __name__ == "__main__":
    # Part 1: Figure 4.1 replication
    env4 = Gridworld4x4()
    V = iterative_policy_evaluation(env4, theta=1e-8)
    np.set_printoptions(precision=1, suppress=True)
    print("Figure 4.1 replication (4x4, equiprobable policy):")
    print(V)

    # Part 2: Windy Gridworld controls
    env = make_default_env(seed=0)

    print("\nDP control (policy iteration) on Windy Gridworld...")
    V_dp, pi_dp = policy_iteration(env, gamma=env.cfg.gamma, theta=1e-6, max_iter=100)
    print("DP value sample at start:", V_dp[env.start[0], env.start[1]])

    print("\nMC on-policy control...")
    Q_on, pi_on = mc_on_policy_control(env, episodes=4000, eps=0.1, gamma=env.cfg.gamma, seed=1)
    print("On-policy MC action at start:", pi_on[env.start])

    print("\nMC off-policy control (ordinary IS)...")
    Q_off, pi_off = mc_off_policy_control(env, episodes=8000, eps_b=0.2, gamma=env.cfg.gamma, seed=2)
    print("Off-policy MC action at start:", pi_off[env.start])

    print("\nTD(0) on-policy control (SARSA)...")
    Q_sarsa, pi_sarsa = sarsa_control(env, episodes=6000, eps=0.1, alpha=0.1, gamma=env.cfg.gamma, seed=3)
    print("SARSA action at start:", pi_sarsa[env.start])

    print("\nTD(0) off-policy control (ordinary IS)...")
    Q_td_off_u, pi_td_off_u = td0_off_policy_is(env, episodes=8000, eps_b=0.2, alpha=0.1, gamma=env.cfg.gamma, seed=4, weighted=False)
    print("Off-policy TD (unweighted IS) action at start:", pi_td_off_u[env.start])

    print("\nTD(0) off-policy control (weighted IS)...")
    Q_td_off_w, pi_td_off_w = td0_off_policy_is(env, episodes=8000, eps_b=0.2, alpha=0.1, gamma=env.cfg.gamma, seed=5, weighted=True)
    print("Off-policy TD (weighted IS) action at start:", pi_td_off_w[env.start])
