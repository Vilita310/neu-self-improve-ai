# replicate_figure_4_1.py
# Replicates S&B Fig 4.1 state values for the 4x4 gridworld under the equiprobable random policy.
# Prints the value function in a 4x4 grid.

import numpy as np
from gridworld_base import Gridworld4x4, iterative_policy_evaluation

if __name__ == "__main__":
    env = Gridworld4x4()
    V = iterative_policy_evaluation(env, theta=1e-6)
    np.set_printoptions(precision=1, suppress=True)
    print("State-value function V (approx):")
    print(V)
