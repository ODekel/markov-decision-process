# Omer Dekel
from typing import Tuple

import numpy as np
import numpy.typing as npt

from markov_decision_process import MDP, TState, q_value, TAction, Action


# Implemented as in Figure 16.6, p. 563 in the book
def value_iteration(mdp: MDP[TState, TAction], epsilon: float
                    ) -> Tuple[npt.NDArray[Action[TAction]], npt.NDArray[np.float32], int, int]:
    utilities = np.zeros_like(mdp.states, dtype=np.float32)
    next_utilities = np.zeros_like(mdp.states, dtype=np.float32)
    policy = np.empty_like(mdp.states, dtype=object)
    for idx in np.ndindex(mdp.states.shape):
        policy[idx] = Action(action=mdp.states[idx], possibilities=())
    q_value_calls = 0
    delta = float('inf')
    iterations = 0
    while delta > epsilon * (1 - mdp.discount_factor) / mdp.discount_factor:
        utilities = next_utilities.copy()
        for idx in np.ndindex(mdp.states.shape):
            actions = mdp.actions(idx)
            if not actions:
                 max_q = utilities[idx]
            else:
                # Search max Q-Value over all actions
                max_q = float('-inf')
                max_action = None
                for action in actions:
                    action_q = q_value(mdp, action, utility_function=lambda i: utilities[i])
                    q_value_calls += 1
                    if action_q > max_q:
                        max_q = action_q
                        max_action = action
                policy[idx] = max_action
            next_utilities[idx] = max_q
        delta = np.abs((next_utilities - utilities)).max()
        iterations += 1
    return policy, utilities, iterations, q_value_calls
