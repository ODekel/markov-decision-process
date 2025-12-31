# Omer Dekel

from typing import Tuple, List

import numpy as np
import numpy.typing as npt

from markov_decision_process import MDP, TState, TAction, q_value, Action


def policy_iteration(mdp: MDP[TState, TAction], initial_policy: npt.NDArray[Action[TAction]], epsilon: float
                     ) -> Tuple[npt.NDArray[Action[TAction]], npt.NDArray[np.float32], int, List[int], int]:
    policy = initial_policy
    utilities = np.zeros_like(mdp.states, dtype=np.float32)
    q_value_calls = 0
    unchanged = False
    iterations = 0
    simplified_iters_count = []
    while not unchanged:
        utilities, simplified_iters, eval_q_value_calls = policy_evaluation(policy, utilities, mdp, epsilon)
        q_value_calls += eval_q_value_calls
        simplified_iters_count.append(simplified_iters)
        unchanged = True
        for idx in np.ndindex(mdp.states.shape):
            # Only change to new action if it improves utility
            max_q = q_value(mdp, policy[idx], utility_function=lambda i: utilities[i])
            q_value_calls += 1
            max_action = policy[idx]
            # Search max Q-Value over all actions
            for action in mdp.actions(idx):
                action_q = q_value(mdp, action, utility_function=lambda i: utilities[i])
                q_value_calls += 1
                if action_q > max_q:
                    unchanged = False
                    max_q = action_q
                    max_action = action
            if max_action != policy[idx]:
                policy[idx] = max_action
        iterations += 1
    return policy, utilities, iterations, simplified_iters_count, q_value_calls


def policy_evaluation(policy: npt.NDArray[TAction], utilities: npt.NDArray[np.float32], mdp: MDP[TState, TAction],
                      epsilon: float) -> Tuple[npt.NDArray[np.float32], int, int]:
    q_value_calls = 0
    evaluated_utilities = np.zeros_like(utilities, dtype=np.float32)
    next_utilities = np.zeros_like(utilities, dtype=np.float32)
    # First pass against the current utilities
    for idx in np.ndindex(evaluated_utilities.shape):
        evaluated_utilities[idx] = q_value(mdp, policy[idx], lambda i: utilities[i])
        q_value_calls += 1
    delta = np.abs((evaluated_utilities - utilities)).max()
    iterations = 1
    while delta > epsilon * (1 - mdp.discount_factor) / mdp.discount_factor:
        # Update all utilities based on the previous ones
        for idx in np.ndindex(evaluated_utilities.shape):
            next_utilities[idx] = q_value(mdp, policy[idx], lambda i: evaluated_utilities[i])
            q_value_calls += 1
        delta = np.abs((next_utilities - evaluated_utilities)).max()
        evaluated_utilities = next_utilities.copy()
        iterations += 1
    return evaluated_utilities, iterations, q_value_calls
