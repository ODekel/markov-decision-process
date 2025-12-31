# Omer Dekel

import sys
from typing import Tuple, List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from markov_decision_process import MDP, Action, ActionResult
from policy_iteration import policy_iteration
from value_iteration import value_iteration

TAction = np.int8
TIndex = Tuple[int, int]
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
UP_ID = np.int8(1)
DOWN_ID = np.int8(2)
LEFT_ID = np.int8(3)
RIGHT_ID = np.int8(4)
ALL_ACTIONS = ((UP, (LEFT, RIGHT), UP_ID),
               (DOWN, (LEFT, RIGHT), DOWN_ID),
               (LEFT, (UP, DOWN), LEFT_ID),
               (RIGHT, (UP, DOWN), RIGHT_ID))


def _add_indices(a: Tuple[int, ...], b: Tuple[int, ...]) -> TIndex:
    return a[0] + b[0], a[1] + b[1]


class MamanMDP(MDP[np.int32, TAction]):
    def __init__(self, states: npt.NDArray[np.int8], rewards: npt.NDArray[np.float32], discount_factor: float,
                 success_probability: float):
        super().__init__(states, rewards, discount_factor)
        # Actions don't change during a run, so precompute the actions for all states
        self._actions = self._calculate_actions(success_probability)

    def actions(self, idx: TIndex) -> Tuple[Action[TAction], ...]:
        return self._actions[idx]

    def reward(self, idx: TIndex) -> np.float32:
        return self._rewards[idx]

    def _calculate_actions(self, success_probability: float) -> npt.NDArray[object]:
        actions = np.empty_like(self._states, dtype=object)
        rows, cols = self._states.shape
        failure_move_prob = (1.0 - success_probability) / 2.0
        valid_cells_mask = self._states == 1
        for idx in np.ndindex(actions.shape):
            if not valid_cells_mask[idx]:
                actions[idx] = ()
                continue
            possible_actions = []
            for action, sides, aid in ALL_ACTIONS:
                # Calculate the resulting index of the action
                result = _add_indices(idx, action)
                if not self._is_valid_cell(result, rows, cols):
                    continue
                action_possibilities = []
                action_success_prob = success_probability
                # 2 sides for each action
                for side_idx in range(2):
                    fail_result = _add_indices(result, sides[side_idx])
                    if self._is_valid_cell(fail_result, rows, cols):
                        action_possibilities.append(ActionResult(result=fail_result, probability=failure_move_prob))
                    else:
                        action_success_prob += failure_move_prob
                action_possibilities.append(ActionResult(result=result, probability=action_success_prob))
                possible_actions.append(Action(action=aid, possibilities=tuple(action_possibilities)))
            actions[idx] = tuple(possible_actions)
        return actions

    def _is_valid_cell(self, idx: TIndex, rows: int, cols: int) -> bool:
        return (0 <= idx[0] < rows) and (0 <= idx[1] < cols) and (self._states[idx] != 0)


def _print_policy(plc: npt.NDArray[Action[TAction]]) -> None:
    display_policy = np.zeros_like(plc, dtype=np.uint8)
    for idx in np.ndindex(plc.shape):
        if plc[idx].action == -1:
            display_policy[idx] = ord('o')
        elif plc[idx].action == 0:
            display_policy[idx] = ord('x')
        elif plc[idx].action == UP_ID:
            display_policy[idx] = ord('^')
        elif plc[idx].action == DOWN_ID:
            display_policy[idx] = ord('v')
        elif plc[idx].action == LEFT_ID:
            display_policy[idx] = ord('<')
        elif plc[idx].action == RIGHT_ID:
            display_policy[idx] = ord('>')
    np.savetxt(sys.stdout, display_policy, fmt='%c', delimiter='')
    print()


def _plot_utilities(utils: npt.NDArray[np.float32], method: str, iters: int, name: str) -> None:
    plt.figure(2)
    max_abs_reward = np.max(np.abs(utils))
    plt.imshow(utils, vmin=-max_abs_reward, vmax=max_abs_reward, cmap='seismic')
    plt.title(f'{method} utilities - {iters} iterations - {name}')
    plt.colorbar()
    plt.savefig(f'{method}_Values_{name}.jpg')
    plt.show()


def _plot_simplified_iterations(simplified_iterations: List[int], name: str) -> None:
    plt.figure(1)
    plt.plot(range(1, len(simplified_iterations) + 1), simplified_iterations, marker='o')
    plt.title(f'Policy Iteration - Simplified Iterations per Policy Evaluation - {name}')
    plt.xlabel('Policy Iteration')
    plt.ylabel('Number of Simplified Iterations')
    plt.savefig(f'PolicyIteration_SimplifiedIterations_{name}.jpg')


args = sys.argv[1:]
if len(args) != 2:
    print('Enter filename and iteration method')
    exit()
data = np.load(args[0])
my_mdp = MamanMDP(states=data['states'], rewards=data['rewards'].astype(np.float32),
                 discount_factor=0.9, success_probability=0.8)
if args[1] == 'ValueIteration':
    policy, utilities, iterations, q_value_calls = value_iteration(my_mdp, epsilon=0.1)
elif args[1] == 'PolicyIteration':
    initial_policy = np.empty_like(my_mdp.states, dtype=object)
    for pidx in np.ndindex(initial_policy.shape):
        if my_mdp.states[pidx] == 1:
            # When possible, UP is the first action
            initial_policy[pidx] = my_mdp.actions(pidx)[0]
        else:
            initial_policy[pidx] = Action(action=my_mdp.states[pidx], possibilities=())
    policy, utilities, iterations, simplified_iters, q_value_calls =(
        policy_iteration(my_mdp, initial_policy=initial_policy, epsilon=0.1))
    _plot_simplified_iterations(simplified_iters, 'Omer Dekel')
else:
    print('Enter valid method')
    exit()

_print_policy(policy)
print("Q-value function calls:", q_value_calls)
_plot_utilities(utilities, args[1], iterations, 'Omer Dekel')
