# Omer Dekel

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Tuple

import numpy as np
import numpy.typing as npt

TState = TypeVar('TState')
TIndex = Tuple[int, ...]
TAction = TypeVar('TAction')


@dataclass
class ActionResult:
    result: TIndex
    probability: float


@dataclass
class Action(Generic[TAction]):
    action: TAction
    possibilities: Tuple[ActionResult, ...]


class MDP(ABC, Generic[TState, TAction]):
    def __init__(self, states: npt.NDArray[TState], rewards: npt.NDArray[np.float32], discount_factor: float):
        self._states = states
        self._rewards = rewards
        self._discount_factor = discount_factor

    @abstractmethod
    def actions(self, idx: TIndex) -> Tuple[Action[TAction], ...]:
        pass

    @abstractmethod
    def reward(self, idx: TIndex) -> np.float32:
        pass

    @property
    def states(self) -> npt.NDArray[TState]:
        return self._states

    @property
    def discount_factor(self) -> float:
        return self._discount_factor


def q_value(mdp: MDP, action: Action[TAction], utility_function: Callable[[TIndex], np.float32]) -> np.float32:
    q = np.float32(0)
    for possibility in action.possibilities:
        next_idx = possibility.result
        q += possibility.probability * (mdp.reward(next_idx) + mdp.discount_factor * utility_function(next_idx))
    return q
