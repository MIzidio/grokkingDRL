import random
import gymnasium as gym
import numpy as np

from typing import Optional

class BanditSlipperyWalk(gym.Env):
    def __init__(self, size: int = 3):
        self.size = size

        self._agent_location = np.array([0], dtype=np.int32)
        self._target1_location = np.array([0], dtype=np.int32)
        self._target2_location = np.array([0], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Discrete(3),
                "target1": gym.spaces.Discrete(3),
                "target2": gym.spaces.Discrete(3),
            }
        )

        self.action_space = gym.spaces.Discrete(2)

        self._action_probabilities = {
            0: np.array([(0, 0.8), (1, 0.2)]),
            1: np.array([(1, 0.8), (0, 0.2)]),
        }

        self._action_to_direction = {
            0: np.array([-1]),
            1: np.array([1]),
        }

    def _get_orbs(self):
        return {
            "agent": self._agent_location, 
            "target1": self._target1_location, 
            "target2": self._target2_location
        }

    def stochastic_probabilities(self, action):
        actions, probabilities = zip(*self._action_probabilities[action])
        return self.np_random.choice(actions, p=probabilities)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = np.array([1])

        self._target1_location = np.array([0])
        self._target2_location = np.array([2])


        observation = self._get_orbs()

        return observation

    def step(self, action):
        self._real_action = self.stochastic_probabilities(action)

        direction = self._action_to_direction[self._real_action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated1 = np.array_equal(self._agent_location, self._target1_location)
        terminated2 = np.array_equal(self._agent_location, self._target2_location)
        terminated = terminated1 or terminated2
        truncated = False
        reward = 1 if terminated2 else 0
        observation = self._get_orbs
        info = {}

        return observation, reward, terminated, truncated, info

    