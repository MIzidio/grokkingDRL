import random
import gymnasium as gym
import numpy as np

from typing import Optional

class BanditSlipperyWalk(gym.Env):
    def __init__(self):
        super().__init__()
        # self.observation_space = gym.spaces.Discrete(3)

        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.state = 1

        return self.state

    def step(self, action):
        action_probabilities = {
            0: np.array([(0, 0.8), (2, 0.2)]),
            1: np.array([(2, 0.8), (0, 0.2)]),
        }
            
        if self.state == 1:
            states, probabilities = zip(*action_probabilities[action])
            next_state = self.np_random.choice(states, p=probabilities)
        else:
            next_state = None

        reward = 1 if next_state == 2 else 0

        return next_state, reward, True, False, {}

    