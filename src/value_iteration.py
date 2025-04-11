import numpy as np
from polices import PWalkSeven
import gymnasium as gym

# env = gym.make("BanditWalk-v0")
gym.pprint_registry()

def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.random.rand(len(P))

    while True:
        Q = np.zeros((len(P), len(P[0])))

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + V[next_state] * gamma * (not done))

        new_V = np.max(Q, axis=1)

        if np.max(np.abs(V - new_V)) < theta:
            break

        V = new_V

    pi = {s:a for s, a in enumerate(V)}

    return V, pi

# [V, pi] = value_iteration(PWalkSeven)
# print(V)
# print(pi)

        