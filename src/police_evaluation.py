import numpy as np
from polices import PWalkSeven, piWalkSeven

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.random.rand(len(P))

    while True:
        new_V = np.zeros(len(P))

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi[s]]:
                new_V[s] += prob * (reward + gamma * V[next_state] * (not done))
        if (np.max(np.abs(new_V - V)) < theta):
            break
        V = new_V

    return V

