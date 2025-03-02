import numpy as np
from polices import PWalkSeven, piWalkSeven

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.random.rand(len(P))

    while True:
        V = np.random.rand(len(P))

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if (np.max(np.abs(prev_V - V)) < theta):
            break
        prev_V = V.copy()

    return prev_V

V = policy_evaluation(piWalkSeven, PWalkSeven)
print(V)

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros(len(P), len(P[0]))

    for s in range(len(P)):
        for a in range(len(P[0])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] = prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

Q = policy_improvement(V, PWalkSeven)
print(Q)


