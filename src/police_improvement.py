import numpy as np
from polices import PWalkSeven, piWalkSeven
from police_evaluation import policy_evaluation

V = policy_evaluation(piWalkSeven, PWalkSeven)

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])))

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}

    return new_pi

Q = policy_improvement(V, PWalkSeven)

new_V = policy_evaluation(Q, PWalkSeven)
