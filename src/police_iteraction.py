import numpy as np

from polices import PWalkSeven, piWalkSeven
from police_evaluation import policy_evaluation
from police_improvement import policy_improvement

def police_iteraction(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = {s:a for s, a in enumerate(random_actions)}

    while True:

        old_pi = {s:pi[s] for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)

        if old_pi == {s:pi[s] for s in range(len(P))}:
            break

    return V, pi