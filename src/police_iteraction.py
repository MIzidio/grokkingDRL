import numpy as np

from polices import PWalkSeven, piWalkSeven
from police_evaluation import policy_evaluation
from police_improvement import policy_improvement

def police_iteraction(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = {s:a for s, a in enumerate(random_actions)}

    while True:

        V = policy_evaluation(pi, P, gamma, theta)
        new_pi = policy_improvement(V, P, gamma)

        if new_pi == pi:
            break

        pi = new_pi

    return V, pi