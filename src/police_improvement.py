# import numpy as np

# def policy_improvement(V, P, gamma=1.0):
#     Q = np.zeros(len(P), len(P[0]))

#     for s in range(len(P)):
#         for a in range(len(P[0])):
#             for prob, next_state, reward, done in P[s][a]:
#                 Q[s][a] = prob * (reward + gamma * V[next_state] * (not done))

#     new_pi = lambda s: {s for s in enumerate(np.argmax(Q, axis=1))}[s]

#     return new_pi

# Q