import numpy as np
import matplotlib.pyplot as plt
from lab1 import rob_bank_func as rbf

# starting point of police is (1,2)
# starting point of you is (0,0)


city = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

start = (0, 0, 1, 2)

env = rbf.Town(city, start)
# env.show()
epsilon = 0.0001
initial_value = []
n_iter = 1000
state = env.map[start]

initial_value = []
for dis_factor in np.arange(0.01, 0.99, 0.02):
    V, policy, Q = rbf.value_iteration(env, dis_factor, epsilon)
    # initial_value.append(np.max(Q[state]))
    initial_value.append(V[0])  # value function (evaluated at the initial state)

lambda_x = np.arange(0.01, 0.99, 0.02)

fig = plt.figure(figsize=(8, 6))
plt.plot(lambda_x, initial_value)
plt.title("Lambda_interval_0.02")
plt.xlabel("Discount factor - lambda")
plt.ylabel("Value function_0.02_V[0]_(evaluated at initial state)")
fig.savefig("Robbing_bank_0.02_V[0].png", dpi=fig.dpi)
plt.show()


initial_value = []
for dis_factor in np.arange(0.05, 0.99, 0.05):
    V, policy, Q = rbf.value_iteration(env, dis_factor, epsilon)
    # initial_value.append(np.max(Q[state]))
    initial_value.append(V[0])  # value function (evaluated at the initial state)

lambda_x = np.arange(0.05, 0.99, 0.05)

fig = plt.figure(figsize=(8, 6))
plt.plot(lambda_x, initial_value)
plt.title("Lambda-comment_0.05")
plt.xlabel("Discount factor - lambda")
plt.ylabel("Value function_0.05_V[0]_(evaluated at initial state)")
fig.savefig("Robbing_bank_0.05_V[0]_.png", dpi=fig.dpi)
plt.show()
