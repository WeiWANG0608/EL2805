import numpy as np
import matplotlib.pyplot as plt
from lab1 import bank_rob_reloaded_func as rd
city = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
start = (0, 0, 3, 3)
env = rd.City(city, police_cant_stay=False)
state = env.map[start]

Q = rd.QLearning(env, start, state)

epsilon_start_1 = 0.1  # only plot eps = 0.1
Qs = rd.SARSA(env, start, state, epsilon_start_1)

epsilon_start = 0.05  # (0.05, 0.1, 0.15, 0.2, 0.25)
Qs = rd.SARSA(env, start, state, epsilon_start)



