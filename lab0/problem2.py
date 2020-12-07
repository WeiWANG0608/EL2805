import numpy as np
from lab0 import Maze_solu

# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze
# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 2, 0]
])

# Description of the weight matrix as a numpy array
w = np.array([
    [0, 1, -100, 10, 10, 10, 10],
    [0, 1, -100, 10, 0, 0, 10],
    [0, 1, -100, 10, 0, 0, 10],
    [0, 1, 1, 1, 0, 0, 10],
    [0, -100, -100, -100, -100, -100, 10],
    [0, 0, 0, 0, 0, 11, 10]
])

env = Maze_solu.Maze(maze, weights=w)
for horizon in range(20,12,-1):

    V, policy = Maze_solu.dynamic_programming(env, horizon)
    method = "DynProg"
    start = (0, 0)
    path = env.simulate(start, policy, method)
    print("Path of DP: ", path)
    print("[DP with weight matrix] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                              "destination. ")
    # Discount Factor
    gamma = 0.50
    epsilon = 0.001
    V, policy = Maze_solu.value_iteration(env, gamma, epsilon)
    method = 'ValIter'
    start = (0, 0)
    path = env.simulate(start, policy, method)
    print("Path of VI: ", path)
    print("[VI with weight matrix] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                              "destination. ")
