import numpy as np
from lab0 import Maze_solu

# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 2, 0]
])

env = Maze_solu.Maze(maze)
env.show()

horizon = 10
V, policy = Maze_solu.dynamic_programming(env, horizon)
method = "DynProg"
start = (0, 0)
path = env.simulate(start, policy, method)
print("Path of DP: ", path)
print("[DP without random reward] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                             "destination. ")

# Value iteration
gamma = 0.95
epsilon = 0.0001
V, policy = Maze_solu.value_iteration(env, gamma, epsilon)
method = "ValIter"
start = (0, 0)
path = env.simulate(start, policy, method)
print("Path of VI: ", path)
print("[VI without random reward] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                             "destination. ")

# with the convention
#  0 = empty cell
#  1 = obstacle
#  2 = exit of the Maze
# -n = trapped cell with probability 0.5. If the cell is trapped the player must stay there for n times.
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1],
    [0, 1, 1, 1, 1, 1, 0],
    [-6, 0, 0, 0, 0, 2, 0]
])

env = Maze_solu.Maze(maze, random_rewards=True)
horizon = 15
V, policy = Maze_solu.dynamic_programming(env, horizon)
method = "DynProg"
start = (0, 0)
path = env.simulate(start, policy, method)
print("Path of DP with random reward: ", path)
print("[DP with random reward] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                          "destination. ")

# Value iteration
# gamma = 0.95
# epsilon = 0.0001
V, policy = Maze_solu.value_iteration(env, gamma, epsilon)
method = "ValIter"
start = (0, 0)
path = env.simulate(start, policy, method)
print("Path of VI with random reward: ", path)
print("[VI with random reward] The total horizon is: ", horizon, ". It took ", len(path)-1, "steps to reach the "
                                                                                          "destination. ")
