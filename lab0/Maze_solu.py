import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Some colours
LIGHT_RED = '#FFC4CC';
LIGHT_GREEN = '#95FD99';
BLACK = '#000000';
WHITE = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

# Implemented methods
methods = ["DynProg", "ValIter"]


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give manes to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Rewards values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False):
        """Constructor of the environment Maze"""
        self.maze = maze
        self.actions = self._actions()
        self.states, self.map = self._states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self._transitions()
        self.rewards = self._rewards(weights=weights, random_rewards=random_rewards)

    def _actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def _states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    states[s] = (i,j)
                    map[(i,j)] = s
                    s += 1
        return states, map

    def _move(self, state, action):
        """ Make a step in maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x, y) on the maze that agent transition to.
        """
        # print("actions: ", self.actions)
        # print("action input: ", action)
        # print(self.actions[action])
        # print(self.actions[action][0])
        # print(self.states[state][0] + self.actions[action][0])
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) \
                             or (self.maze[row, col] == 1)

        if hitting_maze_walls:
            return state
        else:
            return self.map[(row, col)]

    def _transitions(self):
        """ compute the transition probabilities for every state-action pair
            :return numpy.tensor transition probabilities: tensor of transition probabilities of dimension S*S*A
        """
        dimentions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimentions)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self._move(s, a)
                transition_probabilities[next_s, s, a] = 1
        return transition_probabilities

    def _rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self._move(s, a)
                    # print("s: ", s, "next_s", next_s, "Maze[]", self.maze[self.states[next_s]])
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s]] == 2:
                        rewards[s, a] = self.GOAL_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]] < 0:
                        row, col = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s, a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s, a]
                        # The average reward
                        rewards[s, a] = 0.5 * r1 + 0.5 * r2
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self._move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = "ERROR: the argument method must be in {}'.format(methods)"
            raise NameError(error)

        path = list()
        if method == "DynProg":
            # deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # initialize current state and time
            t = 0
            s = self.map[start]
            # add the starting position in the maze to the path
            path.append(start)
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                next_s = self._move(s, policy[s,t])
                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # update time and state
                t = t + 1
                s = next_s
                if path[-2] == path[-1]:
                    path.pop(-1)
                    break
        if method == "ValIter":
            # initialize the time and state
            t = 1
            s = self.map[start]
            path.append(start)
            next_s = self._move(s, policy[s])
            path.append(self.states[next_s])
            # loop while state is not the goal state
            while s != next_s:
                s = next_s
                next_s = self._move(s, policy[s])
                path.append(self.states[next_s])
                t = t + 1
            if path[-1] == path[-2]:
                path.pop(-1)
        return path

    def show(self):
        print("The states are:", self.states)
        print("The actions are: ", self.actions)
        print("The mapping of the states: ", self.map)
        print("The rewards: ", self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env     : The maze environment in which we seek to find the shortest path
        :input int horizon  : The time T up to which we solve the problem
        :return numpy.array V      : Optimal values for every state at every time, dimension S*T
        :return numpy.array policy : Optimal time-varying policy at every state, dimension S*T
    """
    # Dynamic programming requires the knowledge of:
    # - state space
    # - action space
    # - transition probabilities
    # - rewards
    # - finite horizon

    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        policy[:, t] = np.argmax(Q, 1)

    return V, policy


def value_iteration(env, gamma, epsilon):
    """
    :param env:
    :param gamma: (float) the discount factor
    :param epsilon: (float) the accuracy of the value iteration procedure
    :return: numpy.array V[S*T]
    :return: numpy.array policy[S*T]
    """
    # VI requires the knowledge of:
    # - state space
    # - action space
    # - transition probabilities
    # - rewards

    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)

    # iteration counter
    n = 0
    # tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # initialization of VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        n += 1
        # update value function
        V = np.copy(BV)
        # compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # show error
        # print(np.linalg.norm(V-BV))

    # compute policy
    policy = np.argmax(Q, 1)
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows, cols = maze.shape;
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);

    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        if i > 0:
            if path[i] == path[i - 1]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(path[i - 1])].set_facecolor(col_map[maze[path[i - 1]]])
                grid.get_celld()[(path[i - 1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
