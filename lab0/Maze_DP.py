import numpy as np
from collections import deque

# state space S = {(i, j) cell(i,j) that without obstacles}
# action space A = {up, down, left, right, stay}
# transition probabilities P(s'|s,a) = 1 if s' is not obstacles but s'!= s
#                          P{s|s,a} = 1 if towards to wall or obstacles
# 



n_row = 6
n_col = 7


# coordinate of cells
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


# save the path info
class Queue:
    def __init__(self, p: Point, reward: int):
        self.p = p
        self.reward = reward


def isValid(row: int, col: int):
    return (row >= 0) and (row < n_row) and (col >= 0) and (col < n_col)


# get row and column number of 4 neighbouts of a given cell
rowNum = [-1, 0, 0, 1]
colNum = [0, -1, 1, 0]


def BFS(mazemap, rewardmap, src: Point, des: Point):
    visited = [[False for i in range(n_col)] for j in range(n_row)]  # initial all the cell as unvisited
    visited[src.x][src.y] = True  # initial source cell
    q = deque()  # queue for BFS
    s = Queue(src, 0)  # distance of source is 0
    q.append(s)
    rewardmap[0][0] = 0
    path = [[src.x, src.y]]

    while q:
        current = q.popleft()  # Dequeue the front cell
        p = current.p
        # reach the destination
        if p.x == des.x and p.y == des.y:
            print("current reward is: ", current.reward, ". the current coordinate is: (", current.p.x, ",",
                  current.p.y, ")")
            return current.reward, rewardmap
        # otherwise move to adjacent cells
        for i in range(4):
            row = p.x + rowNum[i]
            col = p.y + colNum[i]
            if isValid(row, col) and mazemap[row][col] > -50 and not visited[row][col]:
                visited[row][col] = True
                adjcell = Queue(Point(row, col), current.reward + mazemap[row][col])
                q.append(adjcell)
                rewardmap[row][col] = current.reward + mazemap[row][col]
                print("current reward is: ", current.reward, ". the current coordinate is: (", current.p.x, ",",
                      current.p.y, ")")

    return 1, []


# def opAction(newmp):
#     actionmap = [[[] for i in range(n_col)] for j in range(n_row)]
#     for i in n_row:
#         for j in n_col:
#             for k in range(4):
#                 row = i + rowNum[k]
#                 col = j + colNum[k]
#                 if isValid(row, col):
#
#     return actionmap


if __name__ == '__main__':
    mazemap = [[1, 1, -100, 1, 1, 1, 1],
               [1, 1, -100, 1, 1, 1, 1],
               [1, 1, -100, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, -100, -100, -100, -100, -100, 1],
               [1, 1, 1, 1, 1, 1, 1]]

    rewardmap = [[-100 for i in range(n_col)] for j in range(n_row)]

    source = Point(0, 0)
    destination = Point(5, 5)

    reward, newrewardmap = BFS(mazemap, rewardmap, source, destination)

    if reward != 1:
        print("The shortest path with reward: ", reward, "the new reward map is ", np.array(newrewardmap))
    else:
        print("The shortest path doesn't exist.")
