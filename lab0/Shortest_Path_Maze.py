from collections import deque

n_row = 6
n_col = 7


# coordinate of cells
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


# save the path info
class Queue:
    def __init__(self, p: Point, dist: int):
        self.p = p
        self.dist = dist


def isValid(row: int, col: int):
    return (row >= 0) and (row < n_row) and (col >= 0) and (col < n_col)


# get row and column number of 4 neighbouts of a given cell
rowNum = [-1, 0, 0, 1]
colNum = [0, -1, 1, 0]


def BFS(mazemap, src: Point, des: Point):
    visited = [[False for i in range(n_col)] for j in range(n_row)]  # initial all the cell as unvisited
    visited[src.x][src.y] = True  # initial source cell
    q = deque()  # queue for BFS
    s = Queue(src, 0)  # distance of source is 0
    q.append(s)
    path = [[src.x, src.y]]

    while q:
        print("length of q:", len(q))
        current = q.popleft()  # Dequeue the front cell
        p = current.p
        # reach the destination
        if p.x == des.x and p.y == des.y:
            return current.dist
        # otherwise move to adjacent cells
        for i in range(4):
            row = p.x + rowNum[i]
            col = p.y + colNum[i]
            if isValid(row, col) and mazemap[row][col] > -10 and not visited[row][col]:
                visited[row][col] = True
                adjcell = Queue(Point(row, col), current.dist + 1)
                print("current distance is: ", current.dist, ". the current coordinate is: (", current.p.x, current.p.y, ")")
                q.append(adjcell)

    return -1


if __name__ == '__main__':
    mazemap = [[0, 0, -100, 0, 0, 0, 0],
               [0, 0, -100, 0, 0, 0, 0],
               [0, 0, -100, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, -100, -100, -100, -100, -100, 0],
               [0, 0, 0, 0, 0, 0, 0]]
    source = Point(0, 0)
    destination = Point(5, 5)

    dist = BFS(mazemap, source, destination)

    if dist != -1:
        print("The shortest path is: ", dist)
    else:
        print("The shortest path doesn't exist.")
