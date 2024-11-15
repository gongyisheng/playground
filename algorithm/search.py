board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
from collections import deque

def dfs_main(target_word):
    for i in range(len(board)):
        for j in range(len(board[0])):
            res = dfs(target_word, 0, i, j)
            if res:
                return True
    return False

def dfs(target_word, index, x, y):
    if index == len(target_word):
        return True
    if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]) or board[x][y] != target_word[index]:
        return False
    left = dfs(target_word, index + 1, x, y - 1)
    right = dfs(target_word, index + 1, x , y + 1)
    up = dfs(target_word, index + 1, x - 1, y)
    down = dfs(target_word, index + 1, x + 1, y)
    return left or right or up or down

def bfs_main(target_word):
    queue = deque()
    for i in range(len(board)):
        for j in range(len(board[0])):
            queue.append((i, j, 0))
    
    while queue:
        x, y, index = queue.popleft()
        if index == len(target_word):
            return True
        if x < 0 or x >= len(board) or y < 0 or y >= len(board[0]) or board[x][y] != target_word[index]:
            continue
        queue.append((x, y - 1, index + 1))
        queue.append((x, y + 1, index + 1))
        queue.append((x - 1, y, index + 1))
        queue.append((x + 1, y, index + 1))
    return False

if __name__ == "__main__":
    target_word = "ABCCED"
    print(dfs_main(target_word))
    print(bfs_main(target_word))