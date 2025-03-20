# Complexity:
#   Find: O(α(n)) (almost constant time due to path compression)
#   Union: O(α(n)) (same reason)
#   Connected: O(α(n)) (since it just calls Find)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each node is its own parent initially
        self.rank = [1] * n  # Rank (tree height) is initially 1

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Attach smaller tree under the larger tree
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

if __name__ == "__main__":
    uf = UnionFind(10)
    uf.union(1, 2)
    uf.union(2, 3)
    print(uf.connected(1, 3))  # True
    print(uf.connected(1, 4))  # False

