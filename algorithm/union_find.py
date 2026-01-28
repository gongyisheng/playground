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

class WeightedUnionFind:
    def __init__(self):
        self.parent = {}
        self.weight = {} # define weight[x] as the ratio of x/root_x
    
    def find(self, x):
        if x not in self.parent:
            return None
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.weight[x] *= self.weight[original_parent] # multiply by w(orig_parent) to keep weight as x/new_root_x
        return self.parent[x]
    
    def union(self, x, y, value):
        if x not in self.parent:
            self.parent[x] = x
            self.weight[x] = 1.0
        if y not in self.parent:
            self.parent[y] = y
            self.weight[y] = 1.0
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
            self.weight[root_x] = value * self.weight[y] / self.weight[x] # calc new weight of root_x as ratio of root_x/root_y
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def ratio(self, x, y):
        if self.connected(x, y):
            return self.weight[x] / self.weight[y]
        else:
            return -1

def test_uf():
    uf = UnionFind(10)
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.connected(1, 3)
    assert not uf.connected(1, 4)
    print("uf test passed")

def test_weighted_uf():
    uf = WeightedUnionFind()
    uf.union("a", "b", 2)
    uf.union("b", "c", 2)
    assert uf.ratio("a", "c") == 4
    assert uf.ratio("a", "d") == -1
    print("weighted uf test passed")

if __name__ == "__main__":
    test_uf()
    test_weighted_uf()
