# python container operations
1. deque
```
from collections import deque

# Create a deque
dq = deque([1, 2, 3])

# Append elements to the right
dq.append(4)

# Append elements to the left
dq.appendleft(0)

# Pop elements from the right
dq.pop()

# Pop elements from the left
dq.popleft()

# Extend deque on the right
dq.extend([5, 6])

# Extend deque on the left
dq.extendleft([-1, -2])
```
2. priority queue
```
import heapq

# Option 1. use tuple in pq
# first element is priority, second is object
# lower score has higher priority
pq = []

# Push elements into the priority queue
heapq.heappush(pq, (2, "task2"))

# Pop elements from the priority queue
priority, task = heapq.heappop(pq)

# Option 2. use object with customized compare function in pq
class Task:
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description

    def __lt__(self, other):
        return self.priority < other.priority  # Lower priority number has higher priority

    def __repr__(self):
        return f"({self.priority}, {self.description})"
```

# problem-solving strategies
[LC82](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii) while+while  
[LC101](https://leetcode.com/problems/symmetric-tree) post-order traverse, recursive  
[LC201](https://leetcode.com/problems/bitwise-and-of-numbers-range) observation, common suffix + bit observation  
[LC839](https://leetcode.com/problems/similar-string-groups) string comparison + union-find  
[LC3108](https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph) bit operations + union-find (note: -1&anything = anything, the more number you have, the less & sum you can get)  
[LC55](https://leetcode.com/problems/jump-game) dfs/bfs --> state compress, greedy  
[LC2379](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks) dfs? no, use sliding window  
[LC1976](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination) dfs/bfs? no, use dijkstra(find shortest route) + dp (count ways)  
[LC3169](https://leetcode.com/problems/count-days-without-meetings) sort + state compression, count segment  
[LC3394](https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections) sort + count segment * 2 times  
[LC2003](https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid) sort + medium, why medium? think about a group of dots on a line and you need to choose one point that sum of distance of all dots to that point is smallest. The point should be average (which is not possible in this problem) or medium  
[LC2140](https://leetcode.com/problems/solving-questions-with-brainpower) reverse order dp, ascending order does not work  
[LC2873](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i) 3 loop is fine, but can be done with one loop by recording max value, max diff, and max result  
[LC79](https://leetcode.com/problems/word-search) typical grid dfs, tricks: a. do all validation before exploring b. change path value to # to mark it's visited.  
[LC98](https://leetcode.com/problems/validate-binary-search-tree) inorder + check trend
[LC113](https://leetcode.com/problems/path-sum-ii) preorder dfs + path sum
[LC117](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii) bfs, level order  
[LC1123](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves) post order dfs, return ans_node and depth at the same time  
[LC235](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree) dfs, return lca, use the property of BST  
[LC236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree) dfs, return lca, compare it with LC235  
[LC130](https://leetcode.com/problems/surrounded-regions) dfs, avoid methods that may cause wrong updates (should update only if all return true, but make updates before get result and cannot rollback).  
[LC310](https://leetcode.com/problems/minimum-height-trees) mht, cut leaves. 2 ways to get medium of a list: by index or cut leafs recursively  
[LC1863](https://leetcode.com/problems/sum-of-all-subset-xor-totals) dfs to traverse all subset, add xor sum  
[LC23](https://leetcode.com/problems/merge-k-sorted-lists) merge and sort, be careful is len(lists) == 0  
[LC215](https://leetcode.com/problems/kth-largest-element-in-an-array) mini heap, push then pop  
[LC1054](https://leetcode.com/problems/distant-barcodes) heap, pick most frequent first  
