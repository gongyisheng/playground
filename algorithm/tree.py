from typing import List
import collections


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class NrayTreeNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children else []


# Helper function to build a binary tree from a list of values
def build_binary_tree(vals: List[int]) -> TreeNode:
    def build_tree_helper(vals: List[int], idx: int) -> TreeNode:
        if idx >= len(vals) or vals[idx] is None:
            return None
        node = TreeNode(vals[idx])
        node.left = build_tree_helper(vals, 2 * idx + 1)
        node.right = build_tree_helper(vals, 2 * idx + 2)
        return node

    return build_tree_helper(vals, 0)


# def build_nray_tree(n: int, vals: List[int]) -> NrayTreeNode:
#     def build_tree_helper(vals: List[int], idx: int) -> NaryTreeNode:
#         if idx >= len(vals) or vals[idx] is None:
#             return None
#         node = NaryTreeNode(vals[idx])
#         for i in range(1, n+1):
#             child = build_tree_helper(vals, n*idx + i)
#             node.children.append(child)
#         return node

#     return build_tree_helper(vals, 0)


# LC 144
def preorder_traversal(root: TreeNode) -> List[int]:
    if root is None:
        return []
    vals = []
    vals.append(root.val)
    vals.extend(preorder_traversal(root.left))
    vals.extend(preorder_traversal(root.right))
    return vals


# LC 94
def inorder_traversal(root: TreeNode) -> List[int]:
    if root is None:
        return []
    vals = []
    vals.extend(inorder_traversal(root.left))
    vals.append(root.val)
    vals.extend(inorder_traversal(root.right))
    return vals


# LC 145
def postorder_traversal(root: TreeNode) -> List[int]:
    if root is None:
        return []
    vals = []
    vals.extend(postorder_traversal(root.left))
    vals.extend(postorder_traversal(root.right))
    vals.append(root.val)
    return vals


# LC 102
def levelorder_traversal(root: TreeNode) -> List[List[int]]:
    if root is None:
        return []
    vals = []
    queue = collections.deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        vals.append(level)
    return vals


# LC 429
def levelorder_traversal_nray(root: NrayTreeNode) -> List[List[int]]:
    if root is None:
        return []
    result = []
    queue = collections.deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            queue.extend(node.children)
        result.append(level)
    return result


if __name__ == "__main__":
    vals = [1, 2, 3, None, 4, 5, 6]
    root = build_binary_tree(vals)
    print("preorder: ", preorder_traversal(root))
    print("inorder: ", inorder_traversal(root))
    print("postorder: ", postorder_traversal(root))
    print("levelorder: ", levelorder_traversal(root))
