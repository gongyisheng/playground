from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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
def levelorder_traversal(root: TreeNode) -> List[int]:
    vals = []
    queue = []
    if root:
        queue.append(root)
    while queue:
        node = queue.pop(0)
        vals.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return vals

def build_tree(vals: List[int]) -> TreeNode:
    def build_tree_helper(vals: List[int], idx: int) -> TreeNode:
        if idx >= len(vals) or vals[idx] is None:
            return None
        node = TreeNode(vals[idx])
        node.left = build_tree_helper(vals, 2 * idx + 1)
        node.right = build_tree_helper(vals, 2 * idx + 2)
        return node
    return build_tree_helper(vals, 0)

if __name__ == "__main__":
    vals = [1, 2, 3, None, 4, 5, 6]
    root = build_tree(vals)
    print("preorder: ", preorder_traversal(root))
    print("inorder: ", inorder_traversal(root))
    print("postorder: ", postorder_traversal(root))
    print("levelorder: ", levelorder_traversal(root))