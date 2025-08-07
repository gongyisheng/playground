import ast
import astpretty
import zss

# pip install ast astpretty zss

# Example of AST and AST Edit Distance
# It's used in phi-1 model evaluation to deep dedup finetune dataset
# ref to llm/paper_read/phi-1.md

# An AST is a tree representation of the structure of source code

# Edit Distance is like string edit distance (Levenshtein distance), 
# The total number of operations needed to convert one AST into another is the distance.

# AST Match Rate τ: τ(T1,T2) = 1 - (EditDistance(T1,T2))/max(|T1|,|T2|), |T1| is node number in tree.
# 1 → trees are identical. 0 → trees are completely different.

class ASTNode(zss.Node):
    def __init__(self, node):
        self.node = node
        self.label = type(node).__name__

    def get_children(self):
        children = []
        for field, value in ast.iter_fields(self.node):
            if isinstance(value, ast.AST):
                children.append(ASTNode(value))
            elif isinstance(value, list):
                children.extend(ASTNode(v) for v in value if isinstance(v, ast.AST))
        return children

    def get_label(self):
        # Include key info (e.g., function name) for more meaningful distance
        if isinstance(self.node, ast.FunctionDef):
            return f"FunctionDef:{self.node.name}"
        if isinstance(self.node, ast.Name):
            return f"Name:{self.node.id}"
        return type(self.node).__name__

code_a = """
def add(a, b):
    return a + b
"""

code_b = """
def sum(a, b):
    return a + b
"""


print(ast.dump(ast.parse(code_a)))
print(ast.dump(ast.parse(code_b)))

# Pretty print the ASTs
tree_a = ast.parse(code_a)
tree_b = ast.parse(code_b)

print("=== AST of Code A ===")
astpretty.pprint(tree_a)

print("\n=== AST of Code B ===")
astpretty.pprint(tree_b)

# edit distance
tree_a = ASTNode(ast.parse(code_a))
tree_b = ASTNode(ast.parse(code_b))
distance = zss.simple_distance(
    tree_a, tree_b,
    get_children=ASTNode.get_children,
    get_label=ASTNode.get_label
)
print(distance)