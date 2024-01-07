# code generation
1. factorial number - user request part  
`Write a Python function to calculate the factorial of a given number`
- `Please provide the recursion version.`
- `Please provide the non-recursion version.`  
```
# test case:
factorial(5) = 120
```

2. DFS - user request part
`Write a Python function to perform Deep First Search.`  
- `Please provide the recursion version.`
- `Please provide the non-recursion version.`  
```
# test case:  
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': [],
    'F': []
}

print(dfs(graph, 'A'))  # Output: ['A', 'B', 'C', 'E', 'D', 'F']
```