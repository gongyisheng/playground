# python container operations
1. stack
```
stack = []
stack.append(1)
stack.append(2)
stack.append(3)

# get the first one
stack.pop() 

# view the top element
stack[-1]
```

2. deque
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

3. priority queue
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

4. ordered dict
```
from collections import OrderedDict

d = OrderedDict()

d["key1"] = "value1"
d["key2"] = "value2"
d["key3"] = "value3"

print(d) # OrderedDict({'key1': 'value1', 'key2': 'value2', 'key3', 'value3'})

# move to end
d.move_to_end("key1")
print(d) # OrderedDict({'key2': 'value2', 'key3': 'value3', 'key1': 'value1'})

# pop head/tail item
pair = d.popitem(last=False)
print(d) # OrderedDict({'key2': 'value2', 'key3': 'value3'})
print(p) # ('key1', 'value1')

# pop a specific item
pair = d.pop("key2")
```

5. customized sort
```
# Sample list of tuples
data = [(4, 2), (1, 3), (3, 2), (2, 1), (5, 3)]

# Sort by second item, then by first item if second is equal
sorted_data = sorted(data, key=lambda x: (x[1], x[0]))

# Output the result
print(sorted_data)
```