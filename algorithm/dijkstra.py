import heapq

# time complexity: O((V + E) * log(V))
# space complexity: O(V)
# V is the number of nodes and E is the number of edges

def dijkstra(graph, start):
    # Priority queue to store (dist, node) tuples
    pq = []
    heapq.heappush(pq, (0, start))
    
    # Dictionary to store the shortest dist to each node
    dists = {node: float('inf') for node in graph}
    dists[start] = 0
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        
        # If the popped node has a greater dist than the recorded one, skip it
        if curr_dist > dists[curr_node]:
            continue
        
        # Explore neighbors
        for neighbor, weight in graph[curr_node].items():
            dist = curr_dist + weight
            
            # If found a shorter path, update it
            if dist < dists[neighbor]:
                dists[neighbor] = dist
                heapq.heappush(pq, (dist, neighbor))
    
    return dists

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start_node = 'A'
shortest_paths = dijkstra(graph, start_node)
print(shortest_paths)
