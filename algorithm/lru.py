from collections import OrderedDict

class LRUCacheV1:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move the accessed item to the end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing key and move it to end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove the first (least recently used) item
            self.cache.popitem(last=False)
        self.cache[key] = value

def test_v1():
    lru = LRUCacheV1(2)
    lru.put(1, 1)
    lru.put(2, 2)
    assert lru.get(1) == 1
    lru.put(3, 3)      # Evicts key 2
    assert lru.get(2) == -1
    lru.put(4, 4)      # Evicts key 1
    assert lru.get(1) == -1
    assert lru.get(3) == 3
    assert lru.get(4) == 4

class Node:
    """Doubly linked list node."""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCacheV2:
    """LRU Cache implemented with hash map + doubly linked list."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Sentinel nodes to avoid edge case checks
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the doubly linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """Add a node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # Move to front (mark as recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Create new node
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)

            # Evict LRU if over capacity
            if len(self.cache) > self.capacity:
                lru_node = self.tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

def test_v2():
    lru2 = LRUCacheV2(2)
    lru2.put(1, 1)
    lru2.put(2, 2)
    assert lru2.get(1) == 1
    lru2.put(3, 3)      # Evicts key 2
    assert lru2.get(2) == -1
    lru2.put(4, 4)      # Evicts key 1
    assert lru2.get(1) == -1
    assert lru2.get(3) == 3
    assert lru2.get(4) == 4


if __name__ == "__main__":
    print("Testing LRUCache (OrderedDict):")
    test_v1()

    print("Testing LRUCacheV2 (HashMap + DoublyLinkedList):")
    test_v2()
