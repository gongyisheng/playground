class Node:
    """Doubly linked list node storing key, value, and frequency."""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.freq = 1
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """A doubly linked list with sentinel nodes for O(1) operations."""
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def add_to_front(self, node: Node) -> None:
        """Add node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def remove(self, node: Node) -> None:
        """Remove a node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def remove_last(self) -> Node:
        """Remove and return the last node (least recently used)."""
        if self.size == 0:
            return None
        node = self.tail.prev
        self.remove(node)
        return node

    def is_empty(self) -> bool:
        return self.size == 0


class LFUCache:
    """
    LFU Cache implemented with:
    - key_map: HashMap (key -> Node) for O(1) lookup
    - freq_map: HashMap (freq -> DoublyLinkedList) for O(1) frequency bucket access
    - min_freq: integer tracking minimum frequency for O(1) eviction
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_map = {}   # key -> Node
        self.freq_map = {}  # freq -> DoublyLinkedList
        self.min_freq = 0

    def _update_freq(self, node: Node) -> None:
        """Remove node from current freq list and add to freq+1 list."""
        freq = node.freq

        # Remove from current frequency list
        self.freq_map[freq].remove(node)

        # If current freq list is empty and it was min_freq, increment min_freq
        if self.freq_map[freq].is_empty() and freq == self.min_freq:
            self.min_freq += 1

        # Increment node's frequency
        node.freq += 1

        # Add to new frequency list
        if node.freq not in self.freq_map:
            self.freq_map[node.freq] = DoublyLinkedList()
        self.freq_map[node.freq].add_to_front(node)

    def get(self, key: int) -> int:
        if key not in self.key_map:
            return -1

        node = self.key_map[key]
        self._update_freq(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_map:
            # Update existing key
            node = self.key_map[key]
            node.value = value
            self._update_freq(node)
        else:
            # Evict if at capacity
            if len(self.key_map) >= self.capacity:
                # Remove LRU node from min_freq list
                lru_node = self.freq_map[self.min_freq].remove_last()
                del self.key_map[lru_node.key]

            # Create new node with freq=1
            node = Node(key, value)
            self.key_map[key] = node

            # Add to freq=1 list
            if 1 not in self.freq_map:
                self.freq_map[1] = DoublyLinkedList()
            self.freq_map[1].add_to_front(node)

            # New node always has min_freq=1
            self.min_freq = 1


def test_lfu():
    # Test case 1: Basic operations
    lfu = LFUCache(2)
    lfu.put(1, 1)   # freq: {1: [1]}
    lfu.put(2, 2)   # freq: {1: [2, 1]}
    assert lfu.get(1) == 1  # freq: {1: [2], 2: [1]}
    lfu.put(3, 3)   # Evicts key 2 (LFU), freq: {1: [3], 2: [1]}
    assert lfu.get(2) == -1  # key 2 was evicted
    assert lfu.get(3) == 3   # freq: {2: [3, 1]}
    lfu.put(4, 4)   # Evicts key 1 (LFU with freq=2, but 3 was accessed more recently)
                    # Actually evicts key 1 because both have freq=2, but 1 is LRU
    assert lfu.get(1) == -1
    assert lfu.get(3) == 3
    assert lfu.get(4) == 4

    # Test case 2: LeetCode example
    lfu2 = LFUCache(2)
    lfu2.put(1, 1)
    lfu2.put(2, 2)
    assert lfu2.get(1) == 1
    lfu2.put(3, 3)  # Evicts 2
    assert lfu2.get(2) == -1
    assert lfu2.get(3) == 3
    lfu2.put(4, 4)  # Evicts 1
    assert lfu2.get(1) == -1
    assert lfu2.get(3) == 3
    assert lfu2.get(4) == 4

    # Test case 3: Capacity 1
    lfu3 = LFUCache(1)
    lfu3.put(1, 1)
    assert lfu3.get(1) == 1
    lfu3.put(2, 2)  # Evicts 1
    assert lfu3.get(1) == -1
    assert lfu3.get(2) == 2

    # Test case 4: Update existing key
    lfu4 = LFUCache(2)
    lfu4.put(1, 1)
    lfu4.put(2, 2)
    lfu4.put(1, 10)  # Update key 1, also increases its freq
    lfu4.put(3, 3)   # Should evict key 2 (lower freq)
    assert lfu4.get(2) == -1
    assert lfu4.get(1) == 10


if __name__ == "__main__":
    test_lfu()
    print("All LFU tests passed!")
