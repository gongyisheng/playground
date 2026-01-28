class RadixNode:
    def __init__(self):
        # Children maps: first_char -> (edge_label, child_node)
        # We use first char as key for O(1) lookup of which edge to follow
        self.children = {}
        self.is_end_of_word = False


def common_prefix_length(s1, s2):
    """Find the length of common prefix between two strings."""
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i


class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, word):
        """
        Insert a word into the radix tree.

        The key insight: when we find a partial match with an existing edge,
        we need to SPLIT that edge at the divergence point.
        """
        if not word:
            self.root.is_end_of_word = True
            return

        current = self.root
        i = 0  # Position in word we're inserting

        while i < len(word):
            first_char = word[i]

            # Case 1: No edge starts with this character - create new edge
            if first_char not in current.children:
                new_node = RadixNode()
                new_node.is_end_of_word = True
                current.children[first_char] = (word[i:], new_node)
                return

            # Case 2: Found an edge starting with this character
            edge_label, child_node = current.children[first_char]
            remaining_word = word[i:]

            # Find how much of edge_label matches remaining_word
            prefix_len = common_prefix_length(edge_label, remaining_word)

            # Case 2a: Edge label fully matches, continue to child
            if prefix_len == len(edge_label):
                if prefix_len == len(remaining_word):
                    # Exact match - mark this node as end of word
                    child_node.is_end_of_word = True
                    return
                else:
                    # Move to child and continue with rest of word
                    current = child_node
                    i += prefix_len

            # Case 2b: Partial match - need to SPLIT the edge
            else:
                # Create a new intermediate node at the split point
                split_node = RadixNode()

                # The common prefix leads to split_node
                common_prefix = edge_label[:prefix_len]
                current.children[first_char] = (common_prefix, split_node)

                # Old edge remainder goes from split_node to original child
                old_suffix = edge_label[prefix_len:]
                split_node.children[old_suffix[0]] = (old_suffix, child_node)

                # New word remainder (if any) goes from split_node to new node
                new_suffix = remaining_word[prefix_len:]
                if new_suffix:
                    new_node = RadixNode()
                    new_node.is_end_of_word = True
                    split_node.children[new_suffix[0]] = (new_suffix, new_node)
                else:
                    # Word ends exactly at split point
                    split_node.is_end_of_word = True
                return

    def search(self, word):
        """
        Search for an exact word in the radix tree.

        We traverse edges, matching the edge labels against remaining word.
        """
        if not word:
            return self.root.is_end_of_word

        current = self.root
        i = 0

        while i < len(word):
            first_char = word[i]

            # No edge with this starting character
            if first_char not in current.children:
                return False

            edge_label, child_node = current.children[first_char]
            remaining_word = word[i:]

            # Check if edge_label is a prefix of remaining_word
            if not remaining_word.startswith(edge_label):
                return False

            # Move along
            i += len(edge_label)
            current = child_node

        return current.is_end_of_word

    def starts_with(self, prefix):
        """
        Check if any word in the tree starts with the given prefix.

        Similar to search, but we also succeed if prefix ends mid-edge.
        """
        if not prefix:
            return True

        current = self.root
        i = 0

        while i < len(prefix):
            first_char = prefix[i]

            if first_char not in current.children:
                return False

            edge_label, child_node = current.children[first_char]
            remaining_prefix = prefix[i:]

            # If remaining prefix is shorter than edge, check if edge starts with it
            if len(remaining_prefix) <= len(edge_label):
                return edge_label.startswith(remaining_prefix)

            # If edge doesn't match the beginning of remaining prefix, fail
            if not remaining_prefix.startswith(edge_label):
                return False

            i += len(edge_label)
            current = child_node

        return True

    def delete(self, word):
        """
        Delete a word from the radix tree.

        After deletion, we may need to merge nodes (re-compress).
        Returns True if word was found and deleted.
        """
        if not word:
            if self.root.is_end_of_word:
                self.root.is_end_of_word = False
                return True
            return False

        # We need to track the path for potential merging
        # Stack: (parent_node, first_char_of_edge, edge_label, child_node)
        path = []
        current = self.root
        i = 0

        while i < len(word):
            first_char = word[i]

            if first_char not in current.children:
                return False

            edge_label, child_node = current.children[first_char]
            remaining_word = word[i:]

            if not remaining_word.startswith(edge_label):
                return False

            path.append((current, first_char, edge_label, child_node))
            i += len(edge_label)
            current = child_node

        # Word not in tree (path exists but not marked as word)
        if not current.is_end_of_word:
            return False

        # Unmark as end of word
        current.is_end_of_word = False

        # Now we need to clean up / re-compress
        # Work backwards through the path
        while path:
            parent, first_char, edge_label, node = path.pop()

            # If node has no children and is not end of word, remove it
            if not node.children and not node.is_end_of_word:
                del parent.children[first_char]

            # If node has exactly one child and is not end of word, merge
            elif len(node.children) == 1 and not node.is_end_of_word:
                # Get the single child
                child_first_char = next(iter(node.children))
                child_edge, grandchild = node.children[child_first_char]

                # Merge: combine edge_label + child_edge
                merged_label = edge_label + child_edge
                parent.children[first_char] = (merged_label, grandchild)

            else:
                # No more compression possible up the tree
                break

        return True

    def get_all_words(self):
        """Helper to get all words in the tree (for debugging)."""
        words = []
        self._collect_words(self.root, "", words)
        return words

    def _collect_words(self, node, prefix, words):
        if node.is_end_of_word:
            words.append(prefix)
        for first_char, (edge_label, child) in node.children.items():
            self._collect_words(child, prefix + edge_label, words)


if __name__ == "__main__":
    # Example usage - same as your trie example
    tree = RadixTree()
    tree.insert("hello")
    tree.insert("helium")

    print(tree.search("hello"))      # True
    print(tree.search("helix"))      # False
    print(tree.starts_with("hel"))   # True
    print(tree.starts_with("hey"))   # False

    print("\n--- More tests ---")

    # Test edge splitting
    tree.insert("test")
    tree.insert("team")
    tree.insert("tea")

    print(tree.search("test"))       # True
    print(tree.search("team"))       # True
    print(tree.search("tea"))        # True
    print(tree.search("te"))         # False (not inserted)

    print("\nAll words:", tree.get_all_words())

    # Test deletion
    print("\n--- Delete tests ---")
    tree.delete("tea")
    print(tree.search("tea"))        # False
    print(tree.search("team"))       # True (still exists)
    print("\nAfter deleting 'tea':", tree.get_all_words())
