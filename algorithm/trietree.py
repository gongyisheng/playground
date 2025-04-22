class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class TrieTree:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_end_of_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_end_of_word

    def starts_with(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current.children:
                return False
            current = current.children[letter]
        return True

if __name__ == "__main__":
    # Example usage
    trie = TrieTree()
    trie.insert("hello")
    trie.insert("helium")

    print(trie.search("hello"))     # True
    print(trie.search("helix"))     # False
    print(trie.starts_with("hel"))  # True
    print(trie.starts_with("hey"))  # False
