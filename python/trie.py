from functools import reduce
from collections import defaultdict

class TrieTree:
    def __init__(self):
        T = lambda : defaultdict(T)
        self.trie = T()
    
    def insert(self, word):
        reduce(dict.__getitem__, word, self.trie)['isEnd'] = True

    def search(self, word):
        curr = self.trie
        for w in word:
            if w in curr:
                curr = curr[w]
            else:
                return False
        return curr['isEnd']

if __name__ == '__main__':
    trie = TrieTree()
    trie.insert('apple')
    print(trie.search('apple'))
    print(trie.search('app'))
    trie.insert('app')
    print(trie.search('app'))