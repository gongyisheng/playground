#include "trie.h"
#include <iostream>

int main() {
    trie::TrieTree tree;
    tree.insert("hello");
    tree.insert("world");
    bool res = tree.isPartOf("sdahsdahsduaosjsnaichellowordl");
    std::cout << res << std::endl;
    return 0;
}