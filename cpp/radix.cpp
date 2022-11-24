#include "radix.h"
#include <iostream>

int main() {
    radix::RadixTree tree;
    tree.insert("hello");
    tree.insert("world");
    bool res = tree.isPartOf("sdahsdahsduaosjsnaichellsowodrldl");
    std::cout << res << std::endl;
    return 0;
}