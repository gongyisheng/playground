#include <unordered_map>
#include <string>
#include <queue>

namespace trie{
    // Trie node struct
    struct TrieNode {
        bool isEnd;
        std::unordered_map<char, TrieNode*> children;
    };
    // Trie tree class
    class TrieTree {
        private:
            TrieNode* root;
        public:
            TrieTree() {
                root = new TrieNode();
            }
            void insert(std::string word){
                TrieNode* node = root;
                for (int i=0;i<word.size();i++) {
                    if (node->children.find(word[i]) == node->children.end()) {
                        node->children[word[i]] = new TrieNode();
                    }
                    node = node->children[word[i]];
                }
                node->isEnd = true;
            }
            bool search(std::string word) {
                TrieNode* node = root;
                for (int i=0;i<word.size();i++) {
                    if (node->children.find(word[i]) == node->children.end()) {
                        return false;
                    }
                    node = node->children[word[i]];
                }
                return node->isEnd;
            }
            bool isPartOf(std::string word) {
                std::queue<TrieNode*> q;
                for (int i=0;i<word.size();i++) {
                    if(q.size()!=0){
                        int size = q.size();
                        for(int j=0;j<size;j++){
                            TrieNode* node = q.front();
                            q.pop();
                            if(node->children.find(word[i])!=node->children.end()){
                                if(node->children[word[i]]->isEnd){
                                    return true;
                                }
                                q.push(node->children[word[i]]);
                            }
                        }
                    }
                    if(root->children.find(word[i])!=root->children.end()){
                        q.push(root->children[word[i]]);
                    }
                }
                return false;
            }
    };
}