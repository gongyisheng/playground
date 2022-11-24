#include <unordered_map>
#include <string>
#include <queue>

namespace trie{
    // Trie node struct
    struct TrieNode {
        bool isEnd;
        TrieNode *children[128];
    };
    // Trie tree class
    class TrieTree {
        private:
            TrieNode* root;
        public:
            TrieTree() {
                root = new TrieNode();
                root->isEnd = false;
                std::memset(root->children, NULL, 128);
            }
            void insert(std::string text) {
                TrieNode* curr = root;
                for (int i=0;i<text.size();i++){
                    int c = (int)text[i];
                    TrieNode **curr_children = curr->children;
                    if (curr_children[c] == NULL) {
                        curr_children[c] = new TrieNode();
                    }
                    curr = curr_children[c];
                }
                curr->isEnd = true;
            }
            bool search(std::string word) {
                TrieNode* curr = root;
                for (int i=0;i<word.size();i++) {
                    int c = (int)word[i];
                    if (curr->children[c] != NULL) {
                        return false;
                    }
                    curr = curr->children[c];
                }
                return curr->isEnd;
            }
            bool isPartOf(std::string text){
                std::queue<TrieNode*> q;
                TrieNode **root_children = root->children;
                for (int i=0;i<text.size();i++) {
                    char c = (int)text[i];
                    if(q.size()!=0){
                        int size = q.size();
                        for(int j=0;j<size;j++){
                            TrieNode *node = q.front();
                            TrieNode **node_children = node->children;
                            q.pop();
                            if(node_children[c] != NULL){
                                if(node_children[c]->isEnd){
                                    return true;
                                }
                                q.push(node_children[c]);
                            }
                        }
                    }
                    if(root_children[c] != NULL){
                        q.push(root_children[c]);
                    }
                }
                return false;
            }
    };
}