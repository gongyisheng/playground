#include <string>
#include <queue>

namespace radix{
    // Radix node is a kind of compressed Radix node
    // Radix node struct
    struct RadixNode {
        bool isEnd;
        std::string word;
        RadixNode *children[128];
    };
    // Radix tree class
    class RadixTree {
        private:
            RadixNode* root;
        public:
            RadixTree() {
                root = new RadixNode();
                root->isEnd = false;
                std::memset(root->children, NULL, 128);
            }
            void insert(std::string text) {
                RadixNode* curr = root;
                int i = 0;
                while(i<text.size()) {
                    int c = (int)text[i];
                    RadixNode **curr_children = curr->children;
                    // if the current node has no child with `c` as perfix
                    if (curr_children[c] == NULL) {
                        curr_children[c] = new RadixNode();
                        curr_children[c]->word = text.substr(i, text.size()-i);
                        curr_children[c]->isEnd = true;
                        break;
                    }

                    // if the current node has a child with `c` as perfix
                    std::string existing_word = curr_children[c]->word;
                    // compare & see if the perfix of existing word and the new word are the same
                    int j = 0;
                    while(j<existing_word.size() && i<text.size() && existing_word[j]==text[i]) {
                        j++;
                        i++;
                    }

                    // if reach the end of the existing word
                    if(j==existing_word.size()) {
                        curr = curr_children[c];
                    }
                    else { // if not reach the end of the existing word
                        // create a new node to be appended to the existing node
                        RadixNode *new_node = new RadixNode();
                        new_node->word = existing_word.substr(j, existing_word.size()-j);
                        new_node->isEnd = curr_children[c]->isEnd;
                        *new_node->children = *curr_children[c]->children;
                        // update the existing node
                        RadixNode *existing_node = curr_children[c];
                        existing_node->word = existing_word.substr(0, j);
                        existing_node->isEnd = false;
                        existing_node->children[(int)new_node->word[0]] = new_node;
                        // if the new word is not longer than the existing word
                        if(i==text.size()) {
                            curr_children[c]->isEnd = true;
                        } 
                        else { // if the new word is longer than the existing word
                            RadixNode *new_node2 = new RadixNode();
                            new_node2->word = text.substr(i);
                            new_node2->isEnd = true;
                            curr_children[c]->children[(int)new_node2->word[0]] = new_node2;
                        }
                        break;
                    }
                }
            }
            bool isPartOf(std::string text){
                std::queue<std::pair<RadixNode*, int> > q;
                RadixNode **root_children = root->children;
                for (int i=0;i<text.size();i++) {
                    char c = (int)text[i];
                    if(q.size()!=0){
                        int size = q.size();
                        for(int j=0;j<size;j++){
                            std::pair<RadixNode*, int> p = q.front();
                            RadixNode *node = p.first;
                            RadixNode **node_children = node->children;
                            int index = p.second;
                            q.pop();
                            if(node->word[index]==text[i]) {
                                // if cursor is at the end of the word
                                if(index==node->word.size()-1) {
                                    // reach the end
                                    if(node->isEnd) {
                                        return true;
                                    }
                                    // has next node
                                    else if(node_children[c] != NULL) {
                                        q.push(std::make_pair(node_children[c], 0));
                                    }
                                }
                                else {
                                    q.push(std::make_pair(node, index+1));
                                }
                            }
                        }
                    }
                    if(root_children[c] != NULL){
                        q.push(std::make_pair(root_children[c], 1));
                    }
                }
                return false;
            }
    };
}