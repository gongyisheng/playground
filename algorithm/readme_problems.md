# problem-solving strategies
## Tree
[LC144](https://leetcode.com/problems/binary-tree-preorder-traversal/) basic preorder  
[LC94](https://leetcode.com/problems/binary-tree-inorder-traversal/) basic inorder  
[LC145](https://leetcode.com/problems/binary-tree-postorder-traversal/) basic postorder  
[LC589](https://leetcode.com/problems/n-ary-tree-preorder-traversal) basic preorder, n-ray  
[LC590](https://leetcode.com/problems/n-ary-tree-postorder-traversal/) basic postorder, n-ray  
[LC102](https://leetcode.com/problems/binary-tree-level-order-traversal) basic level order, top down    
[LC107](https://leetcode.com/problems/binary-tree-level-order-traversal-ii) basic level order, button up  
[LC429](https://leetcode.com/problems/n-ary-tree-level-order-traversal/) basic level order, n-ray   
[LC1161](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree) basic level order, record sum and find max
[LC1302](https://leetcode.com/problems/deepest-leaves-sum) traverse + status collect, find max  
[LC872](https://leetcode.com/problems/leaf-similar-trees/) traverse + status collect, compare result  
[LC987](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree) traverse + status collect, sort result  
[LC98](https://leetcode.com/problems/validate-binary-search-tree) inorder dfs + status collect, check trend  
[LC100](https://leetcode.com/problems/same-tree/) traverse + status passing  
[LC101](https://leetcode.com/problems/symmetric-tree) traverse + status passing, almost same as LC100  
[LC572](https://leetcode.com/problems/subtree-of-another-tree) traverse + status passing, similar to LC100  
[LC965](https://leetcode.com/problems/univalued-binary-tree) traverse + status passing, compare static value  
[LC104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) traverse + status passing, max depth  
[LC110](https://leetcode.com/problems/balanced-binary-tree) traverse + status passing, depth + judgement logic  
[LC111](https://leetcode.com/problems/minimum-depth-of-binary-tree) traverse + status passing, depth + judgement logic  
[LC235](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree) traverse + status passing, return lca, use the property of BST  
[LC236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree) traverse + status passing, return lca, compare it with LC235, trick is finding the way to pass status  
[LC1123](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves) traverse + status passing, return ans_node and depth at the same time, compare with LC235 and LC236  
[LC814](https://leetcode.com/problems/binary-tree-pruning) traverse + status passing, trim, change left/right node based on return   
[LC669](https://leetcode.com/problems/trim-a-binary-search-tree) traverse + status passing, trim, change left/right node based on high/low and bst characteristic   
[LC1325](https://leetcode.com/problems/delete-leaves-with-a-given-value) traverse + status passing, trim, change value based on return  
[LC112](https://leetcode.com/problems/path-sum/) traverse + status passing, path sum, keep intermediate status  
[LC113](https://leetcode.com/problems/path-sum-ii) preorder dfs + status collect  
[LC331](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/) serialization, verifiy by count node num and null num  

## BST
[LC98](https://leetcode.com/problems/validate-binary-search-tree) inorder + check trend  
[LC230](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) inorder + path store  
[LC501](https://leetcode.com/problems/find-mode-in-binary-search-tree) inorder + freq map store  
[LC450](https://leetcode.com/problems/delete-node-in-a-bst) recursive search, remove node by condition (need redo)  

## DFS/BFS
[LC55](https://leetcode.com/problems/jump-game) dfs/bfs --> state compress, greedy  
[LC79](https://leetcode.com/problems/word-search) dfs, grid search, tricks: a. do all validation before exploring b. change path value to # to mark it's visited  
[LC117](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii) bfs, level order  
[LC130](https://leetcode.com/problems/surrounded-regions) dfs, avoid methods that may cause wrong updates (should update only if all return true, but make updates before get result and cannot rollback).  
[LC1863](https://leetcode.com/problems/sum-of-all-subset-xor-totals) dfs to traverse all subset, add xor sum  
[LC2039](https://leetcode.com/problems/the-time-when-the-network-becomes-idle/) bfs, tricks: a. just do bfs one time from master to get the dist to every node. b. build model of calculating idle time, additional msg cnt * patience + dst + 1. c. be careful to calculate additional msg cnt.  
[LC967](https://leetcode.com/problems/numbers-with-same-consecutive-differences/) dfs, path choose based on rule, build number  
[LC841](https://leetcode.com/problems/keys-and-rooms) dfs or bfs, search to get final result  
[LC2257](https://leetcode.com/problems/count-unguarded-cells-in-the-grid) dfs, grid search, complicated processing rule  
[LC39](https://leetcode.com/problems/combination-sum/) dfs/dp, dfs use sort+ascend idx to avoid dup, dp use list of comb as state  
[LC93](https://leetcode.com/problems/restore-ip-addresses) dfs, validate substring  

## Binary Search

## Sort
[LC3169](https://leetcode.com/problems/count-days-without-meetings) sort + state compression, count segment  
[LC3394](https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections) sort + count segment * 2 times  
[LC2003](https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid) sort + medium, why medium? think about a group of dots on a line and you need to choose one point that sum of distance of all dots to that point is smallest. The point should be average (which is not possible in this problem) or medium  

## Heap
[LC215](https://leetcode.com/problems/kth-largest-element-in-an-array) mini heap, push then pop  
[LC1054](https://leetcode.com/problems/distant-barcodes) heap, pick most frequent first  

## Two Pointer
[LC88](https://leetcode.com/problems/merge-sorted-array) two pointer, reverse order  
[LC1678](https://leetcode.com/problems/goal-parser-interpretation) two pointer, same dir, count brackets, with status processing  
[LC1679](https://leetcode.com/problems/max-number-of-k-sum-pairs) two pointer, diff dir, count sum = target  

## Linkedlist
[LC3217](https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array) rebuild linkedlist

## Sliding Window
[LC2379](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks) dfs? no, use sliding window  

## Divide and Conquer
[LC23](https://leetcode.com/problems/merge-k-sorted-lists) merge and sort, be careful is len(lists) == 0  

## Graph
[LC200](https://leetcode.com/problems/number-of-islands) dfs, grid search, connected components  
[LC547](https://leetcode.com/problems/number-of-provinces) dfs, connected components  
[LC1976](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination) dfs/bfs? no, use dijkstra(find shortest route) + dp (count ways)  
[LC839](https://leetcode.com/problems/similar-string-groups) string comparison + union-find  
[LC3108](https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph) bit operations + union-find (note: -1&anything = anything, the more number you have, the less & sum you can get)  

## Math
[LC1904](https://leetcode.com/problems/the-number-of-full-rounds-you-have-played) time interval with upper/lower bound calculation  

## Greedy
[LC1578](https://leetcode.com/problems/minimum-time-to-make-rope-colorful) greedy, pattern: meet consecutive color pick the smaller one to remove  

## Dynamic Programing
[LC55](https://leetcode.com/problems/jump-game/) dp, keep max steps can go  
[LC45](https://leetcode.com/problems/jump-game-ii/) dp, keep an array of min steps to reach  
[LC2140](https://leetcode.com/problems/solving-questions-with-brainpower) reverse order dp, ascending order does not work  


## Uncategorized
[LC82](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii) while+while   
[LC201](https://leetcode.com/problems/bitwise-and-of-numbers-range) observation, common suffix + bit observation  
[LC2873](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i) 3 loop is fine, but can be done with one loop by recording max value, max diff, and max result  
[LC310](https://leetcode.com/problems/minimum-height-trees) mht, cut leaves. 2 ways to get medium of a list: by index or cut leafs recursively  
