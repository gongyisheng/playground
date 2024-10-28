# install

# IndexFlatL2 vs IndexIVFFlat
`IndexFlatl2`: Exact search index, use L2 distance, O(N) per query, need to traverse all stored vectors  
`IndexIVFFlat`: An approximate search index, use L2 distance, clusters the data into a predefined number of partitions, O(K*log(N)) per query(k=num of cluster), need training

# benchmark result
```
search: 

```
# refs
```
https://medium.com/pythons-gurus/faiss-vector-database-for-production-llm-applications-90273c78fcbf
https://github.com/facebookresearch/faiss/issues/705
```