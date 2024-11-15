# install on ubuntu
## install dev files
`sudo apt install postgresql-server-dev-17`

## install pgvector
```
cd /tmp
git clone --branch v0.7.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install # may need sudo
```

## usage
```
# enable the extension
CREATE EXTENSION vector;

# create a table
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

# insert vectors
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

# update vectors
UPDATE items SET embedding = '[1,2,3]' WHERE id = 1;

# delete vectors
DELETE FROM items WHERE id = 1;

# query vectors
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

<-> - L2 distance
<#> - (negative) inner product
<=> - cosine distance
<+> - L1 distance (added in 0.7.0)
<~> - Hamming distance (binary vectors, added in 0.7.0)
<%> - Jaccard distance (binary vectors, added in 0.7.0)

SELECT embedding <-> '[3,1,2]' AS distance FROM items;
SELECT (embedding <#> '[3,1,2]') * -1 AS inner_product FROM items;
SELECT 1 - (embedding <=> '[3,1,2]') AS cosine_similarity FROM items;

# index
supported index: HNSW, IVFFlat
```
