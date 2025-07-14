# TP/FP/TN/FN
## terminology
- True Positives: fact=pos, pred=pos
- False Positives: fact=neg, pred=pos
- True Negatives: fact=neg, pred=neg
- False Neagtives: fact=pos, pred=neg

## metrics type
1. precision = TP/(TP+FP)  
correctness of predicted positive
2. recall = TP/(TP+FN)  
correctness of actual positive
3. f1-score = (2\*precision\*recall)/(precision+recall)  
balanced precision and recall
4. accuracy = (TP+TN)/TOTAL  
correctness of model prediction

## usage
1. precision: when false positive is undesirable
2. recall: when false negative is undesirable
3. f1-score: both precision and recall is important
4. accuracy: both pos and neg are important

# L1/L2/cosine similarity
## terminology
- L1 distance: Σ(|ai-bi|)
- L2 distance: straight-line distance between vectors in space (Σ(ai-bi)^2)^0.5
- cosine similarity: direction difference of two vectors, A.B/(|A|.|B|)

## Character
- L1 distance: 
    ```
    NO sensitive to outliers
    NO smooth gradients 
    YES work well for sparse data
    ```
- L2 distance
    ```
    YES sensitive to outliers
    YES smooth gradients
    NO work well for sparse data
    ```
- cosine similarity
    ```
    ```

## usage
- L1 distance: use when data is sparse, robust to outliers.
    ```
    KNN
    image generation (sharpness-preserving)
    anomaly detection
    ```
- L2 distance: use when care about absolute distances, including magnitude.
    ```
    image embedding
    code embedding
    model trained with Contrastive loss
    ```
- cosine similarity: use when care about the direction of the vectors more than the magnitude.
    ```
    semantic similarity (text)
    text embedding (BERT, GPT, etc)
    model trained with Cross-entropy / MLM / next-token prediction
    ```

# Intersection over Union (IoU)
## terminology
IoU: measures overlap between two regions
IoU = area of overlap / area of union

## Usage
Use it for comparsing similarity of two sets   
eg: object detection, SQL execution result  

# Kendall’s τ (tau)
## Terminology
- Kendall's tau: rank correlation between two lists  
- Concordant pair (cp): order of 2 elements is same in 2 lists  
- Disconcordant pair (dcp): order of 2 elements is different in 2 lists
- tau = (cp - dcp) / (pair number) 

```
tau = 1: prefect agreement
tau = 0: random order
tau = -1: perfect disagreement
```

## Usage
Use it for comparing two lists's ranking result  
eg, eval ranking algo  