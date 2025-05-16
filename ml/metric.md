# Metrics
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
