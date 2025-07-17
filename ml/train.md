# Model training
maco parameter tuning
## General
1. weight_decay: 0.01  
    ```
    add λ⋅∣∣w∣∣ to pentalize large weights, prevent overfitting  
    strongly not recommend to set to 0
    ```

## Finetune
1. epoch: 3-5 (small) / 2-4 (large) / 3-10 (with early stop) 
   ``` 
    3-10 cases: NER model, need to observe eval result  
    use early stop if your task aim is not clear
    ```
2. warmup: 5% to 10% of total steps  
    ```
    to prevent gradient unstable
    ```