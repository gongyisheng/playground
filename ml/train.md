# Model training
maco parameter tuning
## General
1. weight_decay: 0.01  
    ```
    add λ⋅∣∣w∣∣ to pentalize large weights, prevent overfitting  
    strongly not recommend to set to 0
    ```
2. gradient_accumulation_steps: int
   ```
   accumulate gradients over several smaller mini-batches and then perform one update
   use it to avoid CUDA out of memory issue, but at cost of training performance
   ```
3. lr_scheduler_type: str
    ```
    configure the learning rate scheduler during training, following is good

    linear: lr reduce to 0 linear change
    cosine: lr reduce to 0 cosine change
    cosine_with_min_lr: lr reduce to min_lr or min_lr_rate cosine change
    ```

## Finetune
1. epoch: 3-5 (small) / 2-4 (large) / 3-10 (with early stop) 
   ``` 
    3-10 cases: NER model, need to observe eval result  
    use early stop if your task aim is not clear

    too many epoch will cause overfitting
    ```
2. warmup: 5% to 10% of total steps  
    ```
    to prevent gradient unstable
    ```

## Unstable Traning
- Loss explodes: learning rate is too high
- Loss fluctuates wildly: too much gradient noise, overfitting, poor batch normalization
- Loss does not decrease: underfitting, learning rate is too low
- Training accuracy > validation accuracy: overfitting
- Validation accuracy increase: overfitting

## Noise in Dataset
It will cause unstable training, several ways to improve:
- remove very long / short examples
- remove duplicated / conflict records
- train on easy & high quality data then expand
- use higher dropout
- use label smoothing (CrossEntropyLoss(label_smoothing=0.1) for classification task)
