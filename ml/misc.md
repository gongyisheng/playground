# misc knowledge

## Xavier Initialization
Xavier initialization (also called Glorot initialization, after Xavier Glorot who proposed it in 2010) is a weight initialization method designed to keep the variance of activations and gradients roughly the same across layers.

Xavier Uniform: Weights sampled from uniform distribution                                                                                   
- W ~ Uniform(-limit, +limit)
- limit = sqrt(6 / (fan_in + fan_out))  

Xavier Normal: Weights sampled from normal distribution (best for sigmoid, tanh)  
- W ~ Normal(0, std)                                                                                                            
- std = sqrt(2 / (fan_in + fan_out))

math behind:

To keep Var(y) ≈ Var(x), we need:  
- Var(W) = 1 / n_in       (for forward pass)  
- Var(W) = 1 / n_out      (for backward pass)  

Xavier balances both:  
- Var(W) = 2 / (n_in + n_out)

## LoRA

In standard LoRA initialization:                                                                                              
- lora_A is initialized with Kaiming/normal init                                                                              
- lora_B is initialized to zeros 

in megatron-bridge:
```
obj.linear_in = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)   # lora_A
obj.linear_out = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device) # lora_B

if lora_A_init_method == "xavier":
    torch.nn.init.xavier_uniform_(obj.linear_in.weight.data)   # lora_A: xavier init
else:
    nn.init.kaiming_uniform_(obj.linear_in.weight.data, a=math.sqrt(5))

obj.linear_out.weight.data.fill_(0)  # lora_B: initialized to ZEROS!
```

So it's expected to see in the first training step, lora_A is not updated, lora_B is updated.  
lora_A gets updated since step 2.  

