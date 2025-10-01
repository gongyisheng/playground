# Attention

## Core Components

The attention mechanism uses three learned linear projections:

- **Query (Q)**: What information to search for at each position
- **Key (K)**: What information each position contains  
- **Value (V)**: The actual content to be aggregated

## Attention Computation

```
Attention(Q, K, V) = softmax(QK^T/√d_k)·V
```

Where:
- `QK^T`: Dot product between queries and keys (attention scores)
- `√d_k`: Scaling factor to prevent gradient vanishing
- `softmax`: Normalizes scores to attention weights
- Final matrix multiplication with V aggregates information

## KV Caching
Optimization technique for efficient inference:

1. **Initial token**: Compute Q₁, K₁, V₁
2. **Subsequent tokens**: 
   - Compute only Qₙ, Kₙ, Vₙ for new token n
   - Reuse cached K₁...Kₙ₋₁ and V₁...Vₙ₋₁ from previous steps
   - Attention: Qₙ attends to all K₁...Kₙ and aggregates V₁...Vₙ  

Computational Complexity:  
- Without caching: O(n²) per sequence
- With KV caching: O(n) per new token
- Memory trade-off: Store n×d×2×num_layers values

## Multi-Head Attention

Parallel attention with different learned projections:
- Split d_model into h heads of dimension d_k = d_model/h
- Each head learns different attention patterns
- Concatenate and project outputs back to d_model

## Limitation
- Self-attention computes `Attention = softmax(QK^T/√d)V` using only embedding values, not positions. Need to add positional encoding to include position information