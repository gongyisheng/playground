# Residual Connections

What is a Residual Connection?

A skip connection that adds input directly to output:

output = F(x) + x

The network learns the residual F(x) rather than the full mapping.

Why Residual Connections?

Problem: In deep networks, gradients are multiplied across layers.

∂Loss/∂x = ∏(i=1 to L) Jᵢ

If each Jᵢ ≈ 0.9: 0.9^96 ≈ 10⁻⁵  → vanishing
If each Jᵢ ≈ 1.1: 1.1^96 ≈ 10⁴   → exploding

Solution: Residual connections provide an identity path.

∂(x + F(x))/∂x = I + ∂F/∂x
                ↑
            Identity term guarantees gradient flow

Transformer Architecture

Each Transformer block has two residual connections:

x = x + Attention(LayerNorm(x))  # Token interaction
x = x + FFN(LayerNorm(x))        # Per-token transformation

Why two?
- Functional separation: Attention (cross-token) and FFN (within-token) serve different purposes
- Operator splitting: Two small steps are more stable than one large step
- Independent optimization: Gradients don't interfere with each other

Pre-Norm vs Post-Norm

# Pre-Norm (GPT, LLaMA) - preferred for deep models
x = x + Sublayer(LayerNorm(x))

# Post-Norm (original Transformer)
x = LayerNorm(x + Sublayer(x))

| Aspect          | Pre-Norm       | Post-Norm            |
|-----------------|----------------|----------------------|
| Gradient path   | Clean identity | Blocked by LayerNorm |
| Deep training   | Stable         | Difficult            |
| Warmup required | Optional       | Necessary            |

LayerNorm's Role

1. Normalizes activations to zero mean, unit variance
2. Stabilizes gradients by bounding Jacobian eigenvalues
3. Decouples layers for independent learning

Key Takeaway

Residual connections create a "gradient highway" that allows gradients to flow directly to early layers, enabling training of very deep networks (e.g., 96-layer GPT-3).