# Transformer Architecture

## Positional Encoding

### Overview
Positional encoding adds position information to token embeddings, enabling transformers to understand sequence order. Without it, transformers would treat "cat ate mouse" and "mouse ate cat" identically since self-attention is permutation-invariant.

### Why It's Needed
Self-attention computes `Attention = softmax(QK^T/√d)V` using only embedding values, not positions. The tensor order [token1, token2, token3] exists but isn't used in dot product calculations. For "cat ate mouse" vs "mouse ate cat", attention scores between any two tokens remain identical regardless of their positions.

### How It Works
Positional encodings are added element-wise to word embeddings:
- Without PE: `[embed_cat, embed_ate, embed_mouse]`
- With PE: `[embed_cat+PE(0), embed_ate+PE(1), embed_mouse+PE(2)]`

This makes each position unique. Now `Attention(cat→mouse)` differs between sentences because the input vectors include position information.

### Mathematical Formulation
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))      # even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))    # odd dimensions
```
Where: `pos` = position, `i` = dimension pair index, `d` = embedding dimension

**Example (d=4, pos=2):**
- Dim 0: sin(2/1) ≈ 0.909
- Dim 1: cos(2/1) ≈ -0.416  
- Dim 2: sin(2/100) ≈ 0.020
- Dim 3: cos(2/100) ≈ 0.999

### Why Sin/Cos for Even/Odd Dimensions
The alternating sin/cos pattern provides critical benefits:

1. **Phase relationships**: Sin and cos are 90° out of phase, creating orthogonal representations that maximize differentiation between positions
2. **Unique signatures**: Each position gets a unique combination of sin/cos values across dimension pairs
3. **Relative position learning**: Trigonometric identities like `sin(a+b) = sin(a)cos(b) + cos(a)sin(b)` enable learning position offsets through linear combinations
4. **Smooth interpolation**: Continuous functions that change gradually, helping generalize to unseen sequence lengths
5. **Bounded values**: Both stay in [-1, 1], maintaining stable gradient flow

The sin/cos pair acts like a 2D coordinate system for each frequency level, providing richer position representation than a single function could.

### Why 10000 as Base
The base 10000 is chosen for practical reasons:

1. **Wide frequency range**: Wavelengths range from 2π to 10000·2π ≈ 62,832, covering local to global position relationships
2. **Typical sequence lengths**: Most transformers handle 100-4000 tokens; 10000 ensures meaningful variation without repetition
3. **Exponential spacing**: Creates useful frequency distribution (e.g., for d=512: wavelengths from ~6 to ~63k tokens)
4. **Empirical optimization**: Found to work well in practice - smaller bases cause unwanted repetition, larger bases change too slowly
5. **Numerical stability**: Large enough for variation, small enough to avoid float32 precision issues

The choice balances coverage of linguistic scales (words, phrases, sentences, paragraphs) within typical context windows.

### Key Properties
1. **Unique patterns**: Each position gets a distinct encoding through varying frequencies (10000^(2i/d))
2. **No exact duplicates**: Different frequencies per dimension pair prevent repetition in typical sequences
3. **Learnable relative positions**: Sinusoidal properties help models understand position relationships
4. **No parameters**: Unlike learned embeddings, requires no training

### Frequency Characteristics
- Lower dimensions: Fast oscillation (period ~6 positions)
- Higher dimensions: Slow oscillation (period ~60,000 positions)
- Creates hierarchical position representation across dimensions