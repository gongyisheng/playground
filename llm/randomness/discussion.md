# Teaching LLMs to Learn Randomness - Project Discussion

## Project Goal
Research LLM capabilities in learning randomness using SFT with distribution as target.

## Initial Approach Ideas

### Proposed Methods
1. Pick a random number from list
2. Pick a random word from list
3. Pick a random number from range
4. More complicated cases (dice, etc.) - potentially using LLM to generate tasks

**Concerns:**
- Methods 1-3 may be too narrow and lead to overfitting
- Method 4 with LLM-generated tasks seems more promising

## Analysis of Proposed Methods

**Current Approach Concerns:**
- Methods 1-3 are indeed quite narrow and may lead to overfitting to specific formats
- The model might learn surface patterns rather than true randomness
- SFT with distribution as target is interesting, but we need to think about how to represent this

**Key Question:** What does "learning randomness" mean for an LLM?
- True randomness generation (uniform, normal, or other distributions)?
- Understanding when to be random vs. deterministic?
- Calibrated uncertainty in responses?

## Better Approaches to Consider

### 1. Hierarchical Task Complexity
Instead of just simple tasks, create a curriculum:
- **Level 1:** Basic random selection (methods 1-3)
- **Level 2:** Constrained randomness (dice rolls, card draws, weighted sampling)
- **Level 3:** Contextual randomness (when should output be random vs deterministic)
- **Level 4:** Distribution-aware tasks (sample from specific distributions)

### 2. Distribution Matching Training
Rather than SFT on individual examples:
- Generate many samples for the same prompt
- Train to match target distribution statistics (mean, variance, entropy)
- Use KL divergence or other distribution metrics as loss
- This could be done with DPO or custom loss functions

### 3. Prompt Diversity Generation
Use LLM to generate diverse randomness tasks:
```
- "Generate a random password with constraints..."
- "Shuffle this list..."
- "Sample 5 items without replacement..."
- "Flip a fair coin 10 times..."
- "Roll 3d6 and sum them..."
```

### 4. Seed-Based Approach
Include random seeds in prompts so model learns to:
- Map seeds to deterministic but pseudo-random outputs
- Understand the concept of reproducible randomness

### 5. Hybrid Approach (Recommended)

## Decided Approach

**Primary Goal:** Research LLM capabilities

**Training Method:** SFT with diverse LLM-generated tasks

**Evaluation Metrics:**
- Statistical tests (Chi-square, KS tests)
- Entropy metrics
- Task-specific accuracy

## Implementation Strategy

### Phase 1: Task Generation Pipeline
```
LLM-based task generator → Diversity filtering → Distribution annotation
```

**Task Categories to Generate:**
1. **Basic sampling** (uniform, weighted)
2. **Combinatorial** (permutations, combinations, shuffling)
3. **Game mechanics** (dice, cards, spinners)
4. **Statistical distributions** (normal, exponential, etc.)
5. **Contextual randomness** (password generation, UUID, lottery)

**Key insight:** For each task, generate:
- The instruction/prompt
- The target distribution (not just one sample!)
- Multiple valid outputs with their probabilities

### Phase 2: Training Data Format

Instead of: `(prompt, single_answer)`, use:
```json
{
  "prompt": "Roll a standard six-sided die",
  "distribution": {
    "1": 0.1667, "2": 0.1667, "3": 0.1667,
    "4": 0.1667, "5": 0.1667, "6": 0.1667
  },
  "samples": ["3", "5", "1", "2", "6", "4", ...]  // multiple examples
}
```

### Phase 3: Training Approaches

**Option A: Multi-sample SFT**
- For each prompt, randomly sample from target distribution
- Model sees same prompt with different correct answers
- Train with standard cross-entropy loss

**Option B: Distribution-aware loss**
- Generate K samples per prompt during training
- Measure KL divergence between model distribution and target
- Custom loss = CE loss + λ * KL divergence

## Open Questions

1. **Model size**: Small models (1-7B) to study learnability, or larger models (70B+)?

2. **Task generation**: Start by manually creating ~50-100 diverse seed tasks, then use LLM to expand to 10K+ variations?

3. **Distribution representation**: How should the model learn the target distribution?
   - Implicit (see many examples during training)
   - Explicit (include distribution in prompt: "Uniform random from 1-6")
   - Hybrid (both approaches)

4. **Interesting research angles**:
   - Can models learn to be "more random" than their temperature sampling allows?
   - Does model size correlate with randomness capability?
   - Can models generalize to unseen random distributions?

## Next Steps

Potential components to build:
- Task generation script using an LLM API
- Dataset creation pipeline
- Training code with distribution-aware evaluation
