# Transformer

## Architecture

## Model Family

| Architecture   | Model         | Use Case                                                  |
|----------------|---------------|-----------------------------------------------------------|
| Encoder-only   | BERT, RoBERTa | Text classification, NER, Question answering(extractive)  |
| Decoder-only   | GPT, LLaMa    | Text generation, Question answering(generative)           |
| Encoder-decoder| T5, BART      | Translation, Summarization                                |

## Inference
Prefill Phase: process input tokens at once
- Tokenization: Converting the input text into tokens (think of these as the basic building blocks the model understands)
- Embedding Conversion: Transforming these tokens into numerical representations that capture their meaning
- Initial Processing: Running these embeddings through the model’s neural networks to create a rich understanding of the context

Decode Phase: autoregressive token generation
- Attention Computation: Looking back at all previous tokens to understand context
- Probability Calculation: Determining the likelihood of each possible next token
- Token Selection: Choosing the next token based on these probabilities
- Continuation Check: Deciding whether to continue or stop generation
