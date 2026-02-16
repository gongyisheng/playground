import torch

from model import Qwen3Model


def sample(logits, temperature=1.0, top_k=-1):
    if temperature == 0 or top_k == 1:
        next_token = torch.argmax(logits, dim=-1)
        return next_token

    # topk filter
    if top_k > 0:
        values, indices = torch.topk(logits, k=top_k)
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, indices, values)

    # softmax, temperature sampling
    probs = torch.softmax(logits / temperature, dim=-1)

    # sample
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(
    model: Qwen3Model,
    prompt_token_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=-1,
    eos_token_id=None,
):
    # add batch dim: [seq] -> [1, seq]
    input_ids = prompt_token_ids.unsqueeze(0)
    prompt_len = input_ids.shape[1]

    # prefill
    logits, kv_cache = model(input_ids)
    next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
    generated = [next_token.item()]

    # decode loop
    for _ in range(max_new_tokens - 1):
        offset = prompt_len + len(generated) - 1
        logits, kv_cache = model(next_token.view(1, 1), offset, kv_cache)
        next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
        generated.append(next_token.item())
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return prompt_token_ids.tolist() + generated
