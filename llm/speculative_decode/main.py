#!/usr/bin/env python3
"""
Speculative Decoding with Qwen3 models using HuggingFace Transformers.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


class SpeculativeDecoder:
    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_dtype: torch.dtype = torch.bfloat16,
        draft_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize speculative decoder with target and draft models.

        Args:
            target_model_name: HuggingFace model name for the target (large) model
            draft_model_name: HuggingFace model name for the draft (small) model
            device: Device to run models on
            target_dtype: Data type for target model
            draft_dtype: Data type for draft model
        """
        print(f"Loading draft model: {draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=draft_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.draft_model.eval()

        print(f"Loading target model: {target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=target_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.target_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.vocab_size = self.target_model.config.vocab_size

    @torch.no_grad()
    def draft_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate k draft tokens using the draft model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            k: Number of tokens to draft
            temperature: Sampling temperature

        Returns:
            draft_tokens: Generated draft tokens [batch_size, k]
            draft_probs: Probability of each drafted token [batch_size, k]
        """
        batch_size = input_ids.shape[0]
        draft_tokens = []
        draft_probs = []

        current_ids = input_ids
        current_mask = attention_mask

        for _ in range(k):
            outputs = self.draft_model(
                input_ids=current_ids,
                attention_mask=current_mask,
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]
                probs = F.softmax(logits, dim=-1)

            # Get probability of the sampled token
            token_probs = probs.gather(dim=-1, index=next_token)  # [batch_size, 1]

            draft_tokens.append(next_token)
            draft_probs.append(token_probs)

            # Update for next iteration
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            current_mask = torch.cat([
                current_mask,
                torch.ones((batch_size, 1), device=self.device, dtype=current_mask.dtype)
            ], dim=-1)

        draft_tokens = torch.cat(draft_tokens, dim=-1)  # [batch_size, k]
        draft_probs = torch.cat(draft_probs, dim=-1)    # [batch_size, k]

        return draft_tokens, draft_probs

    @torch.no_grad()
    def target_verify(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Verify draft tokens with target model in a single forward pass.

        Args:
            input_ids: Original input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            draft_tokens: Draft tokens to verify [batch_size, k]
            temperature: Sampling temperature

        Returns:
            target_probs: Target model probabilities for all positions [batch_size, k+1, vocab_size]
        """
        batch_size = input_ids.shape[0]
        k = draft_tokens.shape[1]

        # Concatenate input with draft tokens
        full_input = torch.cat([input_ids, draft_tokens], dim=-1)
        full_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, k), device=self.device, dtype=attention_mask.dtype)
        ], dim=-1)

        # Single forward pass through target model
        outputs = self.target_model(
            input_ids=full_input,
            attention_mask=full_mask,
        )

        # Get logits for positions we need to verify (last k+1 positions)
        # Position -k-1 predicts first draft token, position -1 predicts next token after last draft
        logits = outputs.logits[:, -(k+1):, :]  # [batch_size, k+1, vocab_size]

        if temperature > 0:
            target_probs = F.softmax(logits / temperature, dim=-1)
        else:
            target_probs = F.softmax(logits, dim=-1)

        return target_probs

    @torch.no_grad()
    def speculative_sample(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply speculative sampling to accept/reject draft tokens.

        Args:
            draft_tokens: Draft tokens [batch_size, k]
            draft_probs: Draft token probabilities [batch_size, k]
            target_probs: Target probabilities [batch_size, k+1, vocab_size]

        Returns:
            accepted_tokens: Tokens to append [batch_size, max_accepted]
            num_accepted: Number of accepted tokens per batch item [batch_size]
        """
        batch_size, k = draft_tokens.shape
        device = draft_tokens.device

        # Get target probabilities for draft tokens
        # target_probs[:, i, :] gives prob dist for position i (predicting token i)
        target_probs_for_draft = target_probs[:, :k, :].gather(
            dim=-1,
            index=draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, k]

        # Compute acceptance ratios
        acceptance_ratios = target_probs_for_draft / (draft_probs + 1e-10)
        acceptance_ratios = torch.clamp(acceptance_ratios, max=1.0)

        # Sample uniform random numbers for rejection sampling
        random_vals = torch.rand_like(acceptance_ratios)

        # Determine which tokens to accept (accept if random < ratio)
        accepted_mask = random_vals < acceptance_ratios  # [batch_size, k]

        # Find first rejection point for each batch item
        # We accept tokens until the first rejection
        cumulative_accepted = accepted_mask.cumprod(dim=-1)  # [batch_size, k]
        num_accepted = cumulative_accepted.sum(dim=-1)  # [batch_size]

        # Prepare output tokens (we'll include up to k+1 tokens)
        # Initialize with draft tokens + one extra position
        output_tokens = torch.zeros((batch_size, k + 1), dtype=draft_tokens.dtype, device=device)
        output_tokens[:, :k] = draft_tokens

        # For each batch item, sample the next token after accepted sequence
        for b in range(batch_size):
            n_acc = num_accepted[b].item()

            if n_acc < k:
                # Some tokens were rejected - sample from modified distribution
                # p'(x) = max(0, p_target(x) - p_draft(x)) / Z
                draft_token_idx = draft_tokens[b, n_acc].item()

                p_target = target_probs[b, n_acc, :]

                # Get draft model's distribution at this position
                # We need to recompute or we can use the adjusted distribution
                # For simplicity, use target distribution for the corrected sample
                p_diff = p_target.clone()
                p_diff[draft_token_idx] = 0  # Zero out rejected token prob

                if p_diff.sum() > 0:
                    p_diff = p_diff / p_diff.sum()
                    next_token = torch.multinomial(p_diff, num_samples=1)
                else:
                    next_token = torch.multinomial(p_target, num_samples=1)

                output_tokens[b, n_acc] = next_token.item()
            else:
                # All tokens accepted - sample one more from target
                next_token = torch.multinomial(target_probs[b, k, :], num_samples=1)
                output_tokens[b, k] = next_token.item()
                num_accepted[b] = k + 1

        return output_tokens, num_accepted

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        k: int = 4,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> list[str]:
        """
        Generate text using speculative decoding.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            k: Number of draft tokens to propose at each step
            temperature: Sampling temperature
            verbose: Whether to print progress

        Returns:
            List of generated texts
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        # Track generation state
        current_ids = input_ids
        current_mask = attention_mask
        tokens_generated = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        total_draft_tokens = 0
        total_accepted_tokens = 0

        start_time = time.time()
        step = 0

        while not finished.all() and tokens_generated.max() < max_new_tokens:
            step += 1

            # Adjust k for remaining tokens
            remaining = max_new_tokens - tokens_generated.max().item()
            current_k = min(k, remaining)

            if current_k <= 0:
                break

            # Draft generation
            draft_tokens, draft_probs = self.draft_generate(
                current_ids, current_mask, current_k, temperature
            )
            total_draft_tokens += current_k * batch_size

            # Target verification
            target_probs = self.target_verify(
                current_ids, current_mask, draft_tokens, temperature
            )

            # Speculative sampling
            output_tokens, num_accepted = self.speculative_sample(
                draft_tokens, draft_probs, target_probs
            )

            # Update sequences
            for b in range(batch_size):
                if finished[b]:
                    continue

                n_acc = num_accepted[b].item()
                new_tokens = output_tokens[b, :n_acc]

                total_accepted_tokens += n_acc
                tokens_generated[b] += n_acc

                # Check for EOS
                if self.tokenizer.eos_token_id in new_tokens:
                    finished[b] = True

            # Properly update all sequences
            max_new = num_accepted.max().item()
            new_ids = torch.zeros((batch_size, max_new), dtype=torch.long, device=self.device)
            new_mask = torch.zeros((batch_size, max_new), dtype=attention_mask.dtype, device=self.device)

            for b in range(batch_size):
                n_acc = num_accepted[b].item()
                new_ids[b, :n_acc] = output_tokens[b, :n_acc]
                new_mask[b, :n_acc] = 1

            current_ids = torch.cat([current_ids, new_ids], dim=-1)
            current_mask = torch.cat([current_mask, new_mask], dim=-1)

            if verbose and step % 10 == 0:
                acceptance_rate = total_accepted_tokens / max(total_draft_tokens, 1)
                print(f"Step {step}: Generated {tokens_generated.float().mean().item():.1f} tokens, "
                      f"Acceptance rate: {acceptance_rate:.2%}")

        elapsed = time.time() - start_time

        if verbose:
            acceptance_rate = total_accepted_tokens / max(total_draft_tokens, 1)
            tokens_per_sec = total_accepted_tokens / elapsed
            print(f"\nGeneration complete:")
            print(f"  Total tokens generated: {total_accepted_tokens}")
            print(f"  Total draft tokens: {total_draft_tokens}")
            print(f"  Acceptance rate: {acceptance_rate:.2%}")
            print(f"  Tokens/second: {tokens_per_sec:.1f}")
            print(f"  Time elapsed: {elapsed:.2f}s")

        # Decode outputs
        outputs = []
        for b in range(batch_size):
            # Find the actual sequence length (exclude padding)
            seq = current_ids[b]
            outputs.append(self.tokenizer.decode(seq, skip_special_tokens=True))

        return outputs


def main():

    target_model = "Qwen/Qwen3-14B"
    draft_model = "Qwen/Qwen3-0.6B"

    max_tokens = 1024
    k = 4

    # Initialize decoder
    decoder = SpeculativeDecoder(
        target_model_name=target_model,
        draft_model_name=draft_model
    )

    # Prepare prompts
    prompts = [
        "introduce yourself",
        "which is bigger, 9.9 or 9.11",
        "how many Rs in \"strawberry\""
    ]

    print(f"\nGenerating with k={k} draft tokens...")
    print(f"Prompts: {prompts}\n")

    # Generate
    outputs = decoder.generate(
        prompts=prompts,
        max_new_tokens=max_tokens,
        k=k
    )

    # Print results
    print("\n" + "="*50)
    print("Generated outputs:")
    print("="*50)
    for i, output in enumerate(outputs):
        print(f"\n[{i+1}] {output}")
        print("-"*50)


if __name__ == "__main__":
    main()
