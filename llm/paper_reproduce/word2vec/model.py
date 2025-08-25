from torch import nn, Tensor

from config import Word2VecConfig


class CbowModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = Word2VecConfig.vocab_size,
        embedding_dim: int = Word2VecConfig.embedding_dim,
    ):
        super(CbowModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs: Tensor):
        embeds = self.embedding(
            context_idxs
        )  # [batch_size, context_size, embedding_dim]
        sum_embeds = embeds.sum(dim=1)  # [batch_size, embedding_dim]
        output = self.linear(sum_embeds)  # [batch_size, vocab_size]
        return output


class SkipGramModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = Word2VecConfig.vocab_size,
        embedding_dim: int = Word2VecConfig.embedding_dim,
    ):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Embedding(embedding_dim, vocab_size)

    def forward(self, context_ids: Tensor):
        embeds = self.embedding(context_ids)  # [batch_size, embedding_dim]
        output = self.linear(embeds)  # [batch_size, vocab_size]
        return output


if __name__ == "__main__":
    # Example usage
    cbow_model = CbowModel()
    # print model parameters count
    total_params = sum(p.numel() for p in cbow_model.parameters())
    print(f"Total parameters (CBOW): {total_params}")

    skip_gram_model = SkipGramModel()
    # print model parameters count
    total_params_skip_gram = sum(p.numel() for p in skip_gram_model.parameters())
    print(f"Total parameters (SkipGram): {total_params_skip_gram}")
