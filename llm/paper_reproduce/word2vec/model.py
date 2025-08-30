from torch import nn, Tensor

from config import Word2VecConfig


class CbowModel(nn.Module):
    def __init__(self, config: Word2VecConfig):
        super(CbowModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, context_idxs: Tensor):
        embeds = self.embedding(
            context_idxs
        )  # [batch_size, context_size, embedding_dim]
        sum_embeds = embeds.mean(dim=1)  # [batch_size, embedding_dim]
        output = self.linear(sum_embeds)  # [batch_size, vocab_size]
        return output


class SkipGramModel(nn.Module):
    def __init__(self, config: Word2VecConfig):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, context_ids: Tensor):
        embeds = self.embedding(context_ids)  # [batch_size, embedding_dim]
        output = self.linear(embeds)  # [batch_size, vocab_size]
        return output


if __name__ == "__main__":

    config = Word2VecConfig()

    # Example usage
    cbow_model = CbowModel(config)
    # print model parameters count
    total_params = sum(p.numel() for p in cbow_model.parameters())
    print(f"Total parameters (CBOW): {total_params}")

    skip_gram_model = SkipGramModel(config)
    # print model parameters count
    total_params_skip_gram = sum(p.numel() for p in skip_gram_model.parameters())
    print(f"Total parameters (SkipGram): {total_params_skip_gram}")
