import torch
import random

from config import Seq2SeqConfig

class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size)
        embedded = self.dropout(self.embedding(input_ids))
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        # we don't need the output, just the hidden and cell states
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decoder usually forwards one token at a time."""
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size)
        target_embedded = self.dropout(self.embedding(input_ids))
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        lstm_output, (hidden, cell) = self.lstm(target_embedded, (hidden, cell))
        # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, output_size)
        output = self.fc(lstm_output)
        return output, hidden, cell