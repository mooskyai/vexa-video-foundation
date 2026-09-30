from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor


class ByteTokenizer:
    """A deterministic tokenizer with no learned or downloaded vocabulary.

    IDs:
      0 PAD
      1 BOS
      2 EOS
      3 UNK (reserved)
      4..259 raw byte values 0..255
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        byte_values = list(text.encode("utf-8"))[: max_length - 2]
        ids = [self.bos_id, *[value + 4 for value in byte_values], self.eos_id]
        ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, ids: list[int]) -> str:
        values: list[int] = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            if 4 <= token_id <= 259:
                values.append(token_id - 4)
        return bytes(values).decode("utf-8", errors="replace")

    def batch(self, texts: list[str], max_length: int, device: torch.device | None = None) -> TokenBatch:
        rows = [self.encode(text, max_length) for text in texts]
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attention_mask = input_ids.ne(self.pad_id)
        return TokenBatch(input_ids=input_ids, attention_mask=attention_mask)


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=layers,
                        enable_nested_tensor=False,
                    )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds configured max_length")
        positions = self.position_embedding[:, : input_ids.shape[1]]
        x = self.token_embedding(input_ids) + positions
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        return self.norm(x)
