from __future__ import annotations

import torch
from torch import nn

from .config import Config


class GRUForecastModel(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gru_outputs, _ = self.gru(inputs)
        last_hidden = gru_outputs[:, -1, :]
        return self.output(last_hidden).squeeze(-1)


def build_model(config: Config) -> GRUForecastModel:
    return GRUForecastModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
