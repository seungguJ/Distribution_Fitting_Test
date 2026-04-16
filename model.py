from __future__ import annotations

import torch
from torch import nn


class TorchCDFRegressor(nn.Module):
    def __init__(self, hidden_size: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.feature_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    @property
    def model_name(self) -> str:
        return "TorchCDFRegressor"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x_value: torch.Tensor) -> torch.Tensor:
        nonnegative_scale = self.feature_net(x_value)
        raw_output = x_value * nonnegative_scale
        output = (2.0 * torch.sigmoid(raw_output)) - 1.0
        return torch.clamp(output, min=0.0, max=1.0 - 1e-7)


def fit_model(
    model: TorchCDFRegressor,
    x_train: list[float],
    y_train: list[float],
    learning_rate: float,
    epochs: int,
) -> list[float]:
    x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def predict_model(model: TorchCDFRegressor, x_values: list[float]) -> list[float]:
    x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor).squeeze(1).tolist()
    return [float(value) for value in predictions]
