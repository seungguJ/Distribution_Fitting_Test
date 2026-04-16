from __future__ import annotations

import torch
from torch import nn


class TorchCDFRegressor(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        layers: list[nn.Module] = [nn.Linear(4, hidden_size), nn.ReLU()]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_activation = nn.Sigmoid()

    @property
    def model_name(self) -> str:
        return "TorchCDFRegressorReLU"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _build_features(self, x_value: torch.Tensor) -> torch.Tensor:
        safe_x = torch.clamp(x_value, min=0.0, max=1.0)
        log_feature = torch.log1p(999.0 * safe_x) / torch.log(torch.tensor(1000.0, device=safe_x.device))
        sqrt_feature = torch.sqrt(safe_x)
        square_feature = safe_x * safe_x
        return torch.cat([safe_x, sqrt_feature, square_feature, log_feature], dim=1)

    def forward(self, x_value: torch.Tensor) -> torch.Tensor:
        features = self._build_features(x_value)
        hidden = self.hidden_net(features)
        raw_output = self.output_layer(hidden)
        output = self.output_activation(raw_output)
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
