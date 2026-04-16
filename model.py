from __future__ import annotations

import math
import random


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softplus(value: float) -> float:
    if value > 20:
        return value
    return math.log1p(math.exp(value))


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


class MonotonicCDFRegressor:
    def __init__(self, hidden_size: int, learning_rate: float, epochs: int, seed: int) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        rng = random.Random(seed)
        self.alpha_raw = [rng.uniform(-0.2, 0.2) for _ in range(hidden_size)]
        self.beta_raw = [rng.uniform(-0.2, 0.2) for _ in range(hidden_size)]
        self.gamma = [rng.uniform(-4.0, 4.0) for _ in range(hidden_size)]

    @property
    def model_name(self) -> str:
        return "MonotonicCDFRegressor"

    @property
    def parameter_count(self) -> int:
        return self.hidden_size * 3

    def _forward(self, x_value: float) -> tuple[list[float], list[float], list[float], float]:
        alpha = _softmax(self.alpha_raw)
        beta = [_softplus(value) + 1e-4 for value in self.beta_raw]
        activations = [_sigmoid((beta_value * x_value) + gamma_value) for beta_value, gamma_value in zip(beta, self.gamma)]
        prediction = sum(alpha_value * activation for alpha_value, activation in zip(alpha, activations))
        return alpha, beta, activations, prediction

    def fit(self, x_train: list[float], y_train: list[float]) -> list[float]:
        losses: list[float] = []
        for _ in range(self.epochs):
            loss_sum = 0.0
            for x_value, target in zip(x_train, y_train):
                alpha, _beta, activations, prediction = self._forward(x_value)
                error = prediction - target
                loss_sum += error * error

                alpha_raw_grads = [error * alpha_value * (activation - prediction) for alpha_value, activation in zip(alpha, activations)]
                beta_raw_grads: list[float] = []
                gamma_grads: list[float] = []
                for alpha_value, beta_raw_value, activation in zip(alpha, self.beta_raw, activations):
                    activation_grad = activation * (1.0 - activation)
                    slope_grad = _sigmoid(beta_raw_value)
                    beta_raw_grads.append(error * alpha_value * activation_grad * x_value * slope_grad)
                    gamma_grads.append(error * alpha_value * activation_grad)

                for index in range(self.hidden_size):
                    self.alpha_raw[index] -= self.learning_rate * alpha_raw_grads[index]
                    self.beta_raw[index] -= self.learning_rate * beta_raw_grads[index]
                    self.gamma[index] -= self.learning_rate * gamma_grads[index]

            losses.append(loss_sum / len(x_train))
        return losses

    def predict(self, x_values: list[float]) -> list[float]:
        predictions: list[float] = []
        for x_value in x_values:
            _, _, _, prediction = self._forward(x_value)
            predictions.append(prediction)
        return predictions
