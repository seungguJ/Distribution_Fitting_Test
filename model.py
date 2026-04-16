from __future__ import annotations

import math
import random


def _tanh(value: float) -> float:
    return math.tanh(value)


class SimpleMLPRegressor:
    def __init__(self, hidden_size: int, learning_rate: float, epochs: int, seed: int) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        rng = random.Random(seed)
        self.hidden_weights = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.hidden_biases = [rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.output_weights = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.output_bias = rng.uniform(-0.5, 0.5)

    def _forward(self, x: float) -> tuple[list[float], float]:
        hidden = [_tanh((weight * x) + bias) for weight, bias in zip(self.hidden_weights, self.hidden_biases)]
        output = sum(weight * value for weight, value in zip(self.output_weights, hidden)) + self.output_bias
        return hidden, output

    def fit(self, x_train: list[float], y_train: list[float]) -> list[float]:
        losses: list[float] = []
        for _ in range(self.epochs):
            loss_sum = 0.0
            for x_value, target in zip(x_train, y_train):
                hidden, prediction = self._forward(x_value)
                error = prediction - target
                loss_sum += error * error

                output_grads = [error * value for value in hidden]
                output_bias_grad = error
                hidden_errors = [error * out_weight * (1 - (hidden_val * hidden_val)) for out_weight, hidden_val in zip(self.output_weights, hidden)]

                for index in range(self.hidden_size):
                    self.output_weights[index] -= self.learning_rate * output_grads[index]
                    self.hidden_weights[index] -= self.learning_rate * hidden_errors[index] * x_value
                    self.hidden_biases[index] -= self.learning_rate * hidden_errors[index]
                self.output_bias -= self.learning_rate * output_bias_grad

            losses.append(loss_sum / len(x_train))
        return losses

    def predict(self, x_values: list[float]) -> list[float]:
        predictions: list[float] = []
        for x_value in x_values:
            _, raw_prediction = self._forward(x_value)
            predictions.append(min(1.0, max(0.0, raw_prediction)))
        return predictions
