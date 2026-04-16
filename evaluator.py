from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def variance(values: list[float], mean_value: float) -> float:
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def mse(y_true: list[float], y_pred: list[float]) -> float:
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    baseline = mean(y_true)
    total = sum((value - baseline) ** 2 for value in y_true)
    residual = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    if total == 0:
        return 1.0
    return 1 - (residual / total)


def monotonic_violations(values: list[float]) -> int:
    violations = 0
    for previous, current in zip(values, values[1:]):
        if current < previous:
            violations += 1
    return violations


def summarize_samples(samples: list[int]) -> dict:
    sample_mean = mean([float(value) for value in samples])
    sample_variance = variance([float(value) for value in samples], sample_mean)
    return {
        "min": min(samples),
        "max": max(samples),
        "mean": sample_mean,
        "variance": sample_variance,
    }


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_samples_csv(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "value"])
        for index, value in enumerate(samples):
            writer.writerow([index, value])


def save_cdf_csv(path: Path, x_values: list[int], cdf_values: list[float], predicted_values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "actual_cdf", "predicted_cdf"])
        for x_value, actual, predicted in zip(x_values, cdf_values, predicted_values):
            writer.writerow([x_value, actual, predicted])


def save_cdf_plot(path: Path, x_values: list[int], actual: list[float], predicted: list[float]) -> None:
    width = 900
    height = 520
    left = 70
    right = 30
    top = 30
    bottom = 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_x = max(x_values) if x_values else 1

    def project(points: list[tuple[int, float]]) -> str:
        projected: list[str] = []
        for x_value, y_value in points:
            px = left + (x_value / max_x) * plot_width if max_x else left
            py = top + (1 - y_value) * plot_height
            projected.append(f"{px:.2f},{py:.2f}")
        return " ".join(projected)

    actual_points = project(list(zip(x_values, actual)))
    predicted_points = project(list(zip(x_values, predicted)))
    y_ticks = "\n".join(
        f'<text x="{left - 10}" y="{top + (1 - tick) * plot_height + 5:.2f}" font-size="12" text-anchor="end">{tick:.1f}</text>'
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    x_ticks = "\n".join(
        f'<text x="{left + ratio * plot_width:.2f}" y="{height - 15}" font-size="12" text-anchor="middle">{int(ratio * max_x)}</text>'
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fffdf7" />
<text x="{width / 2}" y="20" font-size="18" text-anchor="middle" fill="#1d1d1d">Empirical CDF vs Predicted CDF</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="1.5" />
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#444" stroke-width="1.5" />
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{actual_points}" />
<polyline fill="none" stroke="#d62728" stroke-width="3" stroke-dasharray="8 5" points="{predicted_points}" />
<rect x="{width - 240}" y="38" width="190" height="52" fill="#ffffff" stroke="#ddd" />
<line x1="{width - 225}" y1="58" x2="{width - 185}" y2="58" stroke="#1f77b4" stroke-width="3" />
<text x="{width - 175}" y="62" font-size="13" fill="#1d1d1d">Actual CDF</text>
<line x1="{width - 225}" y1="78" x2="{width - 185}" y2="78" stroke="#d62728" stroke-width="3" stroke-dasharray="8 5" />
<text x="{width - 175}" y="82" font-size="13" fill="#1d1d1d">Predicted CDF</text>
{y_ticks}
{x_ticks}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
