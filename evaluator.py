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
        "size": len(samples),
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


def save_cdf_plot(path: Path, x_values: list[int], actual: list[float], predicted: list[float], sample_summary: dict) -> None:
    width = 980
    height = 620
    left = 70
    right = 40
    top = 60
    bottom = 70
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

    def project_point(x_value: int, y_value: float) -> tuple[float, float]:
        px = left + (x_value / max_x) * plot_width if max_x else left
        py = top + (1 - y_value) * plot_height
        return px, py

    predicted_points = project(list(zip(x_values, predicted)))
    actual_scatter = "\n".join(
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="#1f77b4" fill-opacity="0.75" />'
        for px, py in (project_point(x_value, y_value) for x_value, y_value in zip(x_values, actual))
    )
    y_ticks = "\n".join(
        f'<g><line x1="{left}" y1="{top + (1 - tick) * plot_height:.2f}" x2="{width - right}" y2="{top + (1 - tick) * plot_height:.2f}" stroke="#e6e1d7" stroke-width="1" /><text x="{left - 10}" y="{top + (1 - tick) * plot_height + 5:.2f}" font-size="12" text-anchor="end">{tick:.1f}</text></g>'
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    x_ticks = "\n".join(
        f'<g><line x1="{left + ratio * plot_width:.2f}" y1="{top}" x2="{left + ratio * plot_width:.2f}" y2="{height - bottom}" stroke="#f0ece3" stroke-width="1" /><text x="{left + ratio * plot_width:.2f}" y="{height - 20}" font-size="12" text-anchor="middle">{int(ratio * max_x)}</text></g>'
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    schema_lines = [
        f"mean: {sample_summary['mean']:.4f}",
        f"variance: {sample_summary['variance']:.4f}",
        f"min: {sample_summary['min']}",
        f"max: {sample_summary['max']}",
        f"size: {sample_summary['size']}",
    ]
    schema_text = "\n".join(
        f'<text x="{width - 255}" y="{130 + (index * 18)}" font-size="13" fill="#1d1d1d">{line}</text>'
        for index, line in enumerate(schema_lines)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fffaf0" />
<text x="{width / 2}" y="28" font-size="22" text-anchor="middle" fill="#1d1d1d">Empirical CDF Scatter vs ML Approximation Line</text>
<text x="{width / 2}" y="48" font-size="13" text-anchor="middle" fill="#5f584d">Scatter points show actual empirical CDF, red line shows ML approximation</text>
{y_ticks}
{x_ticks}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="1.5" />
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#444" stroke-width="1.5" />
{actual_scatter}
<polyline fill="none" stroke="#d62728" stroke-width="3" points="{predicted_points}" />
<rect x="{width - 280}" y="72" width="230" height="130" fill="#ffffff" stroke="#d9d3c7" rx="10" />
<text x="{width - 255}" y="96" font-size="15" font-weight="bold" fill="#1d1d1d">Data Schema</text>
{schema_text}
<rect x="{width - 280}" y="220" width="230" height="76" fill="#ffffff" stroke="#d9d3c7" rx="10" />
<circle cx="{width - 248}" cy="246" r="4" fill="#1f77b4" fill-opacity="0.75" />
<text x="{width - 230}" y="250" font-size="13" fill="#1d1d1d">Actual empirical CDF scatter</text>
<line x1="{width - 252}" y1="272" x2="{width - 212}" y2="272" stroke="#d62728" stroke-width="3" />
<text x="{width - 202}" y="276" font-size="13" fill="#1d1d1d">ML approximation line</text>
<text x="{left + (plot_width / 2)}" y="{height - 30}" font-size="14" text-anchor="middle" fill="#1d1d1d">x value</text>
<text x="24" y="{top + (plot_height / 2)}" font-size="14" text-anchor="middle" fill="#1d1d1d" transform="rotate(-90 24 {top + (plot_height / 2)})">CDF</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
