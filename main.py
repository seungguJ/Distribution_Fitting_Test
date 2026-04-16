from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from cdf_builder import build_empirical_cdf
from data_generator import generate_dataset
from evaluator import build_artifact_stem, build_plot_filename, mae, monotonic_violations, mse, r2_score, rmse, save_cdf_csv, save_cdf_plot, save_metrics, save_samples_csv, summarize_samples
from model import TorchCDFRegressor, fit_model, predict_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config file must contain a mapping")
    return config


def normalize_x(x_values: list[int], max_value: int) -> list[float]:
    return [value / max_value for value in x_values]


def train_test_split(
    x_values: list[float],
    y_values: list[float],
    train_ratio: float,
    seed: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    total_points = len(x_values)
    train_size = max(1, min(total_points - 1, int(total_points * train_ratio)))
    indices = list(range(total_points))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_indices = set(indices[:train_size])

    x_train: list[float] = []
    y_train: list[float] = []
    x_test: list[float] = []
    y_test: list[float] = []

    for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        if index in train_indices:
            x_train.append(x_value)
            y_train.append(y_value)
        else:
            x_test.append(x_value)
            y_test.append(y_value)
    return x_train, x_test, y_train, y_test


def run_experiment(config: dict, output_root: Path) -> dict:
    samples = generate_dataset(config)
    x_values, cdf_values, _counts = build_empirical_cdf(samples, config["N"])
    normalized_x = normalize_x(x_values, config["N"])

    x_train, x_test, y_train, y_test = train_test_split(
        normalized_x,
        cdf_values,
        config["train_ratio"],
        config["seed"],
    )
    model = TorchCDFRegressor(hidden_size=config["hidden_size"], seed=config["seed"])
    losses = fit_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
    )

    predicted_all = predict_model(model, normalized_x)
    predicted_test = predict_model(model, x_test)

    metrics = {
        "config": config,
        "model_name": model.model_name,
        "model_parameter_count": model.parameter_count,
        "sample_summary": summarize_samples(samples),
        "train_points": len(x_train),
        "test_points": len(x_test),
        "train_final_loss": losses[-1],
        "full_mse": mse(cdf_values, predicted_all),
        "full_rmse": rmse(cdf_values, predicted_all),
        "full_mae": mae(cdf_values, predicted_all),
        "full_r2": r2_score(cdf_values, predicted_all),
        "test_mse": mse(y_test, predicted_test) if y_test else None,
        "test_rmse": rmse(y_test, predicted_test) if y_test else None,
        "test_mae": mae(y_test, predicted_test) if y_test else None,
        "test_r2": r2_score(y_test, predicted_test) if y_test else None,
        "monotonic_violations": monotonic_violations(predicted_all),
    }

    plot_limit = metrics["sample_summary"]["max"]
    plot_x_values = x_values[: plot_limit + 1]
    plot_actual = cdf_values[: plot_limit + 1]
    plot_predicted = predicted_all[: plot_limit + 1]
    artifact_stem = build_artifact_stem(config)
    named_metrics_path = output_root / "metrics" / f"metrics_{artifact_stem}.json"
    named_samples_path = output_root / "data" / f"samples_{artifact_stem}.csv"
    named_cdf_path = output_root / "data" / f"cdf_{artifact_stem}.csv"
    named_plot_path = output_root / "plots" / build_plot_filename(config, metrics["sample_summary"])
    save_metrics(named_metrics_path, metrics)
    save_samples_csv(named_samples_path, samples)
    save_cdf_csv(named_cdf_path, x_values, cdf_values, predicted_all)
    save_cdf_plot(named_plot_path, plot_x_values, plot_actual, plot_predicted, metrics["sample_summary"])
    metrics["named_metrics_path"] = str(named_metrics_path)
    metrics["named_samples_path"] = str(named_samples_path)
    metrics["named_cdf_path"] = str(named_cdf_path)
    metrics["named_plot_path"] = str(named_plot_path)
    save_metrics(named_metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate an empirical CDF with a torch-based regressor.")
    parser.add_argument("--config", default="config.yaml", help="path to a YAML config file")
    parser.add_argument("--output-root", default="outputs", help="directory for metrics, data, and plots")
    parser.add_argument("--distribution-mode", default=None, help="override distribution_mode from config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.distribution_mode is not None:
        config["distribution_mode"] = args.distribution_mode
    output_root = Path(args.output_root)
    metrics = run_experiment(config, output_root)

    print("Run complete")
    print(f"model={metrics['model_name']}, parameter_count={metrics['model_parameter_count']}")
    print(f"sample mean={metrics['sample_summary']['mean']:.4f}, sample variance={metrics['sample_summary']['variance']:.4f}")
    print(f"full mse={metrics['full_mse']:.6f}, full rmse={metrics['full_rmse']:.6f}, full mae={metrics['full_mae']:.6f}, full r2={metrics['full_r2']:.6f}")
    print(f"test mse={metrics['test_mse']:.6f}, test rmse={metrics['test_rmse']:.6f}, test mae={metrics['test_mae']:.6f}, test r2={metrics['test_r2']:.6f}")
    print(f"monotonic violations={metrics['monotonic_violations']}")
    print(f"outputs written to {output_root}/")


if __name__ == "__main__":
    main()
