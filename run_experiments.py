from __future__ import annotations

import csv
import json
from pathlib import Path

from main import run_experiment


BASE_CONFIG = {
    "N": 100,
    "sample_size": 5000,
    "seed": 42,
    "train_ratio": 0.8,
    "hidden_size": 12,
    "epochs": 3000,
    "learning_rate": 0.08,
}

PROFILES = {
    "low_mean_low_var": {"mean": 15, "variance": 40},
    "mid_mean_mid_var": {"mean": 50, "variance": 200},
    "high_mean_high_var": {"mean": 85, "variance": 500},
}

MODES = [
    "lognormal",
    "low_biased",
    "high_biased",
    "wide_spread",
    "edge_focused",
    "noisy_random",
    "mixed",
]


def build_cases() -> list[dict]:
    cases: list[dict] = []
    seed = BASE_CONFIG["seed"]
    for mode_index, mode in enumerate(MODES):
        for profile_index, (profile_name, profile_values) in enumerate(PROFILES.items()):
            case = dict(BASE_CONFIG)
            case.update(profile_values)
            case["distribution_mode"] = mode
            case["seed"] = seed + (mode_index * 100) + profile_index
            case["case_name"] = f"{mode}__{profile_name}"
            cases.append(case)
    return cases


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "distribution_mode",
        "mean",
        "variance",
        "sample_mean",
        "sample_variance",
        "full_mse",
        "full_mae",
        "full_r2",
        "test_mse",
        "test_mae",
        "test_r2",
        "monotonic_violations",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(summary_rows: list[dict]) -> dict:
    best_r2 = max(summary_rows, key=lambda row: row["full_r2"])
    worst_r2 = min(summary_rows, key=lambda row: row["full_r2"])
    worst_test_mse = max(summary_rows, key=lambda row: row["test_mse"])
    avg_full_r2 = sum(row["full_r2"] for row in summary_rows) / len(summary_rows)
    avg_test_r2 = sum(row["test_r2"] for row in summary_rows) / len(summary_rows)
    avg_full_mse = sum(row["full_mse"] for row in summary_rows) / len(summary_rows)
    total_violations = sum(row["monotonic_violations"] for row in summary_rows)
    return {
        "case_count": len(summary_rows),
        "average_full_r2": avg_full_r2,
        "average_test_r2": avg_test_r2,
        "average_full_mse": avg_full_mse,
        "total_monotonic_violations": total_violations,
        "best_full_r2_case": best_r2,
        "worst_full_r2_case": worst_r2,
        "worst_test_mse_case": worst_test_mse,
    }


def main() -> None:
    output_root = Path("outputs/batch")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for case in build_cases():
        case_name = case["case_name"]
        case_output_root = output_root / case_name
        case_config = {key: value for key, value in case.items() if key != "case_name"}
        metrics = run_experiment(case_config, case_output_root)
        summary_rows.append(
            {
                "case_name": case_name,
                "distribution_mode": case_config["distribution_mode"],
                "mean": case_config["mean"],
                "variance": case_config["variance"],
                "sample_mean": metrics["sample_summary"]["mean"],
                "sample_variance": metrics["sample_summary"]["variance"],
                "full_mse": metrics["full_mse"],
                "full_mae": metrics["full_mae"],
                "full_r2": metrics["full_r2"],
                "test_mse": metrics["test_mse"],
                "test_mae": metrics["test_mae"],
                "test_r2": metrics["test_r2"],
                "monotonic_violations": metrics["monotonic_violations"],
            }
        )
        print(
            f"{case_name}: full_r2={metrics['full_r2']:.4f}, "
            f"test_r2={metrics['test_r2']:.4f}, full_mse={metrics['full_mse']:.6f}, "
            f"violations={metrics['monotonic_violations']}"
        )

    summary_path = output_root / "summary.csv"
    report_path = output_root / "report.json"
    write_summary_csv(summary_path, summary_rows)
    report = build_report(summary_rows)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"batch summary written to {summary_path}")
    print(f"batch report written to {report_path}")


if __name__ == "__main__":
    main()
