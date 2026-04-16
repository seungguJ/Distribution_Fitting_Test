from __future__ import annotations


def build_empirical_cdf(samples: list[int], max_value: int) -> tuple[list[int], list[float], list[int]]:
    counts = [0 for _ in range(max_value + 1)]
    for sample in samples:
        counts[sample] += 1

    cdf_values: list[float] = []
    cumulative = 0
    total = len(samples)
    for count in counts:
        cumulative += count
        cdf_values.append(cumulative / total)

    x_values = list(range(max_value + 1))
    return x_values, cdf_values, counts
