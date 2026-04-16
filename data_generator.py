from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    N: int
    sample_size: int
    mean: float
    variance: float
    seed: int
    distribution_mode: str


def _clip_int(value: float, upper: int) -> int:
    return max(0, min(upper, int(round(value))))


def _bounded_gauss(rng: random.Random, mean: float, stddev: float, upper: int) -> int:
    return _clip_int(rng.gauss(mean, stddev), upper)


def _lognormal(rng: random.Random, cfg: GeneratorConfig) -> int:
    bounded_mean = min(max(cfg.mean, 1.0), cfg.N)
    bounded_variance = max(cfg.variance, 1.0)
    sigma_squared = math.log(1.0 + (bounded_variance / (bounded_mean * bounded_mean)))
    sigma = math.sqrt(sigma_squared)
    mu = math.log(bounded_mean) - (sigma_squared / 2.0)
    raw_value = rng.lognormvariate(mu, sigma)
    return _clip_int(raw_value, cfg.N)


def _low_biased(rng: random.Random, cfg: GeneratorConfig) -> int:
    alpha = max(1.2, (cfg.mean + 1) / max(cfg.N - cfg.mean, 1))
    beta = max(2.0, cfg.N / max(cfg.mean + 1, 1))
    return _clip_int(rng.betavariate(alpha, beta) * cfg.N, cfg.N)


def _high_biased(rng: random.Random, cfg: GeneratorConfig) -> int:
    alpha = max(2.0, cfg.N / max(cfg.mean + 1, 1))
    beta = max(1.2, (cfg.mean + 1) / max(cfg.N - cfg.mean, 1))
    return _clip_int(rng.betavariate(alpha, beta) * cfg.N, cfg.N)


def _wide_spread(rng: random.Random, cfg: GeneratorConfig) -> int:
    stddev = max(math.sqrt(max(cfg.variance, 1.0)), cfg.N / 3)
    return _bounded_gauss(rng, cfg.mean, stddev, cfg.N)


def _edge_focused(rng: random.Random, cfg: GeneratorConfig) -> int:
    stddev = max(math.sqrt(max(cfg.variance, 1.0)) / 2, 1.0)
    if rng.random() < 0.5:
        center = cfg.mean / 3
    else:
        center = cfg.N - ((cfg.N - cfg.mean) / 3)
    return _bounded_gauss(rng, center, stddev, cfg.N)


def _noisy_random(rng: random.Random, cfg: GeneratorConfig) -> int:
    if rng.random() < 0.5:
        return rng.randint(0, cfg.N)
    stddev = max(math.sqrt(max(cfg.variance, 1.0)), 1.0)
    return _bounded_gauss(rng, cfg.mean, stddev, cfg.N)


def _mixed(rng: random.Random, cfg: GeneratorConfig) -> int:
    modes = (_lognormal, _low_biased, _high_biased, _wide_spread, _edge_focused, _noisy_random)
    generator = rng.choice(modes)
    return generator(rng, cfg)


MODE_TO_GENERATOR = {
    "lognormal": _lognormal,
    "low_biased": _low_biased,
    "high_biased": _high_biased,
    "wide_spread": _wide_spread,
    "edge_focused": _edge_focused,
    "noisy_random": _noisy_random,
    "mixed": _mixed,
}


def generate_dataset(config: dict) -> list[int]:
    cfg = GeneratorConfig(
        N=config["N"],
        sample_size=config["sample_size"],
        mean=config["mean"],
        variance=config["variance"],
        seed=config["seed"],
        distribution_mode=config["distribution_mode"],
    )
    if cfg.N <= 0:
        raise ValueError("N must be greater than 0")
    if cfg.sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if cfg.distribution_mode not in MODE_TO_GENERATOR:
        raise ValueError(f"unsupported distribution_mode: {cfg.distribution_mode}")

    rng = random.Random(cfg.seed)
    generator = MODE_TO_GENERATOR[cfg.distribution_mode]
    return [generator(rng, cfg) for _ in range(cfg.sample_size)]
