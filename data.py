"""Synthetic multitask data used by GradLens.

The two binary tasks deliberately share an input direction with opposite
signs.  Task A rewards a positive response to feature 0, while Task B rewards
a negative response to the same feature.  That construction makes gradient
conflict on the shared trunk plausible without claiming that the toy problem
represents a real-world benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiTaskData:
    x_train: np.ndarray
    y_a_train: np.ndarray
    y_b_train: np.ndarray
    x_test: np.ndarray
    y_a_test: np.ndarray
    y_b_test: np.ndarray


def _make_split(
    rng: np.random.Generator, n_rows: int, input_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n_rows, input_dim))
    noise_a = rng.normal(scale=0.35, size=n_rows)
    noise_b = rng.normal(scale=0.35, size=n_rows)

    score_a = x[:, 0] + 0.7 * x[:, 1] - 0.3 * x[:, 2] + 0.25 * np.tanh(x[:, 3]) + noise_a
    score_b = -x[:, 0] + 0.7 * x[:, 3] + 0.3 * x[:, 4] - 0.25 * np.tanh(x[:, 1]) + noise_b

    y_a = (score_a > 0).astype(np.int64)
    y_b = (score_b > 0).astype(np.int64)
    return x, y_a, y_b


def make_multitask_data(
    n_train: int = 384,
    n_test: int = 192,
    input_dim: int = 10,
    seed: int = 42,
) -> MultiTaskData:
    """Create reproducible train and test splits for Tasks A and B."""

    if input_dim < 5:
        raise ValueError("input_dim must be at least 5 for the built-in task rules")

    rng = np.random.default_rng(seed)
    x_train, y_a_train, y_b_train = _make_split(rng, n_train, input_dim)
    x_test, y_a_test, y_b_test = _make_split(rng, n_test, input_dim)

    centre = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    return MultiTaskData(
        x_train=(x_train - centre) / scale,
        y_a_train=y_a_train,
        y_b_train=y_b_train,
        x_test=(x_test - centre) / scale,
        y_a_test=y_a_test,
        y_b_test=y_b_test,
    )
