"""Train GradLens in vanilla, conflict-measurement, or mitigation mode."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass

import numpy as np

from data import make_multitask_data
from model import MultiTaskMLP, cosine_similarity, flatten_shared
from report import write_outputs


@dataclass
class TrainConfig:
    epochs: int = 18
    hidden_dim: int = 16
    batch_size: int = 32
    learning_rate: float = 0.01
    seed: int = 42
    n_train: int = 384
    n_test: int = 192
    input_dim: int = 10


class Adam:
    def __init__(self, params: dict[str, np.ndarray], learning_rate: float):
        self.learning_rate = learning_rate
        self.step = 0
        self.first = {name: np.zeros_like(value) for name, value in params.items()}
        self.second = {name: np.zeros_like(value) for name, value in params.items()}

    def update(self, params: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        self.step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, gradient in gradients.items():
            self.first[name] = beta1 * self.first[name] + (1 - beta1) * gradient
            self.second[name] = beta2 * self.second[name] + (1 - beta2) * gradient**2
            first_hat = self.first[name] / (1 - beta1**self.step)
            second_hat = self.second[name] / (1 - beta2**self.step)
            params[name] -= self.learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)


def project_conflicting_gradients(
    gradient_a: dict[str, np.ndarray], gradient_b: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """PCGrad-style projection for the shared W1 and b1 gradients.

    If the flattened shared gradients conflict, each one is projected onto
    the normal plane of the other.  Task-specific head gradients are copied
    unchanged because only one task owns each head.
    """

    projected_a = copy.deepcopy(gradient_a)
    projected_b = copy.deepcopy(gradient_b)
    flat_a = flatten_shared(gradient_a)
    flat_b = flatten_shared(gradient_b)
    dot = float(np.dot(flat_a, flat_b))

    if dot >= 0:
        return projected_a, projected_b

    norm_a_sq = float(np.dot(flat_a, flat_a))
    norm_b_sq = float(np.dot(flat_b, flat_b))
    projected_flat_a = flat_a - (dot / norm_b_sq) * flat_b if norm_b_sq else flat_a
    projected_flat_b = flat_b - (dot / norm_a_sq) * flat_a if norm_a_sq else flat_b

    offset = 0
    for name in ("W1", "b1"):
        size = gradient_a[name].size
        projected_a[name] = projected_flat_a[offset : offset + size].reshape(gradient_a[name].shape)
        projected_b[name] = projected_flat_b[offset : offset + size].reshape(gradient_b[name].shape)
        offset += size
    return projected_a, projected_b


def _average_gradients(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {name: (first[name] + second[name]) / 2.0 for name in first}


def _conflict_summary(cosines: list[float]) -> dict[str, float | int]:
    values = np.asarray(cosines, dtype=float)
    return {
        "steps": int(len(values)),
        "conflict_frequency": float(np.mean(values < 0)) if len(values) else 0.0,
        "mean_cosine_similarity": float(np.mean(values)) if len(values) else 0.0,
        "median_cosine_similarity": float(np.median(values)) if len(values) else 0.0,
        "minimum_cosine_similarity": float(np.min(values)) if len(values) else 0.0,
    }


def train_one_mode(mode: str, config: TrainConfig, data) -> dict:
    if mode not in {"vanilla", "measure", "mitigate"}:
        raise ValueError(f"Unknown mode: {mode}")

    model = MultiTaskMLP(config.input_dim, config.hidden_dim, seed=config.seed)
    optimizer = Adam(model.params, config.learning_rate)
    rng = np.random.default_rng(config.seed + 100)
    cosines: list[float] = []
    epoch_losses: list[float] = []

    for _ in range(config.epochs):
        permutation = rng.permutation(len(data.x_train))
        batch_losses = []
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            x_batch = data.x_train[indices]
            loss_a, gradients_a = model.loss_and_gradients(x_batch, data.y_a_train[indices], "a")
            loss_b, gradients_b = model.loss_and_gradients(x_batch, data.y_b_train[indices], "b")
            batch_losses.append((loss_a + loss_b) / 2.0)

            cosine = cosine_similarity(flatten_shared(gradients_a), flatten_shared(gradients_b))
            if mode in {"measure", "mitigate"}:
                cosines.append(cosine)

            if mode == "mitigate":
                gradients_a, gradients_b = project_conflicting_gradients(gradients_a, gradients_b)
            combined = _average_gradients(gradients_a, gradients_b)
            optimizer.update(model.params, combined)
        epoch_losses.append(float(np.mean(batch_losses)))

    train_loss = (
        model.loss(data.x_train, data.y_a_train, "a") + model.loss(data.x_train, data.y_b_train, "b")
    ) / 2.0
    test_loss = (
        model.loss(data.x_test, data.y_a_test, "a") + model.loss(data.x_test, data.y_b_test, "b")
    ) / 2.0

    return {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "epoch_losses": epoch_losses,
        "train_accuracy": model.accuracy(data.x_train, data.y_a_train, data.y_b_train),
        "test_accuracy": model.accuracy(data.x_test, data.y_a_test, data.y_b_test),
        "gradient_conflict": _conflict_summary(cosines) if mode in {"measure", "mitigate"} else None,
        "cosine_similarity_by_step": cosines if mode in {"measure", "mitigate"} else None,
    }


def run_experiment(modes: list[str], config: TrainConfig) -> dict:
    data = make_multitask_data(config.n_train, config.n_test, config.input_dim, config.seed)
    return {
        "experiment": "GradLens multitask gradient-conflict study",
        "config": asdict(config),
        "dataset": {
            "source": "Synthetic tasks generated by data.py",
            "train_rows": int(len(data.x_train)),
            "test_rows": int(len(data.x_test)),
            "input_dim": int(data.x_train.shape[1]),
        },
        "modes": {mode: train_one_mode(mode, config, data) for mode in modes},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GradLens experiment.")
    parser.add_argument("--mode", choices=["vanilla", "measure", "mitigate", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["vanilla", "measure", "mitigate"] if args.mode == "all" else [args.mode]
    config = TrainConfig(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    results = run_experiment(modes, config)
    written = write_outputs(results, args.output_dir)
    print(f"Wrote GradLens outputs: {written}")


if __name__ == "__main__":
    main()
