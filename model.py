"""A tiny multitask MLP implemented and differentiated by hand in NumPy."""

from __future__ import annotations

import numpy as np

PARAMETERS = ("W1", "b1", "W2_a", "b2_a", "W2_b", "b2_b")
SHARED_PARAMETERS = ("W1", "b1")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


class MultiTaskMLP:
    """One shared tanh trunk with one binary classification head per task."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2_a": rng.normal(scale=np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1)),
            "b2_a": np.zeros(1),
            "W2_b": rng.normal(scale=np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1)),
            "b2_b": np.zeros(1),
        }

    def _hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.params["W1"] + self.params["b1"])

    def forward(self, x: np.ndarray, task: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if task not in {"a", "b"}:
            raise ValueError("task must be 'a' or 'b'")
        hidden = self._hidden(x)
        logits = hidden @ self.params[f"W2_{task}"] + self.params[f"b2_{task}"]
        probabilities = _sigmoid(logits)
        cache = {"x": x, "hidden": hidden, "probabilities": probabilities}
        return probabilities, cache

    def loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray, task: str
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Binary cross-entropy loss and exact manual gradients for one task."""

        probabilities, cache = self.forward(x, task)
        y_column = y.astype(float).reshape(-1, 1)
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        loss = -np.mean(y_column * np.log(clipped) + (1 - y_column) * np.log(1 - clipped))

        n_rows = x.shape[0]
        d_logits = (probabilities - y_column) / n_rows
        hidden = cache["hidden"]

        gradients = {name: np.zeros_like(value) for name, value in self.params.items()}
        gradients[f"W2_{task}"] = hidden.T @ d_logits
        gradients[f"b2_{task}"] = d_logits.sum(axis=0)

        d_hidden = d_logits @ self.params[f"W2_{task}"].T
        d_pre_activation = d_hidden * (1.0 - hidden**2)
        gradients["W1"] = x.T @ d_pre_activation
        gradients["b1"] = d_pre_activation.sum(axis=0)
        return float(loss), gradients

    def loss(self, x: np.ndarray, y: np.ndarray, task: str) -> float:
        loss, _ = self.loss_and_gradients(x, y, task)
        return loss

    def predict(self, x: np.ndarray, task: str) -> np.ndarray:
        probabilities, _ = self.forward(x, task)
        return (probabilities.ravel() >= 0.5).astype(np.int64)

    def accuracy(self, x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> dict[str, float]:
        accuracy_a = float(np.mean(self.predict(x, "a") == y_a))
        accuracy_b = float(np.mean(self.predict(x, "b") == y_b))
        return {
            "task_a": accuracy_a,
            "task_b": accuracy_b,
            "mean": float((accuracy_a + accuracy_b) / 2.0),
        }


def flatten_shared(gradients: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([gradients[name].ravel() for name in SHARED_PARAMETERS])


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)
