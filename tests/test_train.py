import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import flatten_shared
from train import TrainConfig, project_conflicting_gradients, run_experiment


class TrainTests(unittest.TestCase):
    def test_projection_removes_shared_gradient_conflict(self):
        gradient_a = {
            "W1": np.array([[1.0, 0.0]]),
            "b1": np.array([0.0]),
            "W2_a": np.array([[2.0]]),
            "b2_a": np.array([0.0]),
            "W2_b": np.zeros((1, 1)),
            "b2_b": np.zeros(1),
        }
        gradient_b = {
            "W1": np.array([[-1.0, 0.0]]),
            "b1": np.array([0.0]),
            "W2_a": np.zeros((1, 1)),
            "b2_a": np.zeros(1),
            "W2_b": np.array([[3.0]]),
            "b2_b": np.array([0.0]),
        }
        projected_a, projected_b = project_conflicting_gradients(gradient_a, gradient_b)
        dot = np.dot(flatten_shared(projected_a), flatten_shared(projected_b))
        self.assertAlmostEqual(float(dot), 0.0, places=10)

    def test_one_epoch_experiment(self):
        config = TrainConfig(epochs=1, hidden_dim=4, n_train=32, n_test=16, input_dim=6)
        results = run_experiment(["vanilla"], config)
        accuracy = results["modes"]["vanilla"]["test_accuracy"]["mean"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
