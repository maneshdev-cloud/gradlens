import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import make_multitask_data
from model import MultiTaskMLP


class ModelTests(unittest.TestCase):
    def test_data_shapes(self):
        data = make_multitask_data(n_train=20, n_test=10, input_dim=6, seed=1)
        self.assertEqual(data.x_train.shape, (20, 6))
        self.assertEqual(data.y_a_test.shape, (10,))

    def test_gradient_shapes(self):
        model = MultiTaskMLP(input_dim=5, hidden_dim=4, seed=1)
        x = np.random.default_rng(1).normal(size=(7, 5))
        y = np.array([0, 1, 0, 1, 1, 0, 1])
        _, gradients = model.loss_and_gradients(x, y, "a")
        for name, parameter in model.params.items():
            self.assertEqual(gradients[name].shape, parameter.shape)

    def test_w1_gradient_against_finite_difference(self):
        rng = np.random.default_rng(3)
        model = MultiTaskMLP(input_dim=5, hidden_dim=3, seed=3)
        x = rng.normal(size=(4, 5))
        y = np.array([0, 1, 1, 0])
        _, gradients = model.loss_and_gradients(x, y, "a")
        coordinate = (1, 2)
        original = model.params["W1"][coordinate]
        epsilon = 1e-6
        model.params["W1"][coordinate] = original + epsilon
        plus = model.loss(x, y, "a")
        model.params["W1"][coordinate] = original - epsilon
        minus = model.loss(x, y, "a")
        model.params["W1"][coordinate] = original
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(numerical, gradients["W1"][coordinate], places=5)


if __name__ == "__main__":
    unittest.main()
