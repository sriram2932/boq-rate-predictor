"""Tests for the BOQ rate predictor."""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boq_rate_predictor import metrics
from boq_rate_predictor.model import (
    featurize,
    hash_token,
    predict_rate,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestFeatureHashing(unittest.TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(hash_token("concrete"), hash_token("concrete"))
        self.assertEqual(hash_token("grade"), hash_token("grade"))

    def test_hash_stays_in_range(self):
        for token in ["concrete", "steel", "formwork", "t16", "m3"]:
            bucket = hash_token(token)
            self.assertGreaterEqual(bucket, 0)
            self.assertLess(bucket, 128)

    def test_featurize_is_deterministic(self):
        first = featurize("Grade 30 concrete in columns", "m3", 120)
        second = featurize("Grade 30 concrete in columns", "m3", 120)
        self.assertEqual(first, second)

    def test_different_descriptions_give_different_features(self):
        concrete = featurize("Grade 30 concrete in columns", "m3", 120)
        steel = featurize("T16 reinforcement steel bars in beams", "kg", 5000)
        self.assertNotEqual(concrete, steel)


class TestModel(unittest.TestCase):
    def test_fit_and_predict_on_tiny_data(self):
        rows = [
            {"description": "Grade 30 concrete in columns", "unit": "m3", "quantity": 100, "rate": 235.0},
            {"description": "Grade 30 concrete in beams", "unit": "m3", "quantity": 100, "rate": 223.0},
            {"description": "Grade 25 concrete in slabs", "unit": "m3", "quantity": 100, "rate": 180.0},
            {"description": "Grade 25 concrete in foundations", "unit": "m3", "quantity": 100, "rate": 165.0},
            {"description": "T16 reinforcement steel bars in beams", "unit": "kg", "quantity": 5000, "rate": 2.75},
            {"description": "T12 reinforcement steel bars in slabs", "unit": "kg", "quantity": 5000, "rate": 3.05},
        ]
        model = train_model(rows)
        prediction = predict_rate(model, "Grade 30 concrete in columns", "m3", 100)
        self.assertGreater(prediction, 150.0)
        self.assertLess(prediction, 350.0)
        steel_prediction = predict_rate(model, "T16 reinforcement steel bars in beams", "kg", 5000)
        self.assertGreater(steel_prediction, 1.0)
        self.assertLess(steel_prediction, 6.0)


class TestMetrics(unittest.TestCase):
    def test_mae_on_known_values(self):
        self.assertAlmostEqual(metrics.mae([10.0, 20.0], [12.0, 18.0]), 2.0)
        self.assertAlmostEqual(metrics.mae([5.0, 5.0], [5.0, 5.0]), 0.0)

    def test_r_squared_perfect_fit(self):
        self.assertAlmostEqual(metrics.r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_r_squared_mean_prediction_is_zero(self):
        self.assertAlmostEqual(metrics.r_squared([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]), 0.0)


class TestCli(unittest.TestCase):
    def test_train_on_sample_data(self):
        result = subprocess.run(
            [sys.executable, "-m", "boq_rate_predictor", "train", "data/history.csv"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MAE", result.stdout)
        self.assertIn("R-squared", result.stdout)

    def test_predict_on_sample_data(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "boq_rate_predictor",
                "predict",
                "--description",
                "Grade 30 concrete in foundations",
                "--unit",
                "m3",
                "--quantity",
                "120",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Predicted rate", result.stdout)


if __name__ == "__main__":
    unittest.main()
