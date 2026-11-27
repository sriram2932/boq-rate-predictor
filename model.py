"""Ridge regression model for predicting BOQ item rates.

Standard library only. I solve ridge regression with the normal
equations and Gaussian elimination, so this repo has no numpy,
pandas or scikit-learn dependency.
"""

import csv
import hashlib
import math
import random

HASH_SIZE = 128
LOG_QUANTITY_INDEX = HASH_SIZE
BIAS_INDEX = HASH_SIZE + 1
FEATURE_COUNT = HASH_SIZE + 2


def tokenize(text):
    """Split text into lowercase alphanumeric words."""
    tokens = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def hash_token(token):
    """Map a token to a bucket index, deterministically."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % HASH_SIZE


def featurize(description, unit, quantity):
    """Turn one BOQ item into a numeric feature vector.

    Word features come from the description and the unit, hashed into
    128 buckets. Two extra features carry the log of the quantity and
    a bias term.
    """
    features = [0.0] * FEATURE_COUNT
    tokens = set(tokenize(description))
    tokens.add("unit_" + unit.lower())
    for token in tokens:
        features[hash_token(token)] += 1.0
    features[LOG_QUANTITY_INDEX] = math.log1p(float(quantity)) / 5.0
    features[BIAS_INDEX] = 1.0
    return features


def _solve_linear_system(matrix, vector):
    """Solve matrix * x = vector by Gaussian elimination.

    Uses partial pivoting. Raises ValueError if the matrix is singular.
    """
    n = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix: cannot solve the system.")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        for r in range(col + 1, n):
            factor = rows[r][col] / pivot_value
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = rows[i][n]
        for j in range(i + 1, n):
            total -= rows[i][j] * solution[j]
        solution[i] = total / rows[i][i]
    return solution


class RidgeRegressor:
    """Ridge regression solved with the normal equations."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, features, targets):
        """Fit weights from feature vectors and target values."""
        n = FEATURE_COUNT
        ata = [[0.0] * n for _ in range(n)]
        atb = [0.0] * n
        for x, y in zip(features, targets):
            for i in range(n):
                xi = x[i]
                if xi == 0.0:
                    continue
                atb[i] += xi * y
                row = ata[i]
                for j in range(n):
                    row[j] += xi * x[j]
        for i in range(n):
            if i != BIAS_INDEX:
                ata[i][i] += self.alpha
        self.weights = _solve_linear_system(ata, atb)
        return self

    def predict(self, features):
        """Predict a target value for each feature vector."""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        return [sum(w * v for w, v in zip(self.weights, x)) for x in features]


def load_history(path):
    """Read a history CSV into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "item_no": row["item_no"].strip(),
                    "description": row["description"].strip(),
                    "unit": row["unit"].strip(),
                    "quantity": float(row["quantity"]),
                    "rate": float(row["rate"]),
                }
            )
    return rows


def train_model(rows, alpha=1.0):
    """Train a model on history rows. The target is the log rate."""
    features = [featurize(r["description"], r["unit"], r["quantity"]) for r in rows]
    targets = [math.log(r["rate"]) for r in rows]
    model = RidgeRegressor(alpha=alpha)
    model.fit(features, targets)
    return model


def predict_rate(model, description, unit, quantity):
    """Predict the rate in dollars for one BOQ item."""
    features = featurize(description, unit, quantity)
    log_rate = model.predict([features])[0]
    return math.exp(log_rate)


def split_rows(rows, test_size=0.2, seed=42):
    """Split rows into train and test sets, shuffled with a seed."""
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_size))
    return shuffled[:cut], shuffled[cut:]
