"""Command line interface for the BOQ rate predictor."""

import argparse
import os
import sys

from . import metrics
from .model import load_history, predict_rate, split_rows, train_model


def _default_history():
    """Find data/history.csv, from the working directory or the repo."""
    local = os.path.join(os.getcwd(), "data", "history.csv")
    if os.path.exists(local):
        return local
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "data", "history.csv")


def _find_history(path):
    if path:
        return path
    return _default_history()


def cmd_train(args):
    """Train on history and score on a held-out test split."""
    rows = load_history(args.history)
    train_rows, test_rows = split_rows(rows, test_size=args.test_size, seed=args.seed)
    model = train_model(train_rows)
    actual = [r["rate"] for r in test_rows]
    predicted = [
        predict_rate(model, r["description"], r["unit"], r["quantity"])
        for r in test_rows
    ]
    print(f"Trained on {len(train_rows)} items, tested on {len(test_rows)} held-out items.")
    print(f"MAE: SGD {metrics.mae(actual, predicted):.2f}")
    print(f"R-squared: {metrics.r_squared(actual, predicted):.3f}")
    print(f"Mean actual rate: SGD {sum(actual) / len(actual):.2f}")
    return 0


def cmd_predict(args):
    """Train on the full history, then predict one rate."""
    history = _find_history(args.history)
    rows = load_history(history)
    model = train_model(rows)
    rate = predict_rate(model, args.description, args.unit, args.quantity)
    print(f"Predicted rate: SGD {rate:.2f} per {args.unit}")
    return 0


def build_parser():
    """Build the argument parser for the train and predict commands."""
    parser = argparse.ArgumentParser(
        prog="boq_rate_predictor",
        description="Predict BOQ item rates from past priced items.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and test on history data.")
    train.add_argument("history", help="Path to a history CSV file.")
    train.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of items held back for testing (default 0.2).",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split (default 42).",
    )
    train.set_defaults(func=cmd_train)

    predict = subparsers.add_parser("predict", help="Predict a rate for one item.")
    predict.add_argument("--description", required=True, help="BOQ item description.")
    predict.add_argument("--unit", required=True, help="Unit, e.g. m3, kg, m2.")
    predict.add_argument("--quantity", type=float, required=True, help="Quantity.")
    predict.add_argument(
        "--history",
        default=None,
        help="Path to a history CSV file (default data/history.csv).",
    )
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv=None):
    """Entry point for the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
