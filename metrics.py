"""Error metrics for rate predictions."""


def mae(actual, predicted):
    """Mean absolute error between actual and predicted rates."""
    if not actual:
        raise ValueError("Need at least one value to compute MAE.")
    total = sum(abs(a - p) for a, p in zip(actual, predicted))
    return total / len(actual)


def r_squared(actual, predicted):
    """R-squared: 1.0 is a perfect fit, 0.0 is as good as the mean."""
    if not actual:
        raise ValueError("Need at least one value to compute R-squared.")
    mean_actual = sum(actual) / len(actual)
    total = sum((a - mean_actual) ** 2 for a in actual)
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total
