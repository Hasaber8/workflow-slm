from typing import List, Tuple
import numpy as np


def bootstrap_ci(
    values: List[float],
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 1234
) -> Tuple[float, float, float]:
    """Computes mean and (lower, upper) bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(arr)
    indices = rng.randint(0, n, size=(n_samples, n))
    sample_means = np.mean(arr[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(sample_means, alpha * 100))
    upper = float(np.percentile(sample_means, (1.0 - alpha) * 100))
    mean = float(np.mean(arr))
    return round(mean, 4), round(lower, 4), round(upper, 4)


def paired_bootstrap_diff(
    vals_a: List[float],
    vals_b: List[float],
    n_samples: int = 2000,
    ci: float = 0.95,
    seed: int = 1234
) -> Tuple[float, float, float, bool]:
    """Computes paired difference (a - b) confidence interval and whether 0 is inside the interval (tie)."""
    assert len(vals_a) == len(vals_b), "Paired bootstrap requires identical example sets"
    diffs = np.array(vals_a, dtype=float) - np.array(vals_b, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(diffs)
    indices = rng.randint(0, n, size=(n_samples, n))
    sample_diffs = np.mean(diffs[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(sample_diffs, alpha * 100))
    upper = float(np.percentile(sample_diffs, (1.0 - alpha) * 100))
    mean_diff = float(np.mean(diffs))

    # A difference inside the 95 percent interval counting 0 counts as a tie
    is_tie = (lower <= 0.0 <= upper)
    return round(mean_diff, 4), round(lower, 4), round(upper, 4), is_tie
