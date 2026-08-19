"""
Statistical feature engineering on a digit stream (0-9).

Everything here operates on a plain list/array of digits (most recent last).
This module is deliberately self-contained and pure-function where possible
so it can be unit tested against synthetic uniform and biased streams.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Basic distributional features
# ---------------------------------------------------------------------------
def digit_frequencies(digits: List[int]) -> np.ndarray:
    """Empirical frequency (counts) of each digit 0-9."""
    counts = np.zeros(10, dtype=float)
    for d in digits:
        counts[d] += 1
    return counts


def digit_probabilities(digits: List[int]) -> np.ndarray:
    counts = digit_frequencies(digits)
    total = counts.sum()
    if total == 0:
        return np.full(10, 0.1)
    return counts / total


def digit_zscores(digits: List[int]) -> np.ndarray:
    """
    Z-score of each digit's observed count against the expected count under
    a uniform multinomial(n, p=0.1) null. Positive = "hot", negative = "cold".
    """
    n = len(digits)
    if n == 0:
        return np.zeros(10)
    counts = digit_frequencies(digits)
    expected = n * 0.1
    # Var of a binomial(n, 0.1) marginal (multinomial marginal variance)
    var = n * 0.1 * 0.9
    std = np.sqrt(var) if var > 0 else 1.0
    return (counts - expected) / std


def shannon_entropy(digits: List[int]) -> float:
    """Shannon entropy in bits, normalized so max (uniform) = log2(10) ~= 3.3219."""
    p = digit_probabilities(digits)
    p_nonzero = p[p > 0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def normalized_entropy(digits: List[int]) -> float:
    """Entropy scaled to [0, 1], where 1.0 = perfectly uniform."""
    max_h = np.log2(10)
    return shannon_entropy(digits) / max_h


# ---------------------------------------------------------------------------
# Markov chain transition matrices
# ---------------------------------------------------------------------------
def build_transition_counts_order1(digits: List[int], laplace_alpha: float = 1.0) -> np.ndarray:
    """
    10x10 Laplace-smoothed transition COUNTS (not yet row-normalized). This is
    also the natural Dirichlet concentration parameter for Monte Carlo
    resampling of the transition matrix (see monte_carlo.py) — smoothed
    counts double as a weak symmetric prior.
    """
    counts = np.full((10, 10), laplace_alpha, dtype=float)
    for a, b in zip(digits[:-1], digits[1:]):
        counts[a, b] += 1
    return counts


def build_transition_matrix_order1(digits: List[int], laplace_alpha: float = 1.0) -> np.ndarray:
    """
    10x10 row-stochastic matrix: P[i, j] = P(next digit = j | current digit = i).
    Laplace smoothing (alpha) avoids zero-probability rows on sparse data.
    """
    counts = build_transition_counts_order1(digits, laplace_alpha)
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def build_transition_matrix_order2(digits: List[int], laplace_alpha: float = 1.0) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray]:
    """
    Order-2 Markov chain: P(next | prev2, prev1). Returned as a dict keyed by
    (prev2, prev1) -> probability vector over next digit (length 10), plus the
    flat count matrix (100 x 10) for diagnostics.
    """
    counts = np.full((100, 10), laplace_alpha, dtype=float)
    for i in range(len(digits) - 2):
        a, b, c = digits[i], digits[i + 1], digits[i + 2]
        state = a * 10 + b
        counts[state, c] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    table = {(a, b): probs[a * 10 + b] for a in range(10) for b in range(10)}
    return table, counts


def stationary_distribution(transition_matrix: np.ndarray, iters: int = 500) -> np.ndarray:
    """Power-iteration estimate of the stationary distribution of a Markov chain."""
    p = np.full(10, 0.1)
    for _ in range(iters):
        p = p @ transition_matrix
    return p


def n_step_transition(transition_matrix: np.ndarray, n: int) -> np.ndarray:
    """P^n via matrix power — gives P(digit at t+n = j | digit at t = i)."""
    return np.linalg.matrix_power(transition_matrix, n)


def conditional_entropy_order1(transition_matrix: np.ndarray, stationary: np.ndarray) -> float:
    """H(X_{t+1} | X_t) — how much the Markov structure actually reduces uncertainty."""
    h = 0.0
    for i in range(10):
        row = transition_matrix[i]
        row_nonzero = row[row > 0]
        h_row = -np.sum(row_nonzero * np.log2(row_nonzero))
        h += stationary[i] * h_row
    return float(h)


# ---------------------------------------------------------------------------
# Significance tests — the "skeptic" gate
# ---------------------------------------------------------------------------
@dataclass
class SignificanceReport:
    chi_square_stat: float
    chi_square_p: float
    chi_square_significant: bool

    runs_stat: float
    runs_p: float
    runs_significant: bool

    autocorr_lag1: float
    autocorr_significant: bool

    n_significant_tests: int
    is_actionable: bool  # True if enough tests fired to trust the model over uniform


def chi_square_uniformity_test(digits: List[int], alpha: float) -> Tuple[float, float, bool]:
    counts = digit_frequencies(digits)
    n = len(digits)
    if n < 30:
        return 0.0, 1.0, False
    expected = np.full(10, n / 10.0)
    stat, p = stats.chisquare(counts, f_exp=expected)
    return float(stat), float(p), bool(p < alpha)


def runs_test(digits: List[int], alpha: float) -> Tuple[float, float, bool]:
    """
    Wald-Wolfowitz runs test on digits split at the median, testing for
    non-random sequencing (streakiness / anti-persistence) rather than just
    marginal frequency.
    """
    n = len(digits)
    if n < 30:
        return 0.0, 1.0, False
    arr = np.array(digits)
    median = np.median(arr)
    # Drop values equal to median to binarize cleanly
    binary = arr[arr != median] > median
    if len(binary) < 20:
        return 0.0, 1.0, False

    n1 = int(np.sum(binary))
    n2 = int(np.sum(~binary))
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, False

    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))
    mean_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    if var_runs <= 0:
        return 0.0, 1.0, False
    z = (runs - mean_runs) / np.sqrt(var_runs)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p), bool(p < alpha)


def autocorrelation(digits: List[int], lag: int = 1) -> float:
    n = len(digits)
    if n <= lag + 1:
        return 0.0
    arr = np.array(digits, dtype=float)
    arr = arr - arr.mean()
    num = np.sum(arr[:-lag] * arr[lag:])
    den = np.sum(arr ** 2)
    return float(num / den) if den > 0 else 0.0


def autocorr_significance(digits: List[int], lag: int = 1, alpha: float = 0.01) -> bool:
    """Approximate significance of lag-1 autocorrelation vs 0 under large-n normal approx."""
    n = len(digits)
    if n < 50:
        return False
    r = autocorrelation(digits, lag)
    se = 1.0 / np.sqrt(n)
    z = r / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return bool(p < alpha)


def run_significance_battery(digits: List[int], config) -> SignificanceReport:
    chi2_stat, chi2_p, chi2_sig = chi_square_uniformity_test(digits, config.CHI_SQUARE_ALPHA)
    runs_stat, runs_p, runs_sig = runs_test(digits, config.RUNS_TEST_ALPHA)
    ac1 = autocorrelation(digits, lag=1)
    ac_sig = autocorr_significance(digits, lag=1, alpha=config.RUNS_TEST_ALPHA)

    n_sig = sum([chi2_sig, runs_sig, ac_sig])
    return SignificanceReport(
        chi_square_stat=chi2_stat, chi_square_p=chi2_p, chi_square_significant=chi2_sig,
        runs_stat=runs_stat, runs_p=runs_p, runs_significant=runs_sig,
        autocorr_lag1=ac1, autocorr_significant=ac_sig,
        n_significant_tests=n_sig,
        is_actionable=n_sig >= config.MIN_SIGNIFICANT_TESTS,
    )
