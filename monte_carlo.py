"""
Digit-focused Monte Carlo simulator.

Rather than trusting a single point-estimate transition matrix, each
simulated path draws its OWN transition matrix from the Dirichlet posterior
implied by the observed transition counts (this is the Bayesian analogue of
a parametric bootstrap). The path is then simulated forward `horizon` steps
by sampling actual digit realizations from that path's matrix.

This does two things a plain P^n matrix-power estimate can't:
  1. Propagates parameter (estimation) uncertainty into the final outcome
     distribution, not just transition-step randomness.
  2. Gives a Monte Carlo standard error on every downstream probability
     (Over/Under, Even/Odd), which the EV engine uses for a conservative
     lower-confidence-bound gate instead of trading on a noisy point estimate.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np

import config


@dataclass
class MonteCarloResult:
    horizon: int
    digit_probs: np.ndarray        # length-10 mean probability per digit
    digit_probs_se: np.ndarray     # length-10 standard error (bootstrap)
    n_paths: int

    def over_prob(self, barrier: int) -> float:
        return float(self.digit_probs[barrier + 1:].sum())

    def under_prob(self, barrier: int) -> float:
        return float(self.digit_probs[:barrier].sum())

    def even_prob(self) -> float:
        return float(self.digit_probs[0::2].sum())

    def odd_prob(self) -> float:
        return float(self.digit_probs[1::2].sum())

    def prob_se_for_mask(self, mask: np.ndarray) -> float:
        # Conservative combined SE for a subset of digits (sum of variances, assuming
        # worst-case positive correlation is avoided by summing SE^2 -> sqrt).
        return float(np.sqrt(np.sum(self.digit_probs_se[mask] ** 2)))


def _sample_categorical_batch(prob_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorized categorical sampling: prob_rows has shape (n_paths, 10), each
    row a probability distribution. Returns an (n_paths,) array of sampled
    digit indices via inverse-CDF.
    """
    cdf = np.cumsum(prob_rows, axis=1)
    cdf[:, -1] = 1.0  # guard against float drift
    r = rng.random(prob_rows.shape[0])
    # first index where cdf >= r
    return (cdf >= r[:, None]).argmax(axis=1)


def simulate_order1(
    transition_counts: np.ndarray,  # 10x10, Laplace-smoothed counts (Dirichlet concentration)
    current_digit: int,
    horizon: int,
    n_paths: int = None,
    seed: int = None,
) -> np.ndarray:
    """
    Returns an (n_paths,) array of simulated digit outcomes at t+horizon.
    """
    n_paths = n_paths or config.MC_PATHS
    rng = np.random.default_rng(seed if seed is not None else config.MC_SEED)

    # Sample one transition matrix per path from the Dirichlet posterior of each row.
    # Shape: (10 states, n_paths, 10 next-digit-probs)
    sampled_matrices = np.empty((10, n_paths, 10))
    for state in range(10):
        sampled_matrices[state] = rng.dirichlet(transition_counts[state], size=n_paths)

    current = np.full(n_paths, current_digit, dtype=int)
    path_idx = np.arange(n_paths)

    for _ in range(horizon):
        # Gather each path's row for its current digit: shape (n_paths, 10)
        rows = sampled_matrices[current, path_idx, :]
        current = _sample_categorical_batch(rows, rng)

    return current


def run_monte_carlo(
    digits_window,
    transition_counts_order1: np.ndarray,
    horizon: int,
    n_paths: int = None,
) -> MonteCarloResult:
    n_paths = n_paths or config.MC_PATHS
    current_digit = digits_window[-1]

    outcomes = simulate_order1(transition_counts_order1, current_digit, horizon, n_paths)

    counts = np.bincount(outcomes, minlength=10).astype(float)
    probs = counts / n_paths
    # Binomial standard error per digit probability (Monte Carlo sampling error)
    se = np.sqrt(probs * (1 - probs) / n_paths)

    return MonteCarloResult(horizon=horizon, digit_probs=probs, digit_probs_se=se, n_paths=n_paths)


def run_monte_carlo_all_horizons(
    digits_window,
    transition_counts_order1: np.ndarray,
) -> Dict[int, MonteCarloResult]:
    return {h: run_monte_carlo(digits_window, transition_counts_order1, h) for h in config.HORIZONS}
