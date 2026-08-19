"""
Combines order-1 + order-2 Markov chains, entropy, and z-score features into
a single blended probability distribution over the digit at t+1, t+2, t+5 —
then hands that distribution to the Monte Carlo simulator for path-level
outcome probabilities (Over/Under barrier, Even/Odd).

Design principle: the model's influence on the final distribution is scaled
by how statistically significant the detected structure is (see
markov_entropy.run_significance_battery). When nothing significant is found,
weight collapses toward 0 and the engine outputs the uniform prior — i.e. it
correctly reports "no edge" rather than manufacturing one.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import markov_entropy as me
import config


@dataclass
class DigitProbabilityEstimate:
    horizon: int
    distribution: np.ndarray          # length-10, sums to 1
    model_weight: float               # 0..MAX_MODEL_WEIGHT, how much trust given to the model
    significance: me.SignificanceReport
    entropy_normalized: float


def _model_weight_from_significance(rep: me.SignificanceReport) -> float:
    """
    Map the significance battery to a continuous trust weight in
    [0, config.MAX_MODEL_WEIGHT]. More tests firing + lower p-values -> more
    weight, capped so the uniform prior is never fully abandoned (digit
    contracts have a real house edge baked into payouts; overconfidence in a
    noisy model is the single biggest way this bot loses money).
    """
    if not rep.is_actionable:
        return 0.0
    # Use -log10(p) of the strongest test as a confidence proxy, squashed to [0,1]
    min_p = max(min(rep.chi_square_p, rep.runs_p), 1e-12)
    confidence = min(1.0, -np.log10(min_p) / 6.0)  # p=1e-6 -> confidence 1.0
    # Scale further by how many tests agree (more corroboration = more trust)
    corroboration = rep.n_significant_tests / 3.0
    return float(config.MAX_MODEL_WEIGHT * confidence * corroboration)


def estimate_distribution_order1(digits: List[int], horizon: int) -> np.ndarray:
    tm = me.build_transition_matrix_order1(digits)
    tm_n = me.n_step_transition(tm, horizon)
    current_digit = digits[-1]
    return tm_n[current_digit]


def estimate_distribution_order2(digits: List[int], horizon: int) -> np.ndarray:
    """
    Order-2 chain doesn't have a clean matrix-power form across mixed state
    spaces, so we propagate by explicit simulation of the state chain
    (vectorized) for `horizon` steps starting from the true last 2 digits.
    """
    table, _ = me.build_transition_matrix_order2(digits)
    if len(digits) < 2:
        return np.full(10, 0.1)

    # Represent belief as a distribution over (prev2, prev1) states.
    state = (digits[-2], digits[-1])
    belief = np.zeros((10, 10))
    belief[state] = 1.0

    for _ in range(horizon):
        next_belief = np.zeros((10, 10))
        for a in range(10):
            for b in range(10):
                w = belief[a, b]
                if w == 0:
                    continue
                next_probs = table[(a, b)]  # P(c | a,b), length 10
                for c in range(10):
                    p = next_probs[c]
                    if p > 0:
                        next_belief[b, c] += w * p
        belief = next_belief

    return belief.sum(axis=0)  # marginal over the "next digit" position


def blend_distributions(order1: np.ndarray, order2: np.ndarray, empirical_freq: np.ndarray,
                         model_weight: float) -> np.ndarray:
    """
    Final distribution = model_weight * (avg of order1/order2/empirical model
    signal) + (1 - model_weight) * uniform prior.
    """
    uniform = np.full(10, 0.1)
    model_avg = (order1 + order2 + empirical_freq) / 3.0
    blended = model_weight * model_avg + (1 - model_weight) * uniform
    blended = np.clip(blended, 1e-6, None)
    return blended / blended.sum()


def estimate_all_horizons(digits: List[int]) -> Dict[int, DigitProbabilityEstimate]:
    """Main entry point: run the full feature pipeline for one symbol's digit window."""
    rep = me.run_significance_battery(digits, config)
    weight = _model_weight_from_significance(rep)
    entropy_norm = me.normalized_entropy(digits)
    empirical_freq = me.digit_probabilities(digits)

    results = {}
    for horizon in config.HORIZONS:
        if weight == 0.0 or len(digits) < config.MIN_DIGITS_FOR_WARMUP:
            dist = np.full(10, 0.1)
        else:
            o1 = estimate_distribution_order1(digits, horizon)
            o2 = estimate_distribution_order2(digits, horizon)
            dist = blend_distributions(o1, o2, empirical_freq, weight)

        results[horizon] = DigitProbabilityEstimate(
            horizon=horizon,
            distribution=dist,
            model_weight=weight,
            significance=rep,
            entropy_normalized=entropy_norm,
        )
    return results
