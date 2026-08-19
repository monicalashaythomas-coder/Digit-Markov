"""
Turns Monte Carlo probability estimates into a ranked list of tradeable
candidates (symbol, horizon, contract_type, barrier) with positive expected
value — using the LOWER confidence bound of the probability estimate, not
the point estimate, so a lucky noisy sample can't trigger a trade.

Contract types modeled: DIGITOVER, DIGITUNDER, DIGITEVEN, DIGITODD
(Deriv's digit contracts resolve on the last digit of the spot price at the
Nth tick of the contract — i.e. exactly the "digit at horizon N" quantity
this bot estimates).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config
from monte_carlo import MonteCarloResult


@dataclass
class TradeCandidate:
    symbol: str
    horizon: int
    contract_type: str      # DIGITOVER | DIGITUNDER | DIGITEVEN | DIGITODD
    barrier: Optional[int]  # only for OVER/UNDER
    prob_point: float       # point-estimate win probability
    prob_lcb: float         # lower-confidence-bound win probability (used for EV)
    payout_pct: float       # e.g. 0.95 = 95% return on stake if win
    ev_per_unit_stake: float
    edge: float              # prob_lcb - breakeven_prob
    significant: bool


def breakeven_prob(payout_pct: float) -> float:
    """
    Win probability at which EV = 0 for a binary contract paying `payout_pct`
    profit on a win and losing the full stake on a loss.
    breakeven: p*payout - (1-p)*1 = 0  =>  p = 1 / (1 + payout_pct)
    """
    return 1.0 / (1.0 + payout_pct)


def ev_per_unit_stake(prob_win: float, payout_pct: float) -> float:
    return prob_win * payout_pct - (1 - prob_win) * 1.0


def _lcb(prob: float, se: float, z: float = 1.645) -> float:
    """One-sided lower confidence bound (default ~95%)."""
    return max(0.0, prob - z * se)


def evaluate_over_under(
    symbol: str, horizon: int, mc: MonteCarloResult, payout_lookup, significant: bool
) -> List[TradeCandidate]:
    candidates = []
    digit_mask = np.arange(10)
    for barrier in config.OVER_UNDER_BARRIERS:
        for contract_type, mask in (
            ("DIGITOVER", digit_mask > barrier),
            ("DIGITUNDER", digit_mask < barrier),
        ):
            point = float(mc.digit_probs[mask].sum())
            se = mc.prob_se_for_mask(mask)
            lcb = _lcb(point, se)

            payout_pct = payout_lookup(symbol, contract_type, horizon, barrier)
            if payout_pct is None or payout_pct < config.MIN_PAYOUT_PCT:
                continue

            edge = lcb - breakeven_prob(payout_pct)
            ev = ev_per_unit_stake(lcb, payout_pct)

            candidates.append(TradeCandidate(
                symbol=symbol, horizon=horizon, contract_type=contract_type, barrier=barrier,
                prob_point=point, prob_lcb=lcb, payout_pct=payout_pct,
                ev_per_unit_stake=ev, edge=edge, significant=significant,
            ))
    return candidates


def evaluate_even_odd(
    symbol: str, horizon: int, mc: MonteCarloResult, payout_lookup, significant: bool
) -> List[TradeCandidate]:
    candidates = []
    digit_mask = np.arange(10)
    for contract_type, mask in (
        ("DIGITEVEN", digit_mask % 2 == 0),
        ("DIGITODD", digit_mask % 2 == 1),
    ):
        point = float(mc.digit_probs[mask].sum())
        se = mc.prob_se_for_mask(mask)
        lcb = _lcb(point, se)

        payout_pct = payout_lookup(symbol, contract_type, horizon, None)
        if payout_pct is None or payout_pct < config.MIN_PAYOUT_PCT:
            continue

        edge = lcb - breakeven_prob(payout_pct)
        ev = ev_per_unit_stake(lcb, payout_pct)

        candidates.append(TradeCandidate(
            symbol=symbol, horizon=horizon, contract_type=contract_type, barrier=None,
            prob_point=point, prob_lcb=lcb, payout_pct=payout_pct,
            ev_per_unit_stake=ev, edge=edge, significant=significant,
        ))
    return candidates


def rank_candidates(candidates: List[TradeCandidate]) -> List[TradeCandidate]:
    """
    Filter to statistically-significant, EV-positive-past-margin candidates,
    ranked best edge first. This is the single gate that decides whether the
    bot trades Over/Under or Even/Odd on a given symbol/horizon — "whichever
    has better EV signal at the time".
    """
    eligible = [
        c for c in candidates
        if c.significant and c.edge >= config.MIN_EDGE and c.ev_per_unit_stake > 0
    ]
    return sorted(eligible, key=lambda c: c.edge, reverse=True)
