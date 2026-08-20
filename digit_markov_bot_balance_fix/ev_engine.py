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
    Ranks ALL candidates by edge, best first — NO minimum-edge / positive-EV
    filter here anymore (removed deliberately, per explicit instruction, to
    let trades through for live data collection while MIN_EDGE gets tuned).

    This means a candidate with negative edge CAN be returned as "best" if
    every candidate for this symbol/horizon batch is negative-edge. Two
    things still stand between that and money actually being spent on it:
      1. risk_manager.stake_for_candidate floors stake to 0 for non-positive
         Kelly, so execute_trade's `if stake <= 0: return` skips it.
      2. execute_trade still enforces config.MIN_EDGE against the FRESH
         payout right before buying (see main.py) — that check was left in
         place as the one real backstop and was only loosened, not removed.
    If you want candidates ranked in a way that's more useful once you're
    tuning MIN_EDGE back up from real data, consider filtering on
    `c.ev_per_unit_stake > 0` alone (drop the edge threshold only) as a
    middle ground — currently this returns everything, unfiltered.
    """
    return sorted(candidates, key=lambda c: c.edge, reverse=True)
