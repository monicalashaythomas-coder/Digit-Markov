"""
Offline tests for the parts of the bot that don't need a live Deriv
connection: significance testing, Markov propagation, Monte Carlo, and EV
gating. Run with: python -m pytest test_core.py -v  (or just: python test_core.py)
"""
import numpy as np

import config
import ev_engine
import markov_entropy as me
import monte_carlo as mc
import probability_engine as pe
from digit_utils import DigitBuffer, extract_last_digit


def test_extract_last_digit_handles_trailing_zero():
    # The exact bug class caught in digit_ev_validator.py: naive float->str
    # would silently drop this trailing zero.
    assert extract_last_digit(123.40, pip_size=2) == 0
    assert extract_last_digit(123.4, pip_size=2) == 0
    assert extract_last_digit(99.99, pip_size=2) == 9
    assert extract_last_digit(100.0, pip_size=3) == 0


def test_digit_buffer_warmup():
    buf = DigitBuffer(maxlen=10)
    for i in range(5):
        buf.push(i % 10)
    assert not buf.is_warm(6)
    buf.push(5)
    assert buf.is_warm(6)
    assert len(buf) == 6


def test_uniform_stream_not_actionable():
    rng = np.random.default_rng(42)
    digits = list(rng.integers(0, 10, 2000))
    rep = me.run_significance_battery(digits, config)
    assert rep.is_actionable is False, "pure uniform noise should not be flagged actionable"


def test_biased_stream_flags_chi_square():
    rng = np.random.default_rng(42)
    probs = [0.05] * 9 + [1 - 0.05 * 9]
    digits = list(rng.choice(10, 2000, p=probs))
    rep = me.run_significance_battery(digits, config)
    assert rep.chi_square_significant is True
    z = me.digit_zscores(digits)
    assert z[9] > config.DIGIT_ZSCORE_THRESHOLD


def test_sticky_markov_detected_and_propagates():
    rng = np.random.default_rng(7)
    seq = [0]
    for _ in range(2000):
        if rng.random() < 0.4:
            seq.append(seq[-1])
        else:
            seq.append(int(rng.integers(0, 10)))

    rep = me.run_significance_battery(seq, config)
    assert rep.is_actionable is True

    tm = me.build_transition_matrix_order1(seq)
    stat_dist = me.stationary_distribution(tm)
    n1 = me.n_step_transition(tm, 1)[seq[-1]]
    n20 = me.n_step_transition(tm, 20)[seq[-1]]
    # short horizon should be closer to a "sticky" bump at current digit,
    # long horizon should converge toward the stationary distribution
    assert np.abs(n20 - stat_dist).sum() < np.abs(n1 - stat_dist).sum()


def test_monte_carlo_matches_deterministic_matrix_power():
    rng = np.random.default_rng(3)
    seq = [0]
    for _ in range(1500):
        if rng.random() < 0.35:
            seq.append(seq[-1])
        else:
            seq.append(int(rng.integers(0, 10)))

    counts = me.build_transition_counts_order1(seq)
    tm = me.build_transition_matrix_order1(seq)

    for horizon in (1, 2, 5):
        result = mc.run_monte_carlo(seq, counts, horizon, n_paths=30000)
        det = me.n_step_transition(tm, horizon)[seq[-1]]
        # Monte Carlo mean should track the deterministic expectation closely
        assert np.abs(result.digit_probs - det).max() < 0.03, f"horizon {horizon} MC diverged from theory"


def test_probability_engine_falls_back_to_uniform_when_not_actionable():
    rng = np.random.default_rng(11)
    digits = list(rng.integers(0, 10, 1000))
    estimates = pe.estimate_all_horizons(digits)
    for h, est in estimates.items():
        assert est.model_weight == 0.0
        assert np.allclose(est.distribution, 0.1, atol=1e-9)


def test_ev_engine_breakeven_and_gating():
    assert abs(ev_engine.breakeven_prob(1.0) - 0.5) < 1e-9   # even-money payout -> 50% breakeven
    assert ev_engine.breakeven_prob(0.5) > 0.6                # thin payout needs higher win prob

    # Positive-EV candidate should rank; negative-EV should be filtered
    mc_result_win_heavy = mc.MonteCarloResult(
        horizon=1,
        digit_probs=np.array([0.02] * 5 + [0.9] + [0.02] * 4),
        digit_probs_se=np.full(10, 0.001),
        n_paths=20000,
    )

    def payout_lookup(symbol, contract_type, horizon, barrier):
        return 0.95  # 95% payout

    candidates = ev_engine.evaluate_even_odd("1HZ10V", 1, mc_result_win_heavy, payout_lookup, significant=True)
    ranked = ev_engine.rank_candidates(candidates)
    assert len(ranked) >= 1
    assert ranked[0].edge >= config.MIN_EDGE


def test_risk_manager_gates_and_stakes():
    from risk_manager import RiskManager
    rm = RiskManager(starting_balance=1000.0)

    can, _ = rm.can_trade("1HZ10V")
    assert can is True

    candidate = ev_engine.TradeCandidate(
        symbol="1HZ10V", horizon=1, contract_type="DIGITEVEN", barrier=None,
        prob_point=0.6, prob_lcb=0.55, payout_pct=0.95,
        ev_per_unit_stake=0.0725, edge=0.06, significant=True,
    )
    stake = rm.stake_for_candidate(candidate)
    assert 0 < stake <= config.RISK.max_stake_fraction * 1000.0

    # Force consecutive losses to trigger cooldown
    for _ in range(config.RISK.max_consecutive_losses):
        rm.record_trade_open("1HZ10V")
        rm.record_trade_result("1HZ10V", stake=10.0, payout_received=0.0)

    can, reason = rm.can_trade("1HZ10V")
    assert can is False
    assert "cooldown" in reason


if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        test_extract_last_digit_handles_trailing_zero,
        test_digit_buffer_warmup,
        test_uniform_stream_not_actionable,
        test_biased_stream_flags_chi_square,
        test_sticky_markov_detected_and_propagates,
        test_monte_carlo_matches_deterministic_matrix_power,
        test_probability_engine_falls_back_to_uniform_when_not_actionable,
        test_ev_engine_breakeven_and_gating,
        test_risk_manager_gates_and_stakes,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
