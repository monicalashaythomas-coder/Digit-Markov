"""
Configuration for the Digit Markov/Monte Carlo bot.

All tunables live here so main.py / risk_manager.py / ev_engine.py never hardcode
magic numbers. Values are conservative defaults for a LIVE bot — tune deliberately.
"""
import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.environ.get("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN", "")  # REQUIRED for live trading
DERIV_ACCOUNT_ID = os.environ.get("DERIV_ACCOUNT_ID", "")  # optional; auto-resolved if unset
DERIV_USE_REAL_ACCOUNT = os.environ.get("DERIV_USE_REAL", "false").lower() in ("1", "true", "yes")

# Legacy direct-connect endpoint. Kept only as a fallback for accounts that
# haven't been migrated to Deriv's new Options API yet (see deriv_client.py
# docstring) — DerivClient tries the new REST+OTP flow first and only falls
# back to this if that flow's own account-lookup step comes back 404.
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# New Options API — REST bootstrap (Get accounts -> OTP) that produces a
# short-lived, pre-authenticated WebSocket URL. See:
# https://developers.deriv.com/docs/options/websocket/
DERIV_REST_BASE = "https://api.derivws.com"

if DERIV_APP_ID == "1089":
    import logging as _logging
    _logging.getLogger("config").warning(
        "DERIV_APP_ID is unset and defaulting to the shared test app_id '1089'. This is fine for "
        "quick local testing but Deriv increasingly rate-limits/blocks it for unattended, always-on "
        "server workloads (like a Railway-hosted bot) — if you see the WS handshake itself being "
        "rejected with HTTP 401/403, register your own app_id at https://api.deriv.com/ (Apps > "
        "Register application) and set DERIV_APP_ID in Railway's Variables tab."
    )

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
SYMBOLS: List[str] = [
    "1HZ10V",
    "1HZ25V",
    "1HZ50V",
    "1HZ75V",
    "1HZ100V",
]

# Rolling window of last-digit history used to build features (per symbol)
DIGIT_WINDOW = 1000
# Minimum digits observed before a symbol is eligible to trade at all
MIN_DIGITS_FOR_WARMUP = 300

# Horizons (ticks ahead) the probability engine estimates
HORIZONS = (1, 2, 5)

# ---------------------------------------------------------------------------
# Statistical significance gating
# ---------------------------------------------------------------------------
# A symbol/window is only allowed to override the uniform (1/10) prior if it
# clears ALL of these tests. This exists because digit streams are typically
# i.i.d. uniform with a house edge (confirmed via digit_ev_validator.py on
# 1HZ10V) — the point of this bot is to detect the RARE windows where that
# breaks down, not to hallucinate edges out of noise every tick.
CHI_SQUARE_ALPHA = 0.01        # reject uniformity only at strong significance
RUNS_TEST_ALPHA = 0.01
# How many of {chi2, runs, autocorr} must independently cross their alpha
# threshold before a window is trusted over the uniform prior.
# Lowered from 2 -> 1 per explicit instruction (2026-08-20). Understand what
# this trades away: with all three tests at alpha=0.01 and roughly
# independent, P(>=2 of 3 fire on pure noise) ~ 0.03%, vs P(>=1 fires on
# pure noise) ~ 3% -- about a 100x higher false-positive rate. Each single-
# test trigger still gets a small model_weight via the corroboration term
# in probability_engine._model_weight_from_significance (n_sig/3 = 1/3), but
# with MIN_EDGE now at 0.005 and rank_candidates no longer filtering by
# edge, a lone false-positive test has a real (if modest) chance of
# producing a candidate risk_manager will actually stake on. Revisit this
# once live/demo results come in.
MIN_SIGNIFICANT_TESTS = 1

# Z-score threshold for a single digit to be considered "hot"/"cold"
DIGIT_ZSCORE_THRESHOLD = 2.58  # ~99% CI

# Blend weight: how much of the final probability estimate comes from the
# Markov/empirical model vs. the uniform prior, scaled by significance
# strength. 0 = always uniform, 1 = fully trust the model.
MAX_MODEL_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MC_PATHS = 20_000
MC_SEED = None  # set an int for reproducibility during testing

# ---------------------------------------------------------------------------
# EV / trade gating
# ---------------------------------------------------------------------------
# Minimum edge (model probability - breakeven probability implied by payout).
# NOTE: as of this change, this is NO LONGER enforced in ev_engine.rank_candidates
# (that filter was removed on purpose so actionable windows trade regardless
# of edge size, to collect live data). It's still enforced once, in
# main.execute_trade, as a final check against the FRESH payout right before
# buying — i.e. it now only protects against payout drift between ranking
# and execution, not against low-edge trades in general. Lowered from 0.02
# pending re-tightening once live results come in. Override with MIN_EDGE
# env var for fast iteration without a redeploy-edit cycle.
MIN_EDGE = float(os.environ.get("MIN_EDGE", "0.005"))
# Minimum payout ratio (matches expiryrange_compression_bot's threshold) —
# don't trade contracts Deriv is only offering thin payouts on.
MIN_PAYOUT_PCT = 0.52

OVER_UNDER_BARRIERS = [2, 3, 4, 5, 6, 7]  # candidate barrier digits to evaluate

# ---------------------------------------------------------------------------
# Risk management (LIVE from the start — keep this tight)
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    base_stake_fraction: float = 0.005      # 0.5% of balance per trade, baseline
    max_stake_fraction: float = 0.02        # hard cap per trade regardless of Kelly
    kelly_fraction_cap: float = 0.25        # fractional Kelly (quarter-Kelly)
    max_concurrent_positions: int = 3
    max_daily_loss_fraction: float = 0.08   # stop trading for the day past this
    max_consecutive_losses: int = 5         # cool-down trigger
    cooldown_minutes: int = 30
    per_symbol_daily_trade_cap: int = 40
    min_balance_floor: float = 0.0          # absolute floor; 0 disables


RISK = RiskConfig()

# ---------------------------------------------------------------------------
# Logging / persistence
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
FEATURE_LOG_EVERY_N_TICKS = 1  # log every tick's features (cheap, valuable for walk-forward eval)

# How often (in ticks, per symbol) to emit an INFO heartbeat while a symbol
# is still buffering toward MIN_DIGITS_FOR_WARMUP. Exists so "the bot looks
# silent" during the ~5 minute warmup window has a concrete answer (ticks
# ARE arriving, here's the count) instead of dead air.
WARMUP_HEARTBEAT_EVERY_N_TICKS = int(os.environ.get("WARMUP_HEARTBEAT_EVERY_N_TICKS", "50"))

# How often (in ticks, per symbol) to emit a full decision-level snapshot at
# INFO once warmed up — chi2/runs/autocorr stats vs their thresholds, model
# weight, entropy. The significance battery reruns every tick regardless
# (it's cheap); this only throttles how often it's PRINTED, since digit
# streams are uniform ~99% of the time and printing every tick would drown
# real signal in noise.
DECISION_LOG_EVERY_N_TICKS = int(os.environ.get("DECISION_LOG_EVERY_N_TICKS", "20"))
