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
DERIV_APP_ID = os.environ.get("DERIV_APP_ID", "33yLH5BDgaA4vcRK3qwY6")
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN", "pat_dd504873355fa2fa3b84ea9765daa345944464973b330cc6d34dac25e162458a")  # REQUIRED for live trading
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ybxbbfunyddvuwibgbse.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlieGJiZnVueWRkdnV3aWJnYnNlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3OTY3MTc0NSwiZXhwIjoyMDk1MjQ3NzQ1fQ.Ubur1jpVYgyzf69NpwDHbTV4ukx_u4YPLUNF1ZHlwzY")

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
MIN_SIGNIFICANT_TESTS = 2      # how many of {chi2, runs, autocorr} must fire

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
# Minimum edge (model probability - breakeven probability implied by payout)
# required to trade at all, in absolute probability terms.
MIN_EDGE = 0.02
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
