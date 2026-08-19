"""
Supabase persistence — logs every tick's feature snapshot and every trade
decision/outcome. This is what makes walk-forward validation possible: you
can later join predicted probabilities against realized outcomes and check
whether the model's edge is real or was noise (see analysis note at bottom).

Falls back to local-only logging (prints a warning once) if Supabase env
vars aren't set, so the bot can still run without persistence configured.
"""
import json
import logging
import time
from typing import Optional

import config

log = logging.getLogger("persistence")

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None


class Persistence:
    def __init__(self):
        self.client: Optional["Client"] = None
        if create_client and config.SUPABASE_URL and config.SUPABASE_KEY:
            self.client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        else:
            log.warning("Supabase not configured — persistence disabled (logging to stdout only).")

    def _insert(self, table: str, row: dict):
        row = {**row, "ts": row.get("ts", time.time())}
        if self.client:
            try:
                self.client.table(table).insert(row).execute()
                return
            except Exception as e:
                log.error("Supabase insert failed (%s): %s", table, e)
        log.info("[%s] %s", table, json.dumps(row, default=str))

    def log_feature_snapshot(self, symbol: str, digits_window_len: int, significance, entropy_norm: float,
                              model_weight: float):
        self._insert("digit_features", {
            "symbol": symbol,
            "window_len": digits_window_len,
            "chi2_p": significance.chi_square_p,
            "runs_p": significance.runs_p,
            "autocorr_lag1": significance.autocorr_lag1,
            "n_significant_tests": significance.n_significant_tests,
            "is_actionable": significance.is_actionable,
            "entropy_normalized": entropy_norm,
            "model_weight": model_weight,
        })

    def log_trade_decision(self, candidate, stake: float, contract_id: Optional[str] = None):
        self._insert("digit_trades", {
            "symbol": candidate.symbol,
            "horizon": candidate.horizon,
            "contract_type": candidate.contract_type,
            "barrier": candidate.barrier,
            "prob_point": candidate.prob_point,
            "prob_lcb": candidate.prob_lcb,
            "payout_pct": candidate.payout_pct,
            "edge": candidate.edge,
            "ev_per_unit_stake": candidate.ev_per_unit_stake,
            "stake": stake,
            "contract_id": contract_id,
            "status": "open",
        })

    def log_trade_result(self, contract_id: str, symbol: str, stake: float, payout_received: float, won: bool):
        self._insert("digit_trade_results", {
            "contract_id": contract_id,
            "symbol": symbol,
            "stake": stake,
            "payout_received": payout_received,
            "pnl": payout_received - stake,
            "won": won,
        })


# ---------------------------------------------------------------------------
# Suggested Supabase schema (run once, manually, in the Supabase SQL editor):
#
# create table digit_features (
#   id bigint generated always as identity primary key,
#   ts double precision, symbol text, window_len int,
#   chi2_p double precision, runs_p double precision, autocorr_lag1 double precision,
#   n_significant_tests int, is_actionable boolean,
#   entropy_normalized double precision, model_weight double precision
# );
#
# create table digit_trades (
#   id bigint generated always as identity primary key,
#   ts double precision, symbol text, horizon int, contract_type text, barrier int,
#   prob_point double precision, prob_lcb double precision, payout_pct double precision,
#   edge double precision, ev_per_unit_stake double precision,
#   stake double precision, contract_id text, status text
# );
#
# create table digit_trade_results (
#   id bigint generated always as identity primary key,
#   ts double precision, contract_id text, symbol text,
#   stake double precision, payout_received double precision, pnl double precision, won boolean
# );
# ---------------------------------------------------------------------------
