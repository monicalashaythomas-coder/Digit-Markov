"""
Risk management. This bot goes LIVE from the start (per explicit user
choice), so this module is the primary safety net — every trade must pass
through here before execution.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from ev_engine import TradeCandidate


@dataclass
class SymbolDailyStats:
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0


@dataclass
class RiskState:
    starting_balance: float
    balance: float
    day: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    consecutive_losses: int = 0
    cooldown_until: Optional[float] = None  # epoch seconds
    open_positions: int = 0
    per_symbol: Dict[str, SymbolDailyStats] = field(default_factory=dict)
    daily_pnl: float = 0.0


class RiskManager:
    def __init__(self, starting_balance: float, cfg: config.RiskConfig = config.RISK):
        self.cfg = cfg
        self.state = RiskState(starting_balance=starting_balance, balance=starting_balance)

    # -- day rollover -------------------------------------------------
    def _maybe_roll_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.state.day:
            self.state.day = today
            self.state.daily_pnl = 0.0
            self.state.per_symbol = {}
            self.state.consecutive_losses = 0
            self.state.cooldown_until = None

    def _symbol_stats(self, symbol: str) -> SymbolDailyStats:
        if symbol not in self.state.per_symbol:
            self.state.per_symbol[symbol] = SymbolDailyStats()
        return self.state.per_symbol[symbol]

    # -- gating ---------------------------------------------------------
    def can_trade(self, symbol: str) -> (bool, str):
        self._maybe_roll_day()
        s = self.state

        if self.cfg.min_balance_floor and s.balance <= self.cfg.min_balance_floor:
            return False, "balance at or below configured floor"

        if s.daily_pnl <= -abs(self.cfg.max_daily_loss_fraction * s.starting_balance):
            return False, "daily loss limit reached"

        if s.cooldown_until and time.time() < s.cooldown_until:
            return False, f"in cooldown until {s.cooldown_until}"

        if s.open_positions >= self.cfg.max_concurrent_positions:
            return False, "max concurrent positions reached"

        sym_stats = self._symbol_stats(symbol)
        if sym_stats.trades_today >= self.cfg.per_symbol_daily_trade_cap:
            return False, f"per-symbol daily trade cap reached for {symbol}"

        return True, "ok"

    # -- staking ----------------------------------------------------------
    def stake_for_candidate(self, candidate: TradeCandidate) -> float:
        """
        Fractional Kelly stake, capped by both a hard max-fraction and a
        floor at the configured base stake. Kelly fraction for a binary bet:
            f* = (b*p - q) / b   where b = payout_pct, p = win prob, q = 1-p
        We use the LOWER-confidence-bound probability (already embedded in
        candidate.prob_lcb) so sizing is conservative by construction.
        """
        b = candidate.payout_pct
        p = candidate.prob_lcb
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0.0
        kelly = max(0.0, kelly) * self.cfg.kelly_fraction_cap

        fraction = max(self.cfg.base_stake_fraction, kelly)
        fraction = min(fraction, self.cfg.max_stake_fraction)
        return round(fraction * self.state.balance, 2)

    # -- outcome recording --------------------------------------------
    def record_trade_open(self, symbol: str):
        self.state.open_positions += 1
        self._symbol_stats(symbol).trades_today += 1

    def record_trade_result(self, symbol: str, stake: float, payout_received: float):
        """payout_received = 0 on loss, stake + profit on win."""
        self.state.open_positions = max(0, self.state.open_positions - 1)
        pnl = payout_received - stake
        self.state.balance += pnl
        self.state.daily_pnl += pnl

        sym_stats = self._symbol_stats(symbol)
        sym_stats.pnl_today += pnl

        if pnl > 0:
            sym_stats.wins_today += 1
            self.state.consecutive_losses = 0
        else:
            sym_stats.losses_today += 1
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.state.cooldown_until = time.time() + self.cfg.cooldown_minutes * 60
