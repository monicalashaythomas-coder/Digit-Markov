"""
Digit Markov/Monte Carlo bot — main orchestration loop.

Pipeline per incoming tick:
  1. Extract last digit (pip-size-safe) -> push into that symbol's rolling window.
  2. Once warmed up: run significance battery (chi2/runs/autocorr) -> is there
     any real structure to trade, or is this window statistically uniform?
  3. If actionable: build order-1/order-2 Markov models, run the digit-focused
     Monte Carlo (Dirichlet-resampled paths) for horizons 1/2/5.
  4. Price every Over/Under barrier and Even/Odd against cached payouts,
     compute EV using the lower-confidence-bound probability, rank candidates.
  5. If a candidate clears MIN_EDGE and passes risk_manager.can_trade(): fetch
     a FRESH proposal, buy, log, track for settlement.

Run: DERIV_API_TOKEN=... SUPABASE_URL=... SUPABASE_KEY=... python main.py
"""
import asyncio
import logging
import random
import sys

import config
import ev_engine
import markov_entropy as me
import monte_carlo as mc
import probability_engine as pe
from deriv_client import DerivClient, DerivConnectError
from digit_utils import DigitBuffer, extract_last_digit
from payout_cache import PayoutCache
from persistence import Persistence
from risk_manager import RiskManager

# Railway (and most log platforms) classify severity by which stream a line
# arrives on, not by parsing the level text inside the message. Python's
# default StreamHandler (no `stream=` arg) writes to stderr, which meant
# every log line -- INFO, DEBUG, all of it -- was showing up tagged
# "error" in Railway regardless of actual level. Routing to stdout lets
# genuine ERROR/CRITICAL lines (which we still want visible) stand out
# instead of being buried in hundreds of false-positive red lines.
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO),
                     format="%(asctime)s %(name)s %(levelname)s %(message)s",
                     stream=sys.stdout)
log = logging.getLogger("main")

# httpx logs one INFO line per HTTP request (every single Supabase write),
# which drowns out everything else at scale. Bump it to WARNING so only
# genuine problems from that library surface.
logging.getLogger("httpx").setLevel(logging.WARNING)


class DigitBot:
    def __init__(self):
        self.client = DerivClient()
        self.buffers = {sym: DigitBuffer(config.DIGIT_WINDOW) for sym in config.SYMBOLS}
        self.persistence = Persistence()
        self.risk_manager: RiskManager = None  # set after balance known
        self.payout_cache: PayoutCache = None
        self._open_contracts = {}  # contract_id -> (symbol, stake)

    async def start(self):
        await self.client.connect()
        # connect() now awaits an explicit get_balance() call internally, so
        # self.client.balance is the real account balance by the time we get
        # here — no arbitrary sleep needed.
        starting_balance = self.client.balance or 0.0
        if starting_balance <= 0:
            log.error("Account balance reads as $0 (or unavailable) after connect() — "
                       "refusing to start live trading with a broken/empty balance. "
                       "Verify DERIV_API_TOKEN and account funding before retrying.")
            raise SystemExit(1)
        self.risk_manager = RiskManager(starting_balance=starting_balance)

        self.payout_cache = PayoutCache(self.client, config.SYMBOLS)
        self.payout_cache.start()

        self.client.set_contract_settled_handler(self.on_contract_settled)

        for symbol in config.SYMBOLS:
            await self.client.subscribe_ticks(symbol, self.on_tick)
            await asyncio.sleep(0.1)

        log.info("Subscribed to %d symbols. Warmup requires %d digits before trading begins.",
                  len(config.SYMBOLS), config.MIN_DIGITS_FOR_WARMUP)

        await asyncio.Event().wait()  # run forever

    # ------------------------------------------------------------------
    async def on_tick(self, symbol: str, price: float, pip_size: int):
        digit = extract_last_digit(price, pip_size)
        buf = self.buffers[symbol]
        buf.push(digit)

        if not buf.is_warm(config.MIN_DIGITS_FOR_WARMUP):
            return

        digits = buf.as_list()

        try:
            await self.evaluate_and_maybe_trade(symbol, digits)
        except Exception:
            log.exception("Error evaluating %s", symbol)

    # ------------------------------------------------------------------
    async def evaluate_and_maybe_trade(self, symbol: str, digits: list):
        estimates = pe.estimate_all_horizons(digits)
        rep = next(iter(estimates.values())).significance  # same battery for all horizons

        self.persistence.log_feature_snapshot(
            symbol, len(digits), rep, me.normalized_entropy(digits),
            next(iter(estimates.values())).model_weight,
        )

        if not rep.is_actionable:
            return  # honest "no edge" — do nothing, this is the expected common case

        can_trade, reason = self.risk_manager.can_trade(symbol)
        if not can_trade:
            log.debug("Skipping %s: %s", symbol, reason)
            return

        transition_counts = me.build_transition_counts_order1(digits)
        all_candidates = []
        for horizon in config.HORIZONS:
            mc_result = mc.run_monte_carlo(digits, transition_counts, horizon)
            all_candidates += ev_engine.evaluate_over_under(
                symbol, horizon, mc_result, self.payout_cache.as_lookup_fn(), rep.is_actionable)
            all_candidates += ev_engine.evaluate_even_odd(
                symbol, horizon, mc_result, self.payout_cache.as_lookup_fn(), rep.is_actionable)

        ranked = ev_engine.rank_candidates(all_candidates)
        if not ranked:
            return

        best = ranked[0]
        await self.execute_trade(best)

    # ------------------------------------------------------------------
    async def execute_trade(self, candidate: ev_engine.TradeCandidate):
        stake = self.risk_manager.stake_for_candidate(candidate)
        if stake <= 0:
            return

        proposal = await self.client.get_proposal_full(
            candidate.symbol, candidate.contract_type, candidate.horizon, candidate.barrier, stake
        )
        if not proposal:
            log.debug("No live proposal available for %s %s", candidate.symbol, candidate.contract_type)
            return

        # Re-check EV against the FRESH payout before committing real money —
        # the cached payout used for ranking can drift between refresh cycles.
        fresh_payout_pct = (float(proposal["payout"]) - stake) / stake
        if fresh_payout_pct < config.MIN_PAYOUT_PCT:
            return
        fresh_edge = candidate.prob_lcb - ev_engine.breakeven_prob(fresh_payout_pct)
        if fresh_edge < config.MIN_EDGE:
            log.debug("Fresh payout invalidated edge for %s %s — skipping.",
                      candidate.symbol, candidate.contract_type)
            return

        buy = await self.client.buy_contract(proposal["id"], float(proposal["ask_price"]))
        contract_id = str(buy["contract_id"])

        self.risk_manager.record_trade_open(candidate.symbol)
        self._open_contracts[contract_id] = (candidate.symbol, stake)
        self.persistence.log_trade_decision(candidate, stake, contract_id)

        log.info("BUY %s %s h=%d barrier=%s stake=%.2f edge=%.4f payout=%.2f%%",
                  candidate.symbol, candidate.contract_type, candidate.horizon,
                  candidate.barrier, stake, candidate.edge, fresh_payout_pct * 100)

    # ------------------------------------------------------------------
    async def on_contract_settled(self, poc: dict):
        contract_id = str(poc["contract_id"])
        info = self._open_contracts.pop(contract_id, None)
        if not info:
            return
        symbol, stake = info
        payout_received = float(poc.get("payout", 0.0)) if poc.get("status") == "won" else 0.0
        won = poc.get("status") == "won"

        self.risk_manager.record_trade_result(symbol, stake, payout_received)
        self.persistence.log_trade_result(contract_id, symbol, stake, payout_received, won)

        log.info("SETTLED %s contract=%s won=%s pnl=%.2f balance=%.2f",
                  symbol, contract_id, won, payout_received - stake, self.risk_manager.state.balance)


async def main():
    # Reconnect loop with exponential backoff + jitter, run at this level (rather
    # than relying on Railway's ON_FAILURE process restart) so that:
    #  - transient network blips don't kill the whole container and burn through
    #    railway.json's restartPolicyMaxRetries budget in seconds flat, and
    #  - a *permanent* rejection (HTTP 401/403 on the WS handshake itself, e.g.
    #    bad/blocked app_id) fails fast with a clear message instead of hot-looping
    #    against Deriv's edge, which risks turning a config problem into an IP-level
    #    block.
    attempt = 0
    max_permanent_failures = 3
    permanent_failures = 0
    while True:
        attempt += 1
        bot = DigitBot()
        try:
            await bot.start()
            return  # start() only returns on a clean, deliberate shutdown
        except DerivConnectError as e:
            if e.permanent:
                permanent_failures += 1
                log.error(
                    "Deriv connection permanently rejected (attempt %d, %d/%d before giving up): %s",
                    attempt, permanent_failures, max_permanent_failures, e,
                )
                if permanent_failures >= max_permanent_failures:
                    log.error(
                        "Giving up after %d consecutive permanent connection rejections. This is a "
                        "config/network issue, not something a restart will fix — see the guidance "
                        "logged above (register your own DERIV_APP_ID, check whether this host's IP "
                        "is being blocked as datacenter/VPN traffic) before redeploying.",
                        permanent_failures,
                    )
                    raise
                delay = 30 * permanent_failures  # slow, deliberate backoff for auth-type failures
            else:
                delay = min(60, 2 ** min(attempt, 6)) + random.uniform(0, 1)
                log.warning("Deriv connection failed (attempt %d): %s — retrying in %.1fs", attempt, e, delay)
            await asyncio.sleep(delay)
        except Exception:
            delay = min(60, 2 ** min(attempt, 6)) + random.uniform(0, 1)
            log.exception("Unexpected error in bot.start() (attempt %d) — retrying in %.1fs", attempt, delay)
            await asyncio.sleep(delay)
        finally:
            await bot.client.close()


if __name__ == "__main__":
    asyncio.run(main())
