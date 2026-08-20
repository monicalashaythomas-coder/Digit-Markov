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
        self._tick_counts = {sym: 0 for sym in config.SYMBOLS}  # drives heartbeat/snapshot throttling
        # Ticks are now processed concurrently (deriv_client._dispatch spawns
        # on_tick as a background task instead of blocking its own read loop
        # on it -- see deriv_client.py for why that blocking was a deadlock).
        # That means two ticks for the SAME symbol can now overlap: without
        # a guard, both could pass risk_manager.can_trade() before either's
        # open_positions increment lands, and both attempt to execute. This
        # lock serializes evaluation/execution per symbol while still
        # letting different symbols run fully concurrently.
        self._symbol_locks = {sym: asyncio.Lock() for sym in config.SYMBOLS}

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

        # Block here until the connection actually drops -- NOT a fresh,
        # never-set asyncio.Event(). Previously this awaited an Event()
        # nobody ever .set(), so a dead websocket left the process running
        # forever with no subscriptions and no way for main()'s reconnect
        # loop to notice (that loop only reacts to exceptions raised out of
        # start(), and nothing was raising). Now: when the client detects
        # disconnect, we wake up and raise, which main() already knows how
        # to handle with backoff.
        await self.client.disconnected.wait()
        raise DerivConnectError("Deriv websocket connection lost mid-session", permanent=False)

    # ------------------------------------------------------------------
    async def on_tick(self, symbol: str, price: float, pip_size: int):
        digit = extract_last_digit(price, pip_size)
        buf = self.buffers[symbol]
        buf.push(digit)

        self._tick_counts[symbol] += 1
        tick_n = self._tick_counts[symbol]

        if not buf.is_warm(config.MIN_DIGITS_FOR_WARMUP):
            # Proof-of-life during warmup — without this, a symbol that's
            # legitimately still buffering looks identical in the logs to one
            # whose tick subscription silently died.
            if tick_n % config.WARMUP_HEARTBEAT_EVERY_N_TICKS == 0 or len(buf) == 1:
                log.info("WARMUP %-8s %4d/%-4d digits buffered (price=%s last_digit=%d)",
                          symbol, len(buf), config.MIN_DIGITS_FOR_WARMUP, price, digit)
            return

        if len(buf) == config.MIN_DIGITS_FOR_WARMUP:
            log.info("%-8s warmup complete — beginning evaluation.", symbol)

        digits = buf.as_list()
        log_snapshot = (tick_n % config.DECISION_LOG_EVERY_N_TICKS == 0)

        lock = self._symbol_locks[symbol]
        if lock.locked():
            # A trade attempt for this symbol (e.g. awaiting a live proposal
            # or buy confirmation) is still in flight -- skip evaluating
            # this tick rather than stacking a second concurrent attempt on
            # top of it. digits keep accumulating in buf regardless; the
            # next free tick just evaluates against a slightly larger window.
            return

        async with lock:
            try:
                await self.evaluate_and_maybe_trade(symbol, digits, log_snapshot)
            except Exception:
                log.exception("Error evaluating %s", symbol)

    # ------------------------------------------------------------------
    async def evaluate_and_maybe_trade(self, symbol: str, digits: list, log_snapshot: bool = False):
        estimates = pe.estimate_all_horizons(digits)
        est0 = next(iter(estimates.values()))
        rep = est0.significance  # same battery for all horizons

        self.persistence.log_feature_snapshot(
            symbol, len(digits), rep, est0.entropy_normalized, est0.model_weight,
        )

        if log_snapshot:
            # This is the "what are the indicators saying, what are the
            # thresholds" line. Printed every DECISION_LOG_EVERY_N_TICKS
            # ticks (not every tick) so it stays readable at scale, but it
            # fires regardless of whether anything is actionable -- that's
            # the point, so "no edge" is visibly a decision, not silence.
            log.info(
                "SCAN %-8s n=%d | chi2 p=%.4f<%.2f=%s | runs p=%.4f<%.2f=%s | "
                "ac1=%+.4f sig=%s | fired=%d/%d req=%d | actionable=%s | "
                "weight=%.3f/%.2f | entropy=%.3f",
                symbol, len(digits),
                rep.chi_square_p, config.CHI_SQUARE_ALPHA, rep.chi_square_significant,
                rep.runs_p, config.RUNS_TEST_ALPHA, rep.runs_significant,
                rep.autocorr_lag1, rep.autocorr_significant,
                rep.n_significant_tests, 3, config.MIN_SIGNIFICANT_TESTS,
                rep.is_actionable,
                est0.model_weight, config.MAX_MODEL_WEIGHT, est0.entropy_normalized,
            )

        if not rep.is_actionable:
            return  # honest "no edge" — do nothing, this is the expected common case

        can_trade, reason = self.risk_manager.can_trade(symbol)
        if not can_trade:
            # rep.is_actionable is already the rare case, so a risk-gate
            # block on top of that is worth seeing at INFO, not buried at DEBUG.
            log.info("ACTIONABLE %s but blocked by risk_manager: %s", symbol, reason)
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
            log.info("%-8s actionable but 0 candidates — payout_cache likely has no live "
                      "quotes yet for this symbol/horizon set.", symbol)
            return

        # rank_candidates no longer filters by edge/EV — it just sorts. The
        # top-ranked candidate here may still have negative edge if every
        # candidate this cycle does; execute_trade (via risk_manager's Kelly
        # floor) is what actually decides whether real money goes on it.
        best = ranked[0]
        log.info("%-8s %d candidates, best by edge: %s barrier=%s h=%d edge=%.4f ev=%.4f prob_lcb=%.4f",
                  symbol, len(ranked), best.contract_type, best.barrier, best.horizon,
                  best.edge, best.ev_per_unit_stake, best.prob_lcb)
        await self.execute_trade(best)

    # ------------------------------------------------------------------
    async def execute_trade(self, candidate: ev_engine.TradeCandidate):
        stake = self.risk_manager.stake_for_candidate(candidate)
        if stake <= 0:
            log.info("%-8s %s h=%d barrier=%s: raw Kelly <= 0 (prob_lcb=%.4f vs breakeven) "
                      "-- edge is negative at the conservative bound, not staking.",
                      candidate.symbol, candidate.contract_type, candidate.horizon,
                      candidate.barrier, candidate.prob_lcb)
            return

        proposal = await self.client.get_proposal_full(
            candidate.symbol, candidate.contract_type, candidate.horizon, candidate.barrier, stake
        )
        if not proposal:
            log.info("%-8s %s: no live proposal available, skipping.",
                      candidate.symbol, candidate.contract_type)
            return

        # Re-check EV against the FRESH payout before committing real money —
        # the cached payout used for ranking can drift between refresh cycles.
        fresh_payout_pct = (float(proposal["payout"]) - stake) / stake
        if fresh_payout_pct < config.MIN_PAYOUT_PCT:
            log.info("%-8s %s: fresh payout %.2f%% below MIN_PAYOUT_PCT %.2f%%, skipping.",
                      candidate.symbol, candidate.contract_type,
                      fresh_payout_pct * 100, config.MIN_PAYOUT_PCT * 100)
            return
        fresh_edge = candidate.prob_lcb - ev_engine.breakeven_prob(fresh_payout_pct)
        if fresh_edge < config.MIN_EDGE:
            log.info("%-8s %s: fresh edge %.4f < MIN_EDGE %.4f (payout drifted since ranking) "
                      "-- skipping.", candidate.symbol, candidate.contract_type,
                      fresh_edge, config.MIN_EDGE)
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
