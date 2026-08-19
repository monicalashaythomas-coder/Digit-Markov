# Digit Markov / Monte Carlo Bot

Live trading bot for Deriv's `DIGITOVER`/`DIGITUNDER`/`DIGITEVEN`/`DIGITODD`
contracts across multiple 1HZ volatility symbols.

## Philosophy → Architecture

| Your philosophy | Implementation |
|---|---|
| Read the market's Markov structure | `markov_entropy.py`: order-1 (full transition matrix) + order-2 (belief propagation) chains over the last 1000 digits |
| Entropy | Shannon entropy, normalized to [0,1] where 1.0 = perfectly uniform |
| Z-scores of digits | Per-digit z-score of observed vs. expected count under uniform |
| Probability for next 1/2/5 ticks | `probability_engine.py`: n-step Markov propagation (`P^n`), blended across order-1/order-2/empirical frequency |
| Digit-focused Monte Carlo | `monte_carlo.py`: **not** a simple sample-from-point-estimate simulator — each of 20,000 paths draws its *own* transition matrix from the Dirichlet posterior implied by observed transition counts, then simulates forward. This propagates parameter uncertainty (not just transition randomness) into the final distribution and gives every probability a real standard error |

## The one thing to know before running this live

Your own prior investigation (`digit_ev_validator.py`) found Deriv digit
sequences are **genuinely i.i.d. uniform with a consistent house edge** — a
300-combination EV scan found 0 exploitable edges. That result is almost
certainly still true most of the time. This bot is built around that fact,
not in spite of it:

- `markov_entropy.run_significance_battery()` runs chi-square (marginal
  uniformity), a runs test (streakiness), and lag-1 autocorrelation on every
  window. **By default at least 2 of 3 must hit p < 0.01** before the model
  is trusted at all (`config.MIN_SIGNIFICANT_TESTS`).
- Even when significant, `probability_engine._model_weight_from_significance()`
  caps how far the estimate can move from the uniform 1/10 prior
  (`config.MAX_MODEL_WEIGHT = 0.35`) — confidence scales continuously with
  how extreme the p-values are, it's never all-or-nothing.
- The EV/trade gate (`ev_engine.py`) trades on the **lower confidence bound**
  of the Monte Carlo probability, not the point estimate, so a single lucky
  window can't trigger a trade a proper backtest would call noise.
- If a window is genuinely uniform (the common case, per your validator),
  the bot correctly does **nothing** — `evaluate_and_maybe_trade` returns
  early on `rep.is_actionable == False`. A bot that always finds a trade is
  a bot that's lying to itself.

**What this bot cannot give you**: proof that any exploitable structure
exists on volatility indices right now. That's an empirical question the
significance gates + `digit_features`/`digit_trades`/`digit_trade_results`
logging exist to let you answer honestly, on live data, over time. Watch the
realized win rate against `prob_lcb` before scaling stake size up from the
conservative defaults in `config.RiskConfig`.

## Files

- `config.py` — every tunable (symbols, thresholds, risk limits)
- `digit_utils.py` — pip-size-safe last-digit extraction (the exact bug
  class caught in `digit_ev_validator.py`), rolling buffer
- `markov_entropy.py` — transition matrices, entropy, z-scores, significance tests
- `probability_engine.py` — multi-horizon blended probability estimation
- `monte_carlo.py` — Dirichlet-resampled path simulation
- `ev_engine.py` — EV calculation, lower-confidence-bound gating, candidate ranking
- `risk_manager.py` — fractional Kelly (capped), daily loss limits, cooldowns
- `deriv_client.py` — Deriv WebSocket API v3 wrapper (ticks, proposals, buy, settlement)
- `payout_cache.py` — background payout refresh (avoids per-tick API storms)
- `persistence.py` — Supabase logging (schema in file comments)
- `main.py` — orchestration loop
- `test_core.py` — offline unit tests for every statistical/decision component (no network needed)

## Setup (local)

```bash
pip install -r requirements.txt
python test_core.py          # verify the statistical core first, no network needed

export DERIV_API_TOKEN=xxx   # required for live trading
export SUPABASE_URL=xxx      # optional but strongly recommended for walk-forward validation
export SUPABASE_KEY=xxx

python main.py
```

Run the Supabase schema in `persistence.py`'s trailing comment block once in
the Supabase SQL editor before starting the bot.

## Deploying on Railway

Everything Railway needs is included:

- `Procfile` — declares this as a `worker` process (it holds a persistent
  WebSocket connection, it doesn't serve HTTP — Railway won't try to route
  traffic to it since there's no `web:` process type)
- `railway.json` — explicit start command (`python main.py`) and
  `ON_FAILURE` restart policy, since Nixpacks doesn't always guess a
  library-only project's entrypoint correctly
- `runtime.txt` — pins Python 3.11
- `requirements.txt` — numpy/scipy/websockets/supabase, all pure-pip, no
  system packages needed — Nixpacks builds this without extra config
- `.env.example` — the variable names to set, **not** real values

Steps:
1. Push this directory to a GitHub repo (or `railway up` directly from it via the CLI).
2. In the Railway project, add a new service from that repo.
3. In the service's **Variables** tab, set `DERIV_API_TOKEN`,
   `SUPABASE_URL`, `SUPABASE_KEY` (and `DERIV_APP_ID`/`LOG_LEVEL` if you
   want non-defaults) — Railway injects these as real env vars automatically,
   no code changes needed since `config.py` already reads from `os.environ`.
4. Deploy. Railway runs `python main.py` per `railway.json`; watch the
   deploy logs for `Subscribed to N symbols` to confirm it connected.
5. Because it's a `worker`, there's no public URL/port — monitor it via
   Railway's logs tab and your Supabase tables, not HTTP health checks.

One thing worth doing before your first live deploy: run `python test_core.py`
locally (or as a one-off Railway command) so you're not debugging both the
statistics *and* the deployment at the same time.

## Known gaps / next steps

- `deriv_client.py` is written against Deriv's documented WebSocket API v3
  shape but **could not be tested against the live endpoint** from this
  environment (no network egress to `derivws.com` here) — test against your
  demo account before going live with real funds.
- Order-2 Monte Carlo currently uses deterministic belief propagation
  (`probability_engine.estimate_distribution_order2`) rather than
  Dirichlet-resampled paths like order-1; if the order-2 signal turns out to
  matter a lot in your walk-forward logs, extending the Monte Carlo
  resampler to the 100-state order-2 chain is the natural next step.
- Payout cache refresh (45s) trades off freshness vs. API load — the bot
  always re-prices immediately before buying, so this only affects which
  candidates get *considered*, not what's actually paid.
