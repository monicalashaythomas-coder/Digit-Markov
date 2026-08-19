"""
Payout percentages don't move tick-to-tick the way digit probabilities do,
but fetching a fresh `proposal` for every (symbol, contract_type, horizon,
barrier) combination on every tick would be ~200+ API calls/second across 5
symbols — that will get you rate-limited immediately. Instead we refresh the
whole grid on a slow background cadence and let the per-tick decision loop
read from cache. A fresh proposal is still fetched immediately before any
actual buy (see main.py) so execution always uses a live, accurate price.
"""
import asyncio
import logging
from typing import Dict, Optional, Tuple

import config
from deriv_client import DerivClient

log = logging.getLogger("payout_cache")

CacheKey = Tuple[str, str, int, Optional[int]]  # (symbol, contract_type, horizon, barrier)

REFRESH_INTERVAL_SECONDS = 45
PROBE_STAKE = 1.0  # nominal stake used only to price the proposal, not a real trade


class PayoutCache:
    def __init__(self, client: DerivClient, symbols):
        self.client = client
        self.symbols = symbols
        self._cache: Dict[CacheKey, float] = {}
        self._task: Optional[asyncio.Task] = None

    def lookup(self, symbol: str, contract_type: str, horizon: int, barrier: Optional[int]) -> Optional[float]:
        return self._cache.get((symbol, contract_type, horizon, barrier))

    def as_lookup_fn(self):
        return self.lookup

    async def _refresh_once(self):
        for symbol in self.symbols:
            for horizon in config.HORIZONS:
                combos = [("DIGITEVEN", None), ("DIGITODD", None)]
                for barrier in config.OVER_UNDER_BARRIERS:
                    combos.append(("DIGITOVER", barrier))
                    combos.append(("DIGITUNDER", barrier))

                for contract_type, barrier in combos:
                    try:
                        payout = await self.client.get_payout_proposal(
                            symbol, contract_type, horizon, barrier, PROBE_STAKE
                        )
                        if payout is not None:
                            self._cache[(symbol, contract_type, horizon, barrier)] = payout
                    except Exception as e:
                        log.debug("payout probe failed for %s %s h=%s b=%s: %s",
                                  symbol, contract_type, horizon, barrier, e)
                    await asyncio.sleep(0.05)  # gentle pacing, avoid bursty rate limits

    async def _loop(self):
        while True:
            try:
                await self._refresh_once()
            except Exception as e:
                log.error("payout cache refresh failed: %s", e)
            await asyncio.sleep(REFRESH_INTERVAL_SECONDS)

    def start(self):
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._task:
            self._task.cancel()
