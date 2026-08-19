"""
Thin async wrapper around Deriv's WebSocket API v3.

Handles: authorization, multi-symbol tick subscription (with pip_size per
symbol), payout proposals (for real-time EV pricing), buying digit
contracts, and tracking contract settlement for risk_manager feedback.

Reference: https://api.deriv.com/ (api.deriv.com / developers.deriv.com)
"""
import asyncio
import itertools
import json
import logging
from typing import Awaitable, Callable, Dict, Optional

import websockets

import config

log = logging.getLogger("deriv_client")

TICK_HANDLER = Callable[[str, float, int], Awaitable[None]]  # (symbol, price, pip_size)
CONTRACT_SETTLED_HANDLER = Callable[[dict], Awaitable[None]]


class DerivClient:
    def __init__(self, api_token: str = config.DERIV_API_TOKEN, ws_url: str = config.DERIV_WS_URL):
        self.api_token = api_token
        self.ws_url = ws_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id_counter = itertools.count(1)
        self._pending: Dict[int, asyncio.Future] = {}
        self._pip_size: Dict[str, int] = {}
        self._tick_handler: Optional[TICK_HANDLER] = None
        self._contract_settled_handler: Optional[CONTRACT_SETTLED_HANDLER] = None
        self._listener_task: Optional[asyncio.Task] = None
        self.balance: float = 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10)
        self._listener_task = asyncio.create_task(self._listen())
        if self.api_token:
            await self.authorize()
        else:
            log.warning("No DERIV_API_TOKEN set — running without authorization (no live trading possible).")

    async def close(self):
        if self._listener_task:
            self._listener_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _listen(self):
        assert self.ws is not None
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                await self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            log.error("Deriv websocket connection closed unexpectedly.")

    async def _dispatch(self, msg: dict):
        req_id = msg.get("req_id")
        msg_type = msg.get("msg_type")

        if msg_type == "error" or msg.get("error"):
            log.error("Deriv API error: %s", msg.get("error"))

        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                fut.set_result(msg)
            return

        if msg_type == "tick" and self._tick_handler:
            tick = msg["tick"]
            symbol = tick["symbol"]
            price = float(tick["quote"])
            pip_size = self._pip_size.get(symbol, 4)
            await self._tick_handler(symbol, price, pip_size)

        elif msg_type == "proposal_open_contract" and self._contract_settled_handler:
            poc = msg["proposal_open_contract"]
            if poc.get("is_sold") or poc.get("status") in ("won", "lost"):
                await self._contract_settled_handler(poc)

        elif msg_type == "balance":
            self.balance = float(msg["balance"]["balance"])

    # ------------------------------------------------------------------
    # Request/response helper
    # ------------------------------------------------------------------
    async def _send(self, payload: dict, timeout: float = 15.0) -> dict:
        req_id = next(self._req_id_counter)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self.ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"Deriv API request timed out: {payload}")

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------
    async def authorize(self):
        resp = await self._send({"authorize": self.api_token})
        if resp.get("error"):
            raise RuntimeError(f"Authorization failed: {resp['error']}")
        await self._send({"balance": 1, "subscribe": 1})
        return resp["authorize"]

    async def get_active_symbol_pip_size(self, symbol: str) -> int:
        if symbol in self._pip_size:
            return self._pip_size[symbol]
        resp = await self._send({"active_symbols": "brief", "product_type": "basic"})
        for s in resp.get("active_symbols", []):
            pip = s.get("pip")
            if pip is not None:
                # pip like "0.001" -> 3 decimal places
                decimals = len(str(pip).split(".")[-1]) if "." in str(pip) else 0
                self._pip_size[s["symbol"]] = decimals
        return self._pip_size.get(symbol, 4)

    async def subscribe_ticks(self, symbol: str, on_tick: TICK_HANDLER):
        self._tick_handler = on_tick
        await self.get_active_symbol_pip_size(symbol)
        resp = await self._send({"ticks": symbol, "subscribe": 1})
        if resp.get("error"):
            raise RuntimeError(f"Tick subscription failed for {symbol}: {resp['error']}")
        return resp

    async def get_payout_proposal(
        self, symbol: str, contract_type: str, duration_ticks: int,
        barrier: Optional[int], stake: float,
    ) -> Optional[float]:
        """
        Returns payout_pct (profit fraction on win) for a given contract, or
        None if unavailable (e.g. market closed / contract not offered).
        """
        payload = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration_ticks,
            "duration_unit": "t",
            "symbol": symbol,
        }
        if barrier is not None:
            payload["barrier"] = str(barrier)

        resp = await self._send(payload)
        if resp.get("error"):
            return None
        proposal = resp.get("proposal", {})
        payout = proposal.get("payout")
        if payout is None:
            return None
        return (float(payout) - stake) / stake  # profit fraction

    async def get_proposal_full(
        self, symbol: str, contract_type: str, duration_ticks: int,
        barrier: Optional[int], stake: float,
    ) -> Optional[dict]:
        """Full proposal (id, payout, ask_price) needed immediately before buying."""
        payload = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration_ticks,
            "duration_unit": "t",
            "symbol": symbol,
        }
        if barrier is not None:
            payload["barrier"] = str(barrier)
        resp = await self._send(payload)
        if resp.get("error"):
            return None
        return resp.get("proposal")

    async def buy_contract(self, proposal_id: str, price: float) -> dict:
        resp = await self._send({"buy": proposal_id, "price": price})
        if resp.get("error"):
            raise RuntimeError(f"Buy failed: {resp['error']}")
        buy = resp["buy"]
        # Subscribe to the resulting contract for settlement tracking
        await self._send({"proposal_open_contract": 1, "contract_id": buy["contract_id"], "subscribe": 1})
        return buy

    def set_contract_settled_handler(self, handler: CONTRACT_SETTLED_HANDLER):
        self._contract_settled_handler = handler
