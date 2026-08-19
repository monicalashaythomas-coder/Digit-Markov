"""
Async wrapper around Deriv's WebSocket trading API.

Handles: connection bootstrap, multi-symbol tick subscription (with pip_size
per symbol), payout proposals (for real-time EV pricing), buying digit
contracts, and tracking contract settlement for risk_manager feedback.

Reference: https://developers.deriv.com/

CONNECTION BOOTSTRAP — READ THIS BEFORE TOUCHING connect()
------------------------------------------------------------------------------
Deriv has migrated accounts off the old "connect directly, then send an
`authorize` message" flow onto a new REST-bootstrapped flow ("Options API"):

    1. GET  {REST_BASE}/trading/v1/options/accounts             (Bearer token)
       -> pick the demo/real account_id you want to trade on.
    2. POST {REST_BASE}/trading/v1/options/accounts/{id}/otp      (Bearer token)
       -> returns a short-lived (120s), single-use WebSocket URL with an OTP
          already embedded as a query param.
    3. Connect to that URL directly. No `authorize` message needed — the OTP
       in the URL is the authentication. Everything else (ticks, proposal,
       buy, proposal_open_contract, ...) is still the same JSON-RPC-style
       message protocol as before, BUT NOT the same *schema* per message —
       several message types dropped/renamed fields and now reject unknown
       properties (`additionalProperties: false`). Notably:
         - `proposal` (and anything shaped like it): `symbol` was renamed to
           `underlying_symbol`.
         - `active_symbols`: `product_type` (and `landing_company_short`,
           `landing_company`, `loginid`) were removed outright; response
           `symbol`/`pip` became `underlying_symbol`/`pip_size`.
       Sending the old field names doesn't get ignored — Deriv 400s with
       `InputValidationFailed: Properties not allowed: <field>`, which is
       silent/non-fatal per-message (the bot just never gets a valid
       proposal back), so it's easy to miss in a crash-free deploy log.
       Full request/response diffs: https://developers.deriv.com/comparison/

    See: https://developers.deriv.com/docs/options/websocket/

This matters because connecting straight to the *old* endpoint
(`wss://ws.derivws.com/websockets/v3?app_id=...`) for an account that has
already been migrated gets the handshake itself rejected with HTTP 401 —
before Deriv even reads an `authorize` message — which looks identical to a
bad token or a blocked app_id/IP in the traceback, but isn't either of those.

`connect()` below tries the new REST+OTP flow first (this is what Deriv's
current docs describe as *the* way to connect, not merely "new"), and falls
back to the legacy direct-connect+authorize flow only if account resolution
under the new flow comes back 404 (i.e. this account genuinely has no
Options-API accounts yet — not migrated). If your account 401s on both, the
old app_id/IP-blocking causes from before are still worth ruling out.

Uses the `websockets` library's asyncio implementation (websockets.asyncio.client).
"""
import asyncio
import itertools
import json
import logging
import urllib.error
import urllib.request
from typing import Awaitable, Callable, Dict, Optional

from websockets.asyncio.client import connect as ws_connect, ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus

import config

log = logging.getLogger("deriv_client")

TICK_HANDLER = Callable[[str, float, int], Awaitable[None]]  # (symbol, price, pip_size)
CONTRACT_SETTLED_HANDLER = Callable[[dict], Awaitable[None]]


class DerivConnectError(Exception):
    """
    Raised when the connection to Deriv can't be established at all — either
    the REST bootstrap (account lookup / OTP issuance) or the WebSocket
    handshake itself. This is distinct from an in-band Deriv API `error`
    (e.g. a bad request), which happens *after* a successful connection.

    `permanent=True` means retrying immediately is very unlikely to help
    (e.g. HTTP 401/403) — the caller should back off hard and/or stop,
    rather than hot-looping against Deriv's edge and risking an IP-level
    block or OTP/token invalidation.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, permanent: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.permanent = permanent


class DerivClient:
    def __init__(self, api_token: str = config.DERIV_API_TOKEN, app_id: str = config.DERIV_APP_ID,
                 account_id: str = config.DERIV_ACCOUNT_ID, use_real: bool = config.DERIV_USE_REAL_ACCOUNT,
                 legacy_ws_url: str = config.DERIV_WS_URL):
        self.api_token = api_token
        self.app_id = app_id
        self.account_id = account_id or None
        self.use_real = use_real
        self.legacy_ws_url = legacy_ws_url
        self.ws: Optional[ClientConnection] = None
        self._authenticated_via_otp = False  # True once connected via the new flow (no authorize() needed)
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
    def _rest_request(self, path: str, method: str = "GET") -> dict:
        """Blocking REST call (run via run_in_executor) — mirrors the OTP bootstrap
        flow at https://developers.deriv.com/docs/options/websocket/."""
        req = urllib.request.Request(
            f"{config.DERIV_REST_BASE}{path}", method=method,
            headers={
                "Deriv-App-ID": self.app_id,
                "Authorization": f"Bearer {self.api_token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise DerivConnectError(
                f"REST {method} {path} -> HTTP {exc.code}: {body}",
                status_code=exc.code, permanent=exc.code in (401, 403),
            ) from exc
        except urllib.error.URLError as exc:
            raise DerivConnectError(f"Network error calling {path}: {exc.reason}") from exc

    def _resolve_account_id(self) -> str:
        payload = self._rest_request("/trading/v1/options/accounts")
        accounts = payload.get("data") or payload.get("accounts") or []
        if not accounts:
            raise DerivConnectError("Deriv returned no Options-API accounts for this token", permanent=True)
        wanted = "real" if self.use_real else "demo"
        for acc in accounts:
            t = str(acc.get("type") or acc.get("account_type") or "").lower()
            if t == wanted:
                return acc.get("account_id") or acc.get("id")
        first = accounts[0]
        return first.get("account_id") or first.get("id")

    def _fetch_otp_ws_url(self) -> str:
        if not self.account_id:
            self.account_id = self._resolve_account_id()
        payload = self._rest_request(
            f"/trading/v1/options/accounts/{self.account_id}/otp", method="POST")
        url = (payload.get("data") or {}).get("url")
        if not url:
            raise DerivConnectError(f"OTP response missing url: {payload}")
        return url

    async def _connect_new_options_api(self):
        """New flow: REST bootstrap (account lookup + OTP) -> connect to the
        pre-authenticated URL directly. Raises DerivConnectError on failure
        (including a 404 on account lookup, which the caller uses as the
        signal to fall back to the legacy flow — see connect())."""
        loop = asyncio.get_event_loop()
        ws_url = await loop.run_in_executor(None, self._fetch_otp_ws_url)
        safe = ws_url.split("?")[0]
        log.info("Connecting (Options API, account=%s) -> %s", self.account_id, safe)
        try:
            # The ws_demo/ws_real endpoints also require Deriv-App-ID on the
            # upgrade request itself, per the docs — the OTP in the URL
            # authenticates the *account*, this header identifies the *app*.
            self.ws = await ws_connect(
                ws_url, additional_headers={"Deriv-App-ID": self.app_id},
                ping_interval=20, ping_timeout=10,
            )
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            raise DerivConnectError(
                f"Options-API WebSocket handshake rejected: HTTP {status}",
                status_code=status, permanent=status in (401, 403),
            ) from e
        self._authenticated_via_otp = True

    async def _connect_legacy(self):
        """Old flow: connect directly, then send an `authorize` message. Only
        used as a fallback when the new flow's account lookup 404s (i.e. this
        token genuinely has no Options-API accounts — not yet migrated), or
        when there's no token at all (public/unauthenticated ticks only)."""
        log.info("Connecting (legacy direct flow) -> %s", self.legacy_ws_url.split("?")[0])
        try:
            self.ws = await ws_connect(self.legacy_ws_url, ping_interval=20, ping_timeout=10)
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            headers = dict(getattr(e.response, "headers", {}) or {})
            log.error("Legacy WS handshake rejected: HTTP %s headers=%s", status, headers)
            if status in (401, 403):
                log.error(
                    "HTTP %s on the legacy WS handshake usually means either: (1) this Deriv account has "
                    "been migrated to the new Options API and no longer accepts direct connections here at "
                    "all (the new-flow attempt above should have caught this via a 404 on account lookup — "
                    "if it didn't, the account may be in a partially-migrated state); (2) DERIV_APP_ID is "
                    "the shared test value '1089', which Deriv rate-limits/blocks for server workloads; or "
                    "(3) the connection is being rejected by IP (datacenter/VPN traffic)."
                )
            raise DerivConnectError(
                f"Legacy WebSocket handshake rejected: HTTP {status}",
                status_code=status, permanent=status in (401, 403),
            ) from e
        self._authenticated_via_otp = False

    async def connect(self):
        if not self.api_token:
            log.warning("No DERIV_API_TOKEN set — running without authorization (no live trading possible).")
            await self._connect_legacy()
            self._listener_task = asyncio.create_task(self._listen())
            return

        try:
            await self._connect_new_options_api()
        except DerivConnectError as e:
            if e.status_code == 404:
                log.warning(
                    "No Options-API accounts found for this token (404 on account lookup) — this "
                    "account doesn't appear to be migrated to the new Options API yet. Falling back "
                    "to the legacy direct-connect flow."
                )
                await self._connect_legacy()
            else:
                raise

        self._listener_task = asyncio.create_task(self._listen())

        if self._authenticated_via_otp:
            # OTP already authenticated this connection to the account — no
            # `authorize` message needed, but we still need the balance
            # fetched explicitly (see get_balance() docstring) before
            # subscribing to future push updates.
            await self.get_balance()
            await self._send({"balance": 1, "subscribe": 1})
        else:
            await self.authorize()

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
        except ConnectionClosed:
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
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
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
        await self.get_balance()
        await self._send({"balance": 1, "subscribe": 1})
        return resp["authorize"]

    async def get_balance(self) -> float:
        """
        One-shot balance fetch. Unlike the {"balance":1,"subscribe":1} call,
        this response is captured directly from its own awaited future rather
        than relying on the fire-and-forget subscribe response (which
        _dispatch() resolves straight to a pending future and never falls
        through to the `elif msg_type == "balance"` push handler — so that
        value was previously discarded and self.balance stayed at its 0.0
        default until an unrelated future push happened to update it, which
        for a bot that can't stake $0 to trigger one, was never).
        """
        resp = await self._send({"balance": 1})
        if resp.get("error"):
            raise RuntimeError(f"Balance fetch failed: {resp['error']}")
        self.balance = float(resp["balance"]["balance"])
        return self.balance

    async def get_active_symbol_pip_size(self, symbol: str) -> int:
        if symbol in self._pip_size:
            return self._pip_size[symbol]
        # Options API (new): active_symbols no longer accepts `product_type`
        # (additionalProperties: false — it 400s with "Properties not
        # allowed: product_type"). Response field names also changed:
        # `symbol` -> `underlying_symbol`, `pip` -> `pip_size`.
        # See https://developers.deriv.com/comparison/active-symbols/
        resp = await self._send({"active_symbols": "brief"})
        for s in resp.get("active_symbols", []):
            pip = s.get("pip_size")
            if pip is not None:
                # pip like "0.001" -> 3 decimal places
                decimals = len(str(pip).split(".")[-1]) if "." in str(pip) else 0
                self._pip_size[s["underlying_symbol"]] = decimals
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
            # Options API (new) renamed `symbol` -> `underlying_symbol`;
            # the old key 400s with "Properties not allowed: symbol".
            # See https://developers.deriv.com/comparison/proposal/
            "underlying_symbol": symbol,
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
            # Options API (new) renamed `symbol` -> `underlying_symbol`.
            # See https://developers.deriv.com/comparison/proposal/
            "underlying_symbol": symbol,
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
