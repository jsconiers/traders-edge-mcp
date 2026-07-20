#!/usr/bin/env python3
"""Traders Edge MCP server (Python / FastMCP).

A consolidated 0DTE-focused options cockpit for SPX / SPXW, built on free, key-less data:

  * Chain & Greeks     - live SPX/SPXW option chain with implied vol and Greeks (CBOE delayed JSON)
  * Dealer positioning - gamma exposure (GEX), zero-gamma flip, call/put walls, max-pain,
                         0DTE pin & expected move, and dealer DEX / vanna / charm
  * Vol complex        - VIX1D / VIX9D / VIX / VIX3M / VVIX / SKEW and the term-structure regime
  * Event clock        - upcoming high-impact US macro events + live Treasury auctions

Greeks are recomputed analytically (vectorized Black-Scholes via numpy) from open interest and
implied vol, with proper Eastern-time time-to-expiry so 0DTE gamma is realistic. Data is ~15-min
delayed (fine for positioning and regime); overlay a live broker quote for execution pricing.

Data sources (provider by data type):
  Options chain & Greeks inputs    CBOE delayed quotes  (cdn.cboe.com)            no key   ~15-min delayed
  Vol indices (VIX1D..SKEW)        CBOE delayed quotes  (cdn.cboe.com)            no key   ~15-min delayed
  Live SPY / SPX overlay           Robinhood            (robin_stocks)           session  live; de-stales chain
  Live-er 0DTE chain (optional)    Robinhood options    (api.robinhood.com)      session  via _load_chain_smart
  Historical daily closes          Robinhood            (get_stock_historicals)  session
  Earnings dates                   Robinhood            (get_earnings)           session
  Dividends / ex-dividend          Robinhood            (get_fundamentals)       session
  Non-SPX equity-leg pricing       Yahoo Finance        (query1.finance...)      no key   ONLY Yahoo use (risk rollup)
  Macro series (NFCI, spreads...)  FRED / St. Louis Fed (fredgraph.csv)          no key   key only for series search
  Treasury auctions / events       TreasuryDirect       (treasurydirect.gov)     no key
  Economic calendar (tick-precise) FMP or Finnhub                                key      optional; else curated/approx
  Positions & account              Robinhood/Alpaca/E*TRADE                      varies   cross-broker aggregator

Most live/historical/fundamental data comes from the authenticated Robinhood session (robin_stocks);
CBOE is the keyless chain & vol backbone; FRED supplies macro series. Yahoo Finance is used in exactly
one place (pricing non-SPX equity legs in the risk rollup, _price_map) and could be folded into
robin_stocks get_latest_price to drop the dependency.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import csv
import io
import sys
import threading
import time
from typing import Annotated, Any, Optional
from zoneinfo import ZoneInfo

import httpx
import numpy as np
from fastmcp import FastMCP
from pydantic import Field

# ---- configuration ---------------------------------------------------------
CBOE_OPTIONS_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{root}.json"
CBOE_QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{sym}.json"
TREASURYDIRECT_UPCOMING = "https://www.treasurydirect.gov/TA_WS/securities/upcoming"

# The CBOE _SPX.json file contains BOTH AM-settled monthly SPX and PM-settled SPXW (weeklies/0DTE).
SPX_FILE_ROOT = "SPX"

VOL_INDICES = [
    ("VIX1D", "VIX1D", "1-day expected volatility (0DTE regime)"),
    ("VIX9D", "VIX9D", "9-day expected volatility"),
    ("VIX", "VIX", "30-day expected volatility (the VIX)"),
    ("VIX3M", "VIX3M", "3-month expected volatility"),
    ("VVIX", "VVIX", "vol-of-vol (volatility of VIX)"),
    ("SKEW", "SKEW", "tail-risk / skew index"),
]

RISK_FREE = float(os.environ.get("TE_RISK_FREE", "0.043"))   # annualized; gamma is ~insensitive
CONTRACT_MULT = 100                                          # index option contract multiplier
ET = ZoneInfo("America/New_York")

# ---- NYSE trading calendar (B8) --------------------------------------------
# Holidays close the market entirely; half days close at 13:00 ET. Static per-year
# (same pattern as _FOMC_2026); extend each January. 2026-07-03 is the observed
# Independence Day holiday (Jul 4 falls on Saturday), so there is no July half day.
_NYSE_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}
_NYSE_HALF_DAYS_2026 = {"2026-11-27", "2026-12-24"}   # 13:00 ET close


def _session_close_et(d: _dt.date) -> _dt.time:
    """Cash-session close for a date: 13:00 ET on half days, else 16:00 ET."""
    return _dt.time(13, 0) if d.isoformat() in _NYSE_HALF_DAYS_2026 else _dt.time(16, 0)


def _is_trading_day(d: _dt.date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _NYSE_HOLIDAYS_2026


CHAIN_TTL = 90.0          # seconds to cache the (large) CBOE chain pull (cash session open)
CHAIN_TTL_CLOSED = 1800.0 # when the session is closed the chain is static; refetch far less often
QUOTE_TTL = 60.0
REQUEST_TIMEOUT = 30.0    # per-attempt cap; _get_json retries 3x, so a hung CDN fails fast, not at 90s
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG if os.environ.get("DEBUG") == "true" else logging.INFO,
    format="%(asctime)s [traders-edge] %(levelname)s %(message)s",
)
log = logging.getLogger("traders-edge")

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, follow_redirects=True, http2=False,
            headers={"User-Agent": UA, "Accept": "application/json",
                     "Accept-Encoding": "gzip, deflate"},
        )
    return _client


class EdgeError(Exception):
    pass


class _TTLCache(dict):
    """(#7) One place for the (payload, monotonic_ts) + FIFO-bound pattern that was hand-rolled per
    cache. Callers keep their own tuple values and check the TTL at read time; put() just caps memory
    so a long-lived process can't grow its URL / series caches without bound. The single-slot broker
    caches (_rh_cache, _et_cache, _tt_cache, ...) are already bounded and intentionally don't use this."""
    def __init__(self, maxsize: int = 256):
        super().__init__()
        self._maxsize = int(maxsize)

    def put(self, key, value) -> None:
        self.pop(key, None)
        self[key] = value
        while len(self) > self._maxsize:
            oldest = next(iter(self))
            if oldest == key:
                break
            self.pop(oldest, None)


_cache: "_TTLCache" = _TTLCache(512)   # URL responses + parsed-chain + rhchain:<expiry> entries


async def _async_sleep(secs: float) -> None:
    import asyncio
    await asyncio.sleep(secs)


def _redact_url(url: str) -> str:
    """B12: strip API-key-bearing query values before a URL lands in an error message or log."""
    import re
    return re.sub(r"(?i)((?:api_?key|apikey|token|key)=)[^&]+", r"\g<1>***", url or "")


def _cache_put(cache: dict, key, value, max_entries: int = 256) -> None:
    """B15: bounded insert (FIFO eviction by insertion order) so a long-lived process can't grow
    its URL/series caches without limit."""
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > max_entries:
        oldest = next(iter(cache))
        if oldest == key:
            break
        cache.pop(oldest, None)


async def _get_json(url: str, ttl: float) -> Any:
    hit = _cache.get(url)
    if hit and (time.monotonic() - hit[1]) < ttl:
        return hit[0]
    last = None
    r = None
    for attempt in range(3):
        try:
            r = await _get_client().get(url)
            break
        except Exception as exc:  # noqa: BLE001 (transient disconnects; retry)
            last = exc
            if attempt < 2:                # B15: no pointless sleep before the final raise
                await _async_sleep(1.2)
    if r is None:
        raise EdgeError(f"Request failed: {_redact_url(url)} ({last})")
    if r.status_code == 404:
        raise EdgeError(f"Not found (404): {_redact_url(url)}")
    try:
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        raise EdgeError(f"Bad response from {_redact_url(url)}: {str(exc)[:160]}") from exc
    _cache.put(url, (data, time.monotonic()))
    return data


def _parse_occ(sym: str) -> Optional[tuple]:
    """Parse an OCC option symbol like 'SPXW260619C05500000' -> (root, 'YYYY-MM-DD', 'C'/'P', strike)."""
    s = sym.strip().upper()
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    root = s[:i]
    rest = s[i:]
    if len(rest) < 15:
        return None
    yy, mm, dd = rest[0:2], rest[2:4], rest[4:6]
    cp = rest[6]
    strike_raw = rest[7:15]
    if cp not in ("C", "P") or not (yy + mm + dd + strike_raw).isdigit():
        return None
    expiry = f"20{yy}-{mm}-{dd}"
    strike = int(strike_raw) / 1000.0
    return root, expiry, cp, strike


def _today_et() -> _dt.date:
    return _dt.datetime.now(ET).date()


def _year_frac(expiry: str) -> float:
    """Years from 'now' (ET) to the session close on the expiration date (16:00 ET; 13:00 on
    half days -- B8), floored so 0DTE gamma stays finite."""
    try:
        ed = _dt.date.fromisoformat(expiry)
    except ValueError:
        return 1.0 / 365.0
    expiry_dt = _dt.datetime.combine(ed, _session_close_et(ed), tzinfo=ET)
    now = _dt.datetime.now(ET)
    secs = (expiry_dt - now).total_seconds()
    floor = 0.5 / (365.0 * 24.0)        # ~30 minutes, avoids gamma blow-up at the bell
    return max(secs / (365.0 * 24.0 * 3600.0), floor)


mcp = FastMCP("mcp-traders-edge-server")


# ---- vectorized Black-Scholes (q=0; index options) -------------------------
def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26, vectorized (|err| < 1.5e-7)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _greeks_vec(S: float, K: np.ndarray, T: np.ndarray, sigma: np.ndarray,
                is_call: np.ndarray, r: float = RISK_FREE):
    """Return (delta, gamma, vanna, charm) per share. Inputs must be pre-masked to sigma,T,K > 0."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = _norm_pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    cdf_d1 = _norm_cdf(d1)
    delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)
    vanna = -pdf * d2 / sigma
    charm = -pdf * (2.0 * r * T - d2 * sigma * sqrtT) / (2.0 * T * sigma * sqrtT)
    return delta, gamma, vanna, charm


def _gamma_at(S: float, K: np.ndarray, T: np.ndarray, sigma: np.ndarray,
              r: float = RISK_FREE) -> np.ndarray:
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return _norm_pdf(d1) / (S * sigma * sqrtT)


# ---- chain loading / filtering ---------------------------------------------
async def _load_chain() -> dict:
    # The cash session being closed means the CBOE chain is static, so refetch far less often.
    http_ttl = CHAIN_TTL if _market_open_et() else CHAIN_TTL_CLOSED
    raw = await _get_json(CBOE_OPTIONS_URL.format(root=SPX_FILE_ROOT), http_ttl)
    d = raw.get("data", {}) or {}
    seq = d.get("seqno") or d.get("last_trade_time")
    pc = _cache.get("__parsed__")
    # Parse cache is keyed on the feed's sequence number: an unchanged seqno means the chain has not
    # moved, so there is no reason to re-parse ~30k contracts no matter how long ago we last did.
    if pc and seq is not None and pc[0] == seq:
        return pc[1]
    spot = float(d.get("current_price") or 0.0)
    opts = []
    for o in d.get("options", []) or []:
        p = _parse_occ(o.get("option", ""))
        if not p:
            continue
        root, expiry, cp, strike = p
        bid = float(o.get("bid") or 0.0)
        ask = float(o.get("ask") or 0.0)
        last = float(o.get("last_trade_price") or 0.0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
        opts.append({
            "symbol": o.get("option"), "root": root, "expiry": expiry, "cp": cp,
            "strike": strike, "bid": bid, "ask": ask, "mid": mid,
            "iv": float(o.get("iv") or 0.0), "oi": float(o.get("open_interest") or 0.0),
            "volume": float(o.get("volume") or 0.0), "last": last,
            "delta": float(o.get("delta") or 0.0), "gamma": float(o.get("gamma") or 0.0),
        })
    result = {"spot": spot, "options": opts, "asof": d.get("last_trade_time"),
              "iv30": d.get("iv30")}
    _cache["__parsed__"] = (seq, result, time.monotonic())
    return result


def _filter(options: list, root: Optional[str] = None, expiration: Optional[str] = None,
            zero_dte: bool = False) -> list:
    out = options
    if zero_dte:
        today = _today_et().isoformat()
        out = [o for o in out if o["expiry"] == today]
    elif expiration:
        out = [o for o in out if o["expiry"] == expiration]
    if root and root.upper() != "ALL":
        out = [o for o in out if o["root"] == root.upper()]
    return out


def _arrays(opts: list):
    K = np.array([o["strike"] for o in opts], dtype=float)
    isc = np.array([o["cp"] == "C" for o in opts], dtype=bool)
    oi = np.array([o["oi"] for o in opts], dtype=float)
    iv = np.array([o["iv"] for o in opts], dtype=float)
    T = np.array([_year_frac(o["expiry"]) for o in opts], dtype=float)
    return K, isc, oi, iv, T


def _valid_mask(K, oi, iv, T):
    return (K > 0) & (oi > 0) & (iv > 0) & (T > 0)


def _nearest_expiry(options: list, root: str = "SPXW") -> Optional[str]:
    today = _today_et().isoformat()
    exps = sorted({o["expiry"] for o in options
                   if (root.upper() == "ALL" or o["root"] == root.upper()) and o["expiry"] >= today})
    return exps[0] if exps else None


# ---- Robinhood SPXW chain: PRIMARY 0DTE feed (CBOE delayed = stale-flagged fallback) ----
RH_CHAIN_TTL = 60.0
_rh_chain_id_cache = {"id": None, "ts": 0.0}


def _rh_sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _rh_chain_id_sync():
    """Resolve (and cache ~1h) the Robinhood SPXW options chain_id."""
    from robin_stocks.robinhood.helper import request_get
    now = time.monotonic()
    if _rh_chain_id_cache["id"] and (now - _rh_chain_id_cache["ts"]) < 3600:
        return _rh_chain_id_cache["id"]
    probe = request_get("https://api.robinhood.com/options/instruments/", "results",
                        {"chain_symbol": "SPXW", "state": "active"}) or []
    cid = probe[0].get("chain_id") if probe else None
    if cid:
        _rh_chain_id_cache.update(id=cid, ts=now)
    return cid


def _rh_expiries_sync():
    """Sorted SPXW expiration dates from Robinhood."""
    from robin_stocks.robinhood.helper import request_get
    cid = _rh_chain_id_sync()
    if not cid:
        return []
    detail = request_get(f"https://api.robinhood.com/options/chains/{cid}/", "regular") or {}
    return sorted(detail.get("expiration_dates", []) or [])


def _rh_chain_sync(expiry: str) -> dict:
    """Pull the SPXW chain for ONE expiry from Robinhood, normalized to the _load_chain shape.
    Returns {spot, options, asof, source, freshness}. Raises EdgeError if unavailable/too thin.
    spot is parity-implied (C-P at ATM) so it tracks live during RTH; Greeks are recomputed
    downstream from iv, so only strike/cp/oi/iv/mid/delta need to be accurate here."""
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood.helper import request_get
    _rh_login_sync()
    cid = _rh_chain_id_sync()
    if not cid:
        raise EdgeError("RH SPXW chain_id unresolved")
    insts = []
    for ot in ("call", "put"):
        found = request_get("https://api.robinhood.com/options/instruments/", "pagination",
                            {"chain_id": cid, "expiration_dates": expiry,
                             "state": "active", "type": ot}) or []
        insts += found
    if not insts:
        raise EdgeError(f"RH returned no SPXW instruments for {expiry}")
    if len(insts) > 1500:
        try:
            spy = _rh_sf((rh.get_quotes(["SPY"]) or [{}])[0].get("last_trade_price")) * 10.0
        except Exception:
            spy = 0.0
        if spy > 0:
            insts = [c for c in insts if abs(_rh_sf(c.get("strike_price")) - spy) <= 800.0]
    byid = {c.get("id"): c for c in insts if c.get("id")}
    ids = list(byid)
    md_all = []
    for i in range(0, len(ids), 40):
        batch = ids[i:i + 40]
        md = request_get("https://api.robinhood.com/marketdata/options/", "results",
                         {"ids": ",".join(batch)}) or []
        md_all += [m for m in md if m]
    options = []
    for m in md_all:
        oid = (m.get("instrument") or "").rstrip("/").split("/")[-1]
        c = byid.get(oid)
        if not c:
            continue
        cp = "C" if c.get("type") == "call" else "P"
        K = _rh_sf(c.get("strike_price"))
        bid = _rh_sf(m.get("bid_price"))
        ask = _rh_sf(m.get("ask_price"))
        mark = _rh_sf(m.get("mark_price") or m.get("adjusted_mark_price"))
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else mark
        options.append({
            # B3: real OCC symbol so RH-served rows match option_quote lookups and the CBOE row
            # shape; the RH instrument UUID is preserved under rhId.
            "symbol": _occ_symbol("SPXW", expiry, cp, K), "rhId": c.get("id"),
            "root": "SPXW", "expiry": expiry, "cp": cp, "strike": K,
            "bid": bid, "ask": ask, "mid": mid, "iv": _rh_sf(m.get("implied_volatility")),
            "oi": _rh_sf(m.get("open_interest")), "volume": _rh_sf(m.get("volume")),
            "last": _rh_sf(m.get("last_trade_price")), "delta": _rh_sf(m.get("delta")),
            "gamma": _rh_sf(m.get("gamma")),
        })
    n_iv = sum(1 for o in options if o["iv"] > 0)
    n_oi = sum(1 for o in options if o["oi"] > 0)
    if min(n_iv, n_oi) < 20:
        raise EdgeError(f"RH SPXW chain too thin (iv:{n_iv} oi:{n_oi}) for {expiry}")
    strikes = sorted({o["strike"] for o in options})
    cmid = {o["strike"]: o["mid"] for o in options if o["cp"] == "C"}
    pmid = {o["strike"]: o["mid"] for o in options if o["cp"] == "P"}
    par = []
    for K in strikes:
        c, p = cmid.get(K), pmid.get(K)
        if c and p and c > 0 and p > 0:
            par.append((abs(c - p), K + (c - p)))
    par.sort()
    sp = sorted(s for _, s in par[:11])
    spot = sp[len(sp) // 2] if sp else 0.0
    open_now = _market_open_et()
    fresh = {"feed": "robinhood", "marketOpen": open_now,
             "verdict": "live" if open_now else "last-close", "stale": (not open_now),
             "paritySpot": round(spot, 2), "strikesWithIV": n_iv, "strikesWithOI": n_oi}
    asof = _dt.datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%S")
    return {"spot": spot, "options": options, "asof": asof, "iv30": None,
            "source": "robinhood_live", "freshness": fresh}


async def _rh_chain_cached(expiry: str) -> Optional[dict]:
    """Cached _rh_chain_sync for ONE expiry. Returns None (rather than raising) when RH has no
    contracts for that date, so callers can fall through to another expiry."""
    import asyncio
    hit = _cache.get(f"rhchain:{expiry}")
    if hit and (time.monotonic() - hit[1]) < RH_CHAIN_TTL:
        return hit[0]
    try:
        ch = await asyncio.to_thread(_rh_chain_sync, expiry)
    except EdgeError as exc:
        log.info("RH has no chain for %s (%s)", expiry, str(exc)[:120])
        return None
    _cache.put(f"rhchain:{expiry}", (ch, time.monotonic()))
    return ch


async def _maybe_merge_spx(rh_ch: dict, expiry: str, root: str) -> dict:
    """B10: the RH feed carries ONLY the SPXW book. For root='ALL' on a date where AM-settled SPX
    monthlies also trade (OpEx), merge the CBOE SPX rows for that expiry so GEX/walls don't
    silently lose the monthly open interest. CBOE failure degrades to RH-only with a warning."""
    if root.upper() != "ALL":
        return rh_ch
    try:
        cboe = await _load_chain()
    except EdgeError:
        rh_ch = dict(rh_ch)
        fr = dict(rh_ch.get("freshness") or {})
        fr["spxMergeWarning"] = "root=ALL requested but CBOE unavailable; SPX (AM-settled) legs missing."
        rh_ch["freshness"] = fr
        return rh_ch
    spx_rows = [o for o in (cboe.get("options") or [])
                if o.get("root") == "SPX" and o.get("expiry") == expiry]
    if spx_rows:
        rh_ch = dict(rh_ch)
        rh_ch["options"] = list(rh_ch["options"]) + spx_rows
        fr = dict(rh_ch.get("freshness") or {})
        fr["spxMergedFromCBOE"] = len(spx_rows)
        rh_ch["freshness"] = fr
    return rh_ch


async def _load_chain_smart(zero_dte: bool = False, expiration: Optional[str] = None,
                            root: str = "SPXW", prefer_rh: bool = True) -> dict:
    """Chain feed for the 0DTE tools: Robinhood live = PRIMARY, CBOE delayed = stale-flagged
    fallback. All-expiration requests always use CBOE (RH can't pull the whole multi-expiry book).
    Returns the _load_chain shape plus 'source' and 'freshness'."""
    import asyncio
    single = bool(zero_dte or expiration)
    if prefer_rh and single and root.upper() in ("SPXW", "ALL"):
        try:
            today = _today_et().isoformat()
            # NEVER gate 0DTE on _rh_expiries_sync(). RH's chain-level `expiration_dates` lists
            # openable FUTURE expiries and intermittently omits the same-day one -- verified
            # 2026-07-16: 12/12 calls returned 40 expiries starting 2026-07-17, with today
            # absent, while _rh_chain_sync(today) served 476 live contracts for that same date.
            # Gating on it silently rolled the whole 0DTE map (flip, walls, max pain, expected
            # move) onto TOMORROW's book while still reporting freshness "live". _rh_chain_sync
            # filters instruments by expiration_dates= directly, so just ask for today and only
            # go looking for another expiry if that genuinely comes back empty.
            target = expiration or (today if zero_dte else None)
            if target:
                rh_ch = await _rh_chain_cached(target)
                if rh_ch is not None:
                    return await _maybe_merge_spx(rh_ch, target, root)
            if zero_dte and not expiration:
                # Today really has no contracts (holiday / series rolled off) -- now, and only
                # now, ask RH what the next real expiry is.
                exps = await asyncio.to_thread(_rh_expiries_sync)
                nxt = next((e for e in exps if e > today), None)
                if nxt:
                    rh_ch = await _rh_chain_cached(nxt)
                    if rh_ch is not None:
                        return await _maybe_merge_spx(rh_ch, nxt, root)
        except Exception as exc:  # noqa: BLE001 -- any RH failure falls back to CBOE
            log.info("RH chain unavailable (%s); falling back to CBOE delayed", str(exc)[:140])
    ch = dict(await _load_chain())
    ch["source"] = "cboe_delayed"
    ch["freshness"] = _staleness(ch.get("asof"))
    return ch


def _feed_warnings(meta: dict) -> list:
    """Surface requested-but-unavailable position sources prominently (e.g. a stale E*TRADE OAuth
    that would otherwise silently drop legs from the net Greeks)."""
    warns = []
    for src in ("alpaca", "robinhood", "etrade", "file"):
        err = meta.get(f"{src}Error")
        if err:
            warns.append(f"{src} requested but NOT included ({err}) -- its legs are missing from these totals.")
    return warns



@mcp.tool()
async def traders_edge_status() -> dict:
    """Health check: confirms the CBOE options + vol feeds are reachable and reports current spot."""
    t0 = time.monotonic()
    out: dict = {"sources": {"options": "CBOE delayed JSON", "vol": "CBOE indices",
                             "events": "TreasuryDirect + curated"}, "note": "Data ~15 min delayed."}
    try:
        ch = await _load_chain()
        out["spot"] = round(ch["spot"], 2)
        out["contracts"] = len(ch["options"])
        out["asof"] = ch.get("asof")
        out["freshness"] = _staleness(ch.get("asof"))
        out["reachable"] = True
    except EdgeError as exc:
        out["reachable"] = False
        out["error"] = str(exc)
    out["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return out


@mcp.tool()
async def expirations(
    root: str = "SPXW",
    limit: Annotated[int, Field(ge=1, le=60)] = 20,
) -> dict:
    """List available SPX/SPXW option expirations (and days-to-expiry), soonest first.

    `root` is 'SPXW' (weeklies/0DTE, PM-settled), 'SPX' (AM-settled monthlies), or 'ALL'.
    """
    try:
        ch = await _load_chain()
    except EdgeError as exc:
        return {"error": str(exc)}
    today = _today_et()
    rootU = root.upper()
    exps = sorted({o["expiry"] for o in ch["options"]
                   if (rootU == "ALL" or o["root"] == rootU) and o["expiry"] >= today.isoformat()})
    rows = []
    for e in exps[:limit]:
        dte = (_dt.date.fromisoformat(e) - today).days
        rows.append({"expiration": e, "dte": dte, "zeroDTE": dte == 0})
    return {"root": rootU, "spot": round(ch["spot"], 2), "count": len(rows), "expirations": rows,
            "source": "cboe_delayed", "freshness": _staleness(ch.get("asof"))}


@mcp.tool()
async def options_chain(
    root: str = "SPXW",
    expiration: Optional[str] = None,
    zero_dte: bool = False,
    around_pct: Annotated[float, Field(ge=0.2, le=50.0)] = 5.0,
    max_strikes: Annotated[int, Field(ge=1, le=200)] = 40,
) -> dict:
    """SPX/SPXW option chain near the money, with implied vol and recomputed Greeks.

    Defaults to the nearest SPXW expiration; set `zero_dte=True` for today's expiry, or pass an
    explicit `expiration` (YYYY-MM-DD). `around_pct` keeps strikes within +/- that % of spot, and
    `max_strikes` caps how many strikes are returned (closest to spot). Greeks (delta, gamma) are
    recomputed analytically from each contract's IV and time-to-expiry.
    """
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte,     # B3: RH-live primary for 0DTE work
                                     expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if zero_dte else (expiration or _nearest_expiry(ch["options"], root))
    rows = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    if not rows:
        return {"error": "No contracts for that root/expiration.",
                "hint": "Try expirations() to see valid dates, or zero_dte=False."}
    lo, hi = spot * (1 - around_pct / 100.0), spot * (1 + around_pct / 100.0)
    rows = [o for o in rows if lo <= o["strike"] <= hi]
    rows.sort(key=lambda o: (abs(o["strike"] - spot), o["strike"], o["cp"]))
    rows = rows[:max_strikes]
    rows.sort(key=lambda o: (o["strike"], o["cp"]))
    K, isc, oi, iv, T = _arrays(rows)
    m = _valid_mask(K, oi * 0 + 1, iv, T)   # allow oi==0 here (display), require iv,T>0
    delta = np.full(len(rows), np.nan)
    gamma = np.full(len(rows), np.nan)
    if m.any():
        d, g, _, _ = _greeks_vec(spot, K[m], T[m], iv[m], isc[m])
        delta[m] = d
        gamma[m] = g
    out = []
    for i, o in enumerate(rows):
        out.append({
            "symbol": o["symbol"], "type": o["cp"], "strike": o["strike"],
            "bid": o["bid"], "ask": o["ask"], "mid": round(o["mid"], 2),
            "iv": round(o["iv"], 4) if o["iv"] else None,
            "delta": round(float(delta[i]), 4) if not np.isnan(delta[i]) else None,
            "gamma": round(float(gamma[i]), 6) if not np.isnan(gamma[i]) else None,
            "openInterest": int(o["oi"]), "volume": int(o["volume"]),
        })
    resolved = "today" if zero_dte else exp
    return {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
            "source": ch.get("source"), "freshness": ch.get("freshness"),
            "strikes": len(out), "chain": out}


@mcp.tool()
async def option_quote(symbol: str) -> dict:
    """Full detail for one option by OCC symbol (e.g. 'SPXW260619C05500000'): quote, IV, Greeks.

    Greeks (delta, gamma, vanna, charm) are recomputed analytically from the contract's IV and
    Eastern-time time-to-expiry.
    """
    sym = (symbol or "").strip().upper()
    parsed = _parse_occ(sym)
    if not parsed:
        return {"error": f"Could not parse OCC symbol '{symbol}'."}
    root, expiry, cp, strike = parsed
    try:
        ch = await _load_chain_smart(expiration=expiry, root=root)   # B3: RH-live for SPXW
    except EdgeError as exc:
        return {"error": str(exc)}
    o = next((x for x in ch["options"] if (x.get("symbol") or "").upper() == sym), None)
    if o is None:
        # Terms-based fallback: feed rows can differ cosmetically from the OCC string.
        o = next((x for x in ch["options"]
                  if x.get("root") == root and x.get("cp") == cp and x.get("expiry") == expiry
                  and abs((x.get("strike") or 0.0) - strike) < 1e-6), None)
    if not o:
        return {"error": f"{sym} not found in the current SPX chain."}
    spot = ch["spot"]
    T = _year_frac(expiry)
    detail = {"symbol": sym, "root": root, "type": cp, "strike": strike, "expiration": expiry,
              "dte": (_dt.date.fromisoformat(expiry) - _today_et()).days,
              "spot": round(spot, 2), "bid": o["bid"], "ask": o["ask"], "mid": round(o["mid"], 2),
              "last": o["last"], "iv": round(o["iv"], 4) if o["iv"] else None,
              "openInterest": int(o["oi"]), "volume": int(o["volume"]),
              "source": ch.get("source"), "freshness": ch.get("freshness")}
    if o["iv"] > 0 and T > 0:
        K = np.array([strike]); isc = np.array([cp == "C"])
        d, g, vn, cm = _greeks_vec(spot, K, np.array([T]), np.array([o["iv"]]), isc)
        detail["greeks"] = {"delta": round(float(d[0]), 4), "gamma": round(float(g[0]), 6),
                            "vanna": round(float(vn[0]), 4), "charm": round(float(cm[0]), 6)}
    return detail


# ---- dealer-positioning math -----------------------------------------------
# Convention: dealers long calls / short puts -> call gamma adds, put gamma subtracts.
def _gex_components(spot: float, opts: list) -> Optional[dict]:
    K, isc, oi, iv, T = _arrays(opts)
    m = _valid_mask(K, oi, iv, T)
    if not m.any():
        return None
    K, isc, oi, iv, T = K[m], isc[m], oi[m], iv[m], T[m]
    _, gamma, _, _ = _greeks_vec(spot, K, T, iv, isc)
    dollar_gamma = gamma * oi * CONTRACT_MULT * spot * spot * 0.01   # $ per 1% move (unsigned)
    signed = np.where(isc, dollar_gamma, -dollar_gamma)
    return {"K": K, "isc": isc, "oi": oi, "iv": iv, "T": T,
            "dollar_gamma": dollar_gamma, "signed": signed, "total": float(signed.sum())}


def _gamma_flip(spot: float, comp: dict) -> Optional[float]:
    K, isc, oi, iv, T = comp["K"], comp["isc"], comp["oi"], comp["iv"], comp["T"]
    levels = np.linspace(0.90 * spot, 1.10 * spot, 81)
    totals = np.empty_like(levels)
    for i, L in enumerate(levels):
        g = _gamma_at(float(L), K, T, iv)
        dg = g * oi * CONTRACT_MULT * L * L * 0.01
        totals[i] = float(np.where(isc, dg, -dg).sum())
    sign = np.sign(totals)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    best = None
    for j in idx:
        x0, x1, y0, y1 = levels[j], levels[j + 1], totals[j], totals[j + 1]
        if y1 == y0:
            continue
        xc = x0 - y0 * (x1 - x0) / (y1 - y0)
        if best is None or abs(xc - spot) < abs(best - spot):
            best = xc
    return float(best) if best is not None else None


def _per_strike(comp: dict):
    from collections import defaultdict
    call, put, net = defaultdict(float), defaultdict(float), defaultdict(float)
    for k, c, v, sg in zip(comp["K"], comp["isc"], comp["dollar_gamma"], comp["signed"]):
        (call if c else put)[float(k)] += float(v)
        net[float(k)] += float(sg)
    return call, put, net


def _max_pain(opts: list) -> Optional[float]:
    cs = [(o["strike"], o["oi"]) for o in opts if o["cp"] == "C" and o["oi"] > 0]
    ps = [(o["strike"], o["oi"]) for o in opts if o["cp"] == "P" and o["oi"] > 0]
    strikes = sorted({o["strike"] for o in opts if o["oi"] > 0})
    if not strikes or not cs or not ps:
        return None
    Kc = np.array([s for s, _ in cs]); Oc = np.array([o for _, o in cs])
    Kp = np.array([s for s, _ in ps]); Op = np.array([o for _, o in ps])
    best, bestpain = None, None
    for P in strikes:
        pain = float((np.maximum(0.0, P - Kc) * Oc).sum() + (np.maximum(0.0, Kp - P) * Op).sum())
        if bestpain is None or pain < bestpain:
            bestpain, best = pain, P
    return best


def _expected_move(spot: float, opts: list) -> Optional[dict]:
    from collections import defaultdict
    by = defaultdict(dict)
    for o in opts:
        by[o["strike"]][o["cp"]] = o["mid"]
    for k in sorted(by.keys(), key=lambda x: abs(x - spot)):
        cp = by[k]
        if cp.get("C", 0) > 0 and cp.get("P", 0) > 0:
            straddle = cp["C"] + cp["P"]
            return {"atmStrike": k, "straddle": round(straddle, 2),
                    "expectedMovePts": round(straddle, 1),
                    "expectedMovePct": round(100.0 * straddle / spot, 2)}
    return None


def _mm(x: float) -> float:
    return round(x / 1e6, 1)


@mcp.tool()
async def gamma_exposure(
    root: str = "ALL",
    expiration: Optional[str] = None,
    zero_dte: bool = False,
) -> dict:
    """Total dealer gamma exposure (GEX) for SPX, with the zero-gamma flip level and regime.

    By default aggregates the whole SPX/SPXW book (all expirations). Set `zero_dte=True` for today's
    expiry only, or pass an `expiration` (YYYY-MM-DD). Positive GEX => dealers long gamma (vol-dampening,
    mean-reverting tape); negative => short gamma (moves get amplified). The flip is the spot level where
    net dealer gamma crosses zero. Units: $ per 1% move. ~15-min delayed.
    """
    if zero_dte and expiration:   # B10: was silently discarding the expiration
        return {"error": "Pass either zero_dte=True or an explicit expiration, not both."}
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte, expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if (zero_dte or not expiration) else expiration
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    comp = _gex_components(spot, opts)
    if not comp:
        return {"error": "No valid contracts (need OI and IV) for that selection."}
    flip = _gamma_flip(spot, comp)
    total = comp["total"]
    scope = "today" if zero_dte else (exp or "all expirations")
    return {
        "root": root.upper(), "scope": scope, "spot": round(spot, 2),
        "totalGEX": round(total, 0), "totalGEX_$mm_per_1pct": _mm(total),
        "regime": "long gamma (vol-dampening)" if total > 0 else "short gamma (vol-amplifying)",
        "source": ch.get("source"), "freshness": ch.get("freshness"),
        "gammaFlip": round(flip, 2) if flip else None,
        "spotVsFlip": (None if not flip else
                       ("above flip (stabilizing)" if spot > flip else "below flip (unstable)")),
        "contractsUsed": int(len(comp["K"])), "asof": ch.get("asof"),
    }


@mcp.tool()
async def gamma_walls(
    expiration: Optional[str] = None,
    zero_dte: bool = False,
    root: str = "ALL",
    top: Annotated[int, Field(ge=1, le=15)] = 6,
) -> dict:
    """Largest gamma strikes (call wall / put wall), net-gamma strikes, and max-pain for an expiration.

    Defaults to the nearest SPXW expiration; `zero_dte=True` uses today's expiry. The call wall (biggest
    call-gamma strike above spot) often caps rallies; the put wall (biggest put-gamma strike below spot)
    often cushions selloffs. Max-pain is the strike that minimizes total option-holder payout.
    """
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte,     # B3: RH-live primary, was CBOE-only
                                     expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if zero_dte else (expiration or _nearest_expiry(ch["options"], root))
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    note = None
    resolved = "today" if zero_dte else exp
    if zero_dte and not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        resolved = exp
        note = f"No contracts expire today; showing nearest expiry {exp}."
    comp = _gex_components(spot, opts)
    if not comp:
        return {"error": "No valid contracts (need OI and IV) for that selection."}
    call, put, net = _per_strike(comp)
    call_above = {k: v for k, v in call.items() if k >= spot}
    put_below = {k: v for k, v in put.items() if k <= spot}
    call_wall = max(call_above.items(), key=lambda kv: kv[1]) if call_above else (
        max(call.items(), key=lambda kv: kv[1]) if call else None)
    put_wall = max(put_below.items(), key=lambda kv: kv[1]) if put_below else (
        max(put.items(), key=lambda kv: kv[1]) if put else None)
    top_abs = sorted(net.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    out = {
        "root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
        "asof": ch.get("asof"), "source": ch.get("source"), "freshness": ch.get("freshness"),
        "callWall": ({"strike": call_wall[0], "gamma_$mm": _mm(call_wall[1])} if call_wall else None),
        "putWall": ({"strike": put_wall[0], "gamma_$mm": _mm(put_wall[1])} if put_wall else None),
        "maxPain": _max_pain(opts),
        "topGammaStrikes": [{"strike": k, "netGamma_$mm": _mm(v)} for k, v in top_abs],
    }
    if note:
        out["note"] = note
    return out


@mcp.tool()
async def zero_dte_exposure(root: str = "SPXW") -> dict:
    """One-shot 0DTE dashboard: GEX, zero-gamma flip, call/put walls, max-pain pin, and expected move.

    Filters to today's SPXW expiration (falls back to the nearest expiry with a note if nothing expires
    today). The pin (max-pain) and the gamma walls are where price tends to get magnetized into the
    close on a positive-gamma day; the expected move is the ATM straddle (~1-sigma for the session).
    """
    try:
        ch = await _load_chain_smart(zero_dte=True, root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    opts = _filter(ch["options"], root=root, zero_dte=True)
    note = None
    resolved = "today"
    if not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        note = f"No contracts expire today; showing nearest expiry {exp}."
        resolved = exp
    comp = _gex_components(spot, opts)
    if not comp:
        return {"error": "No valid 0DTE contracts (need OI and IV)."}
    flip = _gamma_flip(spot, comp)
    call, put, net = _per_strike(comp)
    call_above = {k: v for k, v in call.items() if k >= spot}
    put_below = {k: v for k, v in put.items() if k <= spot}
    cw = max(call_above.items(), key=lambda kv: kv[1]) if call_above else None
    pw = max(put_below.items(), key=lambda kv: kv[1]) if put_below else None
    top_abs = sorted(net.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    total = comp["total"]
    out = {
        "expiration": resolved, "spot": round(spot, 2), "asof": ch.get("asof"),
        "source": ch.get("source"), "freshness": ch.get("freshness"),
        "totalGEX": round(total, 0), "totalGEX_$mm_per_1pct": _mm(total),
        "regime": "long gamma (pinning / mean-revert)" if total > 0 else "short gamma (trend / amplify)",
        "gammaFlip": round(flip, 2) if flip else None,
        "spotVsFlip": (None if not flip else ("above flip" if spot > flip else "below flip")),
        "callWall": ({"strike": cw[0], "gamma_$mm": _mm(cw[1])} if cw else None),
        "putWall": ({"strike": pw[0], "gamma_$mm": _mm(pw[1])} if pw else None),
        "maxPainPin": _max_pain(opts),
        "expectedMove": _expected_move(spot, opts),
        "gammaConcentration": [{"strike": k, "netGamma_$mm": _mm(v)} for k, v in top_abs],
        "contractsUsed": int(len(comp["K"])),
    }
    if note:
        out["note"] = note
    return out


@mcp.tool()
async def dealer_exposure(
    root: str = "ALL",
    expiration: Optional[str] = None,
    zero_dte: bool = False,
) -> dict:
    """Dealer delta (DEX), vanna, and charm exposure for SPX, under the long-calls / short-puts frame.

    DEX is the net dollar-delta dealers carry (they hedge it in futures). Vanna exposure is how that
    delta shifts per 1% change in implied vol (drives 'vanna rallies' as vol bleeds). Charm exposure is
    how delta drifts per day from time decay (the 'charm flow' into expiration). Same dealer sign
    convention as GEX: calls add, puts subtract.
    """
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte,     # B3: RH-live primary, was CBOE-only
                                     expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if (zero_dte or not expiration) else expiration
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    scope = "today" if zero_dte else (exp or "all expirations")
    if zero_dte and not opts:
        nxt = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=nxt)
        scope = f"{nxt} (nothing expires today)"
    K, isc, oi, iv, T = _arrays(opts)
    m = _valid_mask(K, oi, iv, T)
    if not m.any():
        return {"error": "No valid contracts (need OI and IV) for that selection."}
    K, isc, oi, iv, T = K[m], isc[m], oi[m], iv[m], T[m]
    delta, gamma, vanna, charm = _greeks_vec(spot, K, T, iv, isc)
    sign = np.where(isc, 1.0, -1.0)             # dealer: long calls, short puts
    notional = oi * CONTRACT_MULT
    dex = float((sign * delta * notional * spot).sum())
    vex = float((sign * vanna * notional * spot * 0.01).sum())     # per 1% vol
    chex = float((sign * charm * notional * spot / 365.0).sum())   # per day
    return {
        "root": root.upper(), "scope": scope, "spot": round(spot, 2), "asof": ch.get("asof"),
        "source": ch.get("source"), "freshness": ch.get("freshness"),
        "dealerDelta_DEX_$mm": _mm(dex),
        "dealerDelta_interpretation": ("net long delta (dealers sell futures into strength)"
                                       if dex > 0 else "net short delta (dealers buy futures into weakness)"),
        "vannaExposure_$mm_per_1pct_vol": _mm(vex),
        "vannaNote": ("falling vol -> dealers buy (supportive)" if vex > 0 else
                      "falling vol -> dealers sell (pressure)"),
        "charmExposure_$mm_per_day": _mm(chex),
        "charmNote": "delta the dealer desk must re-hedge per day from time decay (charm flow)",
        "contractsUsed": int(len(K)),
    }


# ---- vol complex -----------------------------------------------------------
async def _vol_index(sym: str) -> Optional[float]:
    try:
        data = await _get_json(CBOE_QUOTE_URL.format(sym=sym), QUOTE_TTL)
    except EdgeError:
        return None
    d = data.get("data", {}) or {}
    v = d.get("current_price")
    return float(v) if v is not None else None


def _vix_band(v: float) -> str:
    if v < 13:
        return "calm"
    if v < 20:
        return "normal"
    if v < 30:
        return "elevated"
    return "stress"


@mcp.tool()
async def vix_complex() -> dict:
    """Live CBOE volatility complex: VIX1D, VIX9D, VIX, VIX3M, VVIX, and SKEW, with a regime read.

    VIX1D is the 0DTE-relevant gauge. VVIX is vol-of-vol (option demand on VIX itself). SKEW > ~145
    signals heavy tail-hedging. ~15-min delayed.
    """
    import asyncio
    syms = [c[0] for c in VOL_INDICES]
    vals = await asyncio.gather(*[_vol_index(s) for s in syms])
    out: dict = {"asof": "CBOE delayed (~15 min)"}
    table = {}
    for (sym, _key, desc), v in zip(VOL_INDICES, vals):
        table[sym] = {"value": round(v, 2) if v is not None else None, "desc": desc}
    out["indices"] = table
    vix = table.get("VIX", {}).get("value")
    vix1d = table.get("VIX1D", {}).get("value")
    vvix = table.get("VVIX", {}).get("value")
    skew = table.get("SKEW", {}).get("value")
    if vix is not None:
        out["vixRegime"] = _vix_band(vix)
    if vix1d is not None and vix is not None and vix > 0:
        out["vix1d_vs_vix"] = round(vix1d / vix, 2)
        out["frontEnd"] = ("near-term stress bid" if vix1d > vix else "near-term calm vs 30d")
    if vvix is not None:
        out["vvixNote"] = ("elevated vol-of-vol (>100)" if vvix > 100 else "subdued vol-of-vol")
    if skew is not None:
        out["skewNote"] = ("heavy tail hedging (>145)" if skew > 145 else "moderate tail demand")
    return out


@mcp.tool()
async def vix_term_structure() -> dict:
    """VIX term structure (VIX1D -> VIX9D -> VIX -> VIX3M) and the contango/backwardation regime.

    Upward slope (contango) is the calm-market default; an inverted curve (backwardation), where the
    front end trades above the back, signals acute stress and is often a mean-reversion tell.
    """
    import asyncio
    order = [("VIX1D", 1), ("VIX9D", 9), ("VIX", 30), ("VIX3M", 90)]
    vals = await asyncio.gather(*[_vol_index(s) for s, _ in order])
    curve = []
    for (sym, dte), v in zip(order, vals):
        curve.append({"index": sym, "tenorDays": dte, "value": round(v, 2) if v is not None else None})
    v9 = next((c["value"] for c in curve if c["index"] == "VIX9D"), None)
    v3m = next((c["value"] for c in curve if c["index"] == "VIX3M"), None)
    regime, ratio = None, None
    if v9 and v3m and v3m > 0:
        ratio = round(v9 / v3m, 3)
        regime = "backwardation (stress)" if ratio > 1.0 else "contango (calm)"
    return {"asof": "CBOE delayed (~15 min)", "curve": curve,
            "vix9d_vix3m_ratio": ratio, "regime": regime}


# ---- event-schedule helpers ------------------------------------------------
_FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
              "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

# LAST-RESORT FALLBACK ONLY -- these are GUESSES, not confirmed release dates, and they HAVE
# been wrong. Reality check run 2026-07-15 against live actuals: this table had CPI(June) on
# 07-15 (actual 07-14) and PPI(June) on 07-16 (actual 07-15) -- both off by a full day, which
# produced a bad 0DTE session plan. Entries marked [v] are now verified against printed actuals;
# the rest remain unverified. _fetch_tv_calendar (keyless, live) should make this unreachable.
_CURATED_2026 = [
    ("2026-06-25", "08:30", "GDP (Q1 final)", "medium"),
    ("2026-06-26", "08:30", "PCE Price Index (May)", "high"),
    ("2026-07-14", "08:30", "CPI (June)", "high"),               # [v] corrected, was 07-15
    ("2026-07-15", "08:30", "PPI (June)", "medium"),             # [v] corrected, was 07-16
    ("2026-07-16", "08:30", "Retail Sales (June)", "high"),      # [v]
    ("2026-07-30", "08:30", "PCE Price Index (June)", "high"),   # [v] corrected, was 07-31
    ("2026-08-12", "08:30", "CPI (July)", "high"),               # [v]
    ("2026-08-13", "08:30", "PPI (July)", "medium"),             # [v] added
    ("2026-08-14", "08:30", "Retail Sales (July)", "high"),      # [v]
    ("2026-09-11", "08:30", "CPI (August)", "high"),             # UNVERIFIED
    ("2026-09-16", "08:30", "Retail Sales (August)", "high"),    # UNVERIFIED
    ("2026-10-14", "08:30", "CPI (September)", "high"),          # UNVERIFIED
    ("2026-11-13", "08:30", "CPI (October)", "high"),            # UNVERIFIED
    ("2026-12-10", "08:30", "CPI (November)", "high"),           # UNVERIFIED
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + _dt.timedelta(days=offset + 7 * (n - 1))


def _nth_business_day(year: int, month: int, n: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    cnt = 0
    while True:
        if d.weekday() < 5:
            cnt += 1
            if cnt == n:
                return d
        d += _dt.timedelta(days=1)


def _months_in_window(start: _dt.date, end: _dt.date):
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


async def _treasury_auctions(start: _dt.date, end: _dt.date) -> list:
    try:
        data = await _get_json(TREASURYDIRECT_UPCOMING, 600)
    except EdgeError:
        return []
    out = []
    for a in (data or []):
        ad = (a.get("auctionDate") or "")[:10]
        try:
            d = _dt.date.fromisoformat(ad)
        except ValueError:
            continue
        if not (start <= d <= end):
            continue
        sec = a.get("securityType", "Treasury")
        term = a.get("securityTerm", "") or ""
        out.append({"date": ad, "time": "13:00", "name": f"{term} {sec} auction".strip(),
                    "importance": "medium", "source": "TreasuryDirect (live)"})
    return out


TV_CALENDAR_URL = "https://economic-calendar.tradingview.com/events"
TV_CALENDAR_HEADERS = {"Origin": "https://www.tradingview.com",
                       "Referer": "https://www.tradingview.com/"}
_TV_IMPORTANCE = {1: "high", 0: "medium", -1: "low"}


async def _fetch_tv_calendar(start: _dt.date, end: _dt.date) -> Optional[list]:
    """Live US macro calendar from TradingView's public economic-calendar endpoint. NO KEY NEEDED.

    Unofficial/undocumented, but keyless and tick-accurate: real release dates with
    actual/forecast/previous. This is the same source the tradingview MCP uses. Any failure
    returns None so the caller falls back to FMP/Finnhub/curated.
    """
    url = (f"{TV_CALENDAR_URL}?from={start.isoformat()}T00:00:00.000Z"
           f"&to={end.isoformat()}T23:59:59.000Z&countries=US&minImportance=0")
    try:
        r = await _get_client().get(url, headers=TV_CALENDAR_HEADERS)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:  # noqa: BLE001 -- any failure falls back to the next source
        log.info("TradingView calendar unavailable (%s); falling back", str(exc)[:140])
        return None
    rows = raw.get("result") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        return None
    out = []
    for ev in rows:
        ds = ev.get("date") or ""
        try:
            when = _dt.datetime.fromisoformat(ds.replace("Z", "+00:00")).astimezone(ET)
        except Exception:  # noqa: BLE001
            continue
        name = ev.get("title") or ev.get("indicator") or ""
        if not name:
            continue
        period = ev.get("period")
        out.append({"date": when.date().isoformat(), "time": when.strftime("%H:%M"),
                    "name": (f"{name} ({period})" if period else name),
                    "importance": _TV_IMPORTANCE.get(ev.get("importance"), "medium"),
                    "source": "TradingView (live)",
                    "actual": ev.get("actual"), "forecast": ev.get("forecast"),
                    "previous": ev.get("previous")})
    return out or None


async def _fetch_live_calendar(start: _dt.date, end: _dt.date) -> Optional[list]:
    fmp = os.environ.get("FMP_API_KEY")
    fin = os.environ.get("FINNHUB_API_KEY")
    s, e = start.isoformat(), end.isoformat()
    if fmp:
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={s}&to={e}&apikey={fmp}"
        try:
            data = await _get_json(url, 300)
        except EdgeError:
            return None
        out = []
        for ev in (data or []):
            if ev.get("country") not in ("US", "USD", "United States"):
                continue
            imp = (ev.get("impact") or "").lower()
            if imp not in ("high", "medium"):
                continue
            dt = ev.get("date") or ""
            out.append({"date": dt[:10], "time": dt[11:16], "name": ev.get("event", ""),
                        "importance": imp, "source": "FMP (live)"})
        return out
    if fin:
        url = f"https://finnhub.io/api/v1/calendar/economic?from={s}&to={e}&token={fin}"
        try:
            data = await _get_json(url, 300)
        except EdgeError:
            return None
        rows = (data or {}).get("economicCalendar") or (data or {}).get("economicData") or []
        out = []
        for ev in rows:
            if ev.get("country") != "US":
                continue
            imp = (ev.get("impact") or "").lower()
            if imp not in ("high", "medium"):
                continue
            t = ev.get("time", "") or ev.get("date", "")
            out.append({"date": t[:10], "time": (t[11:16] if len(t) >= 16 else ""),
                        "name": ev.get("event", ""), "importance": imp, "source": "Finnhub (live)"})
        return out
    # No key set: TradingView's keyless endpoint is still a LIVE source. Only if that fails
    # does the caller drop to the curated guess table.
    return await _fetch_tv_calendar(start, end)


def _build_events(start: _dt.date, end: _dt.date) -> list:
    ev = []
    d = start
    while d.weekday() != 3:               # next Thursday
        d += _dt.timedelta(days=1)
    while d <= end:
        ev.append({"date": d.isoformat(), "time": "08:30", "name": "Initial Jobless Claims",
                   "importance": "medium", "source": "rule"})
        d += _dt.timedelta(days=7)
    for (y, m) in _months_in_window(start, end):
        nfp = _nth_weekday(y, m, 4, 1)
        if start <= nfp <= end:
            ev.append({"date": nfp.isoformat(), "time": "08:30",
                       "name": "Nonfarm Payrolls (Employment Situation)",
                       "importance": "high", "source": "rule (approx)"})
        for n, nm in ((1, "ISM Manufacturing PMI"), (3, "ISM Services PMI")):
            bd = _nth_business_day(y, m, n)
            if start <= bd <= end:
                ev.append({"date": bd.isoformat(), "time": "10:00", "name": nm,
                           "importance": "medium", "source": "rule (approx)"})
    for ds in _FOMC_2026:
        if start <= _dt.date.fromisoformat(ds) <= end:
            ev.append({"date": ds, "time": "14:00", "name": "FOMC rate decision",
                       "importance": "high", "source": "FOMC schedule"})
    for (ds, tm, name, imp) in _CURATED_2026:
        if start <= _dt.date.fromisoformat(ds) <= end:
            ev.append({"date": ds, "time": tm, "name": name, "importance": imp,
                       "source": "scheduled (verify)"})
    return ev


@mcp.tool()
async def economic_calendar(
    days: Annotated[int, Field(ge=1, le=60)] = 10,
    importance: str = "all",
) -> dict:
    """Upcoming high-impact US macro events + live Treasury auctions over the next `days` days.

    `importance` is 'high', 'medium', or 'all'. Uses a live API if FMP_API_KEY or FINNHUB_API_KEY is
    set, otherwise a built-in schedule: rule-based releases (jobless claims, NFP, ISM), the 2026 FOMC
    calendar, a curated macro table, and live Treasury auctions. Each event is source-tagged. Times ET.
    """
    start = _today_et()
    end = start + _dt.timedelta(days=days)
    live = await _fetch_live_calendar(start, end)
    curated_drift = None
    if live is not None:
        events = live
        src = (live[0].get("source") if live else "live")
        # (#11a) Self-verify the hand-maintained curated table against the live feed while we have it:
        # match by event name and flag any curated date that disagrees, so _build_events can be
        # corrected instead of silently drifting (you've been doing this by hand with [v] markers).
        try:
            live_by_name = {}
            for _e in live:
                live_by_name.setdefault((_e.get("name") or "").strip().lower(), _e.get("date"))
            drift = []
            for _c in _build_events(start, end):
                _lv = live_by_name.get((_c.get("name") or "").strip().lower())
                if _lv and _c.get("date") and _lv != _c["date"]:
                    drift.append({"event": _c.get("name"), "curatedDate": _c["date"], "liveDate": _lv})
            if drift:
                curated_drift = drift
                log.warning("economic_calendar: curated table drift vs live feed: %s", drift)
        except Exception:  # noqa: BLE001
            pass
    else:
        events, src = _build_events(start, end), "curated + rules (UNVERIFIED GUESSES)"
    events = events + await _treasury_auctions(start, end)
    imp = importance.lower()
    if imp in ("high", "medium"):
        wanted = {"high"} if imp == "high" else {"high", "medium"}
        events = [e for e in events if e["importance"] in wanted]
    events.sort(key=lambda e: (e["date"], e.get("time", "")))
    out = {"window": {"from": start.isoformat(), "to": end.isoformat()},
           "source": src, "count": len(events), "events": events}
    if curated_drift:
        out["curatedTableDrift"] = curated_drift
        out["curatedTableDriftNote"] = ("The hardcoded curated macro table disagrees with the live "
                                        "feed on these dates -- update _build_events / the rules table.")
    if live is None:
        out["dataQuality"] = "UNVERIFIED"
        out["warning"] = ("LIVE CALENDAR UNAVAILABLE - the TradingView endpoint failed and no "
                          "FMP_API_KEY/FINNHUB_API_KEY is set. Dates below are hardcoded estimates "
                          "that have previously been wrong by a full day. Do NOT plan a session "
                          "around them without verifying.")
    return out


@mcp.tool()
async def next_event(importance: str = "high") -> dict:
    """The single next US macro event (with an ET countdown) at or above the given importance.

    `importance` is 'high', 'medium', or 'all'. Great for a quick 'how long until the next vol event?'
    before sizing a 0DTE position.
    """
    start = _today_et()
    end = start + _dt.timedelta(days=60)
    live = await _fetch_live_calendar(start, end)
    events = live if live is not None else _build_events(start, end)
    events = events + await _treasury_auctions(start, end)
    imp = importance.lower()
    wanted = {"high"} if imp == "high" else ({"high", "medium"} if imp == "medium"
                                             else {"high", "medium", "low"})
    events = [e for e in events if e["importance"] in wanted]
    now = _dt.datetime.now(ET)
    fut = []
    for e in events:
        tm = e.get("time") or "08:30"
        try:
            hh, mm = int(tm[:2]), int(tm[3:5])
        except ValueError:
            hh, mm = 8, 30
        edt = _dt.datetime.combine(_dt.date.fromisoformat(e["date"]), _dt.time(hh, mm), tzinfo=ET)
        if edt > now:
            fut.append((edt, e))
    if not fut:
        return {"note": "No upcoming events in the next 60 days for that filter."}
    fut.sort(key=lambda x: x[0])
    edt, e = fut[0]
    hrs = (edt - now).total_seconds() / 3600.0
    out = {"event": e["name"], "date": e["date"], "timeET": e.get("time"),
           "importance": e["importance"], "source": e["source"],
           "countdownHours": round(hrs, 1), "countdown": f"{int(hrs // 24)}d {int(hrs % 24)}h"}
    for k in ("actual", "forecast", "previous"):
        if e.get(k) is not None:
            out[k] = e[k]
    src = (e.get("source") or "").lower()
    if "verify" in src or "approx" in src or "rule" in src:
        # This is the failure that put a phantom CPI on 2026-07-15 and drove a bad session plan.
        # A guess must never be returned looking like a confirmed date.
        out["UNVERIFIED"] = True
        out["warning"] = ("This date is a hardcoded/rule-based ESTIMATE, not a confirmed release "
                          "date. This table has been wrong by a full day. Verify before sizing "
                          "or planning a session around it.")
    return out


# ======================================================================
# Tier 2: FRED macroeconomic data (key-less fredgraph backend)
# ======================================================================

FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_UA = "Mozilla/5.0"  # FRED's CDN tarpits long browser UAs; a simple one passes
FRED_API = "https://api.stlouisfed.org/fred"
FRED_CACHE_TTL = 1800.0          # 30 min; macro series update slowly
FRED_TIMEOUT = 30.0

# Friendly catalog of the series the higher-level tools use.
SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate (monthly, %)",
    "DFF": "Effective Federal Funds Rate (daily, %)",
    "DGS3MO": "3-Month Treasury (%)", "DGS2": "2-Year Treasury (%)",
    "DGS5": "5-Year Treasury (%)", "DGS10": "10-Year Treasury (%)",
    "DGS30": "30-Year Treasury (%)", "T10Y2Y": "10Y-2Y Treasury spread (%)",
    "T10Y3M": "10Y-3M Treasury spread (%)",
    "CPIAUCSL": "CPI, all items (index)", "CPILFESL": "Core CPI, ex food & energy (index)",
    "PCEPI": "PCE price index", "PCEPILFE": "Core PCE price index",
    "T5YIE": "5-Year breakeven inflation (%)", "T10YIE": "10-Year breakeven inflation (%)",
    "UNRATE": "Unemployment rate (%)", "PAYEMS": "Nonfarm payrolls (thousands)",
    "CIVPART": "Labor force participation rate (%)", "ICSA": "Initial jobless claims",
    "CES0500000003": "Avg hourly earnings (index)",
    "GDPC1": "Real GDP (chained $bn, SAAR)", "INDPRO": "Industrial production (index)",
    "RSAFS": "Retail sales (advance, $mm)",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "BAMLH0A0HYM2": "ICE BofA US High-Yield OAS (%)",
    "BAMLC0A0CM": "ICE BofA US Corporate (IG) OAS (%)",
    "DTWEXBGS": "Trade-weighted US Dollar Index (broad)",
    "VIXCLS": "CBOE VIX (close)", "DCOILWTICO": "WTI crude oil ($/bbl)",
    "SAHMREALTIME": "Sahm Rule recession indicator (real-time, %)",
    "M2SL": "M2 money supply ($bn)", "WALCL": "Fed total assets ($mm)",
    "MORTGAGE30US": "30-Year fixed mortgage rate (%)",
    "UMCSENT": "U. Michigan consumer sentiment",
}


_fred_client: Optional[httpx.Client] = None
_fred_cache: "_TTLCache" = _TTLCache(256)   # (#7) bounded FRED series cache


def _fred_get_client() -> httpx.Client:
    global _fred_client
    if _fred_client is None:
        _fred_client = httpx.Client(timeout=FRED_TIMEOUT, follow_redirects=True,
                               headers={"User-Agent": FRED_UA})
    return _fred_client


# Cap concurrent hits to FRED's keyless fredgraph endpoint so a burst (e.g. regime_classifier pulling
# 4 series at once on a cold cache) can't trip rate limiting. These fetches run in to_thread workers,
# so a threading semaphore is the right primitive; it gates only the HTTP GET below, not cache hits.
_FRED_HTTP_SEM = threading.BoundedSemaphore(3)

# (#11e) Cap concurrent robin_stocks calls the same way. Tools like watchlist_radar / event_risk_radar
# / covered_call_manager fan out one _next_earnings thread per symbol; a 20-name list would otherwise
# hit robin_stocks with ~20 simultaneous requests. These run in to_thread workers, so a threading
# semaphore is the right primitive.
_RH_SYNC_SEM = threading.BoundedSemaphore(4)


class FredError(Exception):
    pass


def _fetch_series(series_id: str, start: Optional[str] = None,
                  end: Optional[str] = None) -> list[tuple[str, float]]:
    """Return [(date, value)] for a FRED series via the key-less fredgraph CSV endpoint."""
    sid = series_id.strip().upper()
    params = {"id": sid}
    if start:
        params["cosd"] = start
    if end:
        params["coed"] = end
    key = f"{sid}|{start}|{end}"
    hit = _fred_cache.get(key)
    if hit and (time.monotonic() - hit[1]) < FRED_CACHE_TTL:
        return hit[0]
    last = None
    for _ in range(3):
        try:
            with _FRED_HTTP_SEM:                 # cap concurrent FRED hits (released during backoff)
                r = _fred_get_client().get(FREDGRAPH, params=params)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    else:
        raise FredError(f"Request failed for {sid}: {last}")
    if r.status_code != 200:
        raise FredError(f"FRED returned HTTP {r.status_code} for {sid}")
    rows: list[tuple[str, float]] = []
    reader = csv.reader(io.StringIO(r.text))
    header = next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        d, v = row[0].strip(), row[1].strip()
        if v in ("", "."):
            continue
        try:
            rows.append((d, float(v)))
        except ValueError:
            continue
    if not rows:
        raise FredError(f"No observations for series '{sid}' (check the series ID).")
    _fred_cache.put(key, (rows, time.monotonic()))
    return rows


def _latest_obs(series_id: str) -> Optional[tuple[str, float]]:
    # Bounded windows keep downloads small (full history of daily series is large/slow).
    # B5: an empty window raises FredError -- CONTINUE to the wider window instead of bailing,
    # so sparse/long-lag series still resolve. (The old `return None` made 1800 dead code.)
    for win in (450, 1800):
        try:
            rows = _fetch_series(series_id, start=_recent_window(win))
        except FredError:
            continue
        if rows:
            return rows[-1]
    return None


def _recent_window(days: int) -> str:
    return (_today_et() - _dt.timedelta(days=days)).isoformat()   # B15: ET, not machine-local


def _yoy(series_id: str) -> Optional[dict]:
    """Latest year-over-year % change for a (monthly) series."""
    rows = _fetch_series(series_id, start=_recent_window(800))
    if len(rows) < 13:
        return None
    last_date, last_val = rows[-1]
    target = _dt.date.fromisoformat(last_date) - _dt.timedelta(days=365)
    prior = min(rows, key=lambda r: abs((_dt.date.fromisoformat(r[0]) - target).days))
    if prior[1] == 0:
        return None
    yoy = (last_val / prior[1] - 1.0) * 100.0
    return {"date": last_date, "value": round(last_val, 3),
            "yoy_pct": round(yoy, 2), "priorDate": prior[0]}


def _change(series_id: str, periods: int) -> Optional[dict]:
    """Latest level and absolute change over `periods` observations (e.g. MoM=1)."""
    rows = _fetch_series(series_id, start=_recent_window(1500))
    if len(rows) <= periods:
        return None
    last_date, last_val = rows[-1]
    prior_date, prior_val = rows[-1 - periods]
    return {"date": last_date, "value": round(last_val, 2),
            "change": round(last_val - prior_val, 2), "priorDate": prior_date}




@mcp.tool()
def fred_status() -> dict:
    """Health check: confirms the key-less FRED endpoint is reachable and reports the latest Fed Funds."""
    obs = _latest_obs("DFF") or _latest_obs("FEDFUNDS")
    if not obs:
        return {"reachable": False, "error": "Could not reach FRED fredgraph endpoint."}
    return {"reachable": True, "source": "FRED fredgraph (key-less)",
            "effectiveFedFunds": {"date": obs[0], "value": obs[1]},
            "catalogSeries": len(SERIES), "searchEnabled": bool(os.environ.get("FRED_API_KEY"))}


@mcp.tool()
def series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 24,
) -> dict:
    """Fetch observations for any FRED series ID (e.g. 'UNRATE', 'DGS10', 'CPIAUCSL').

    `start`/`end` are YYYY-MM-DD; `limit` caps how many of the most-recent observations are returned.
    Browse series IDs at https://fred.stlouisfed.org.
    """
    sid = series_id.strip().upper()
    try:
        rows = _fetch_series(sid, start=start, end=end)
    except FredError as exc:
        return {"error": str(exc)}
    tail = rows[-limit:]
    return {"series": sid, "title": SERIES.get(sid, "(see fred.stlouisfed.org)"),
            "observations": [{"date": d, "value": v} for d, v in tail],
            "count": len(tail), "totalAvailable": len(rows),
            "latest": {"date": rows[-1][0], "value": rows[-1][1]}}


@mcp.tool()
def latest(series_ids: str) -> dict:
    """Latest value for one or more FRED series IDs (comma-separated, e.g. 'DGS10,UNRATE,VIXCLS')."""
    ids = [s.strip().upper() for s in series_ids.split(",") if s.strip()]
    out = {}
    for sid in ids:
        obs = _latest_obs(sid)
        out[sid] = ({"date": obs[0], "value": obs[1], "title": SERIES.get(sid, "")}
                    if obs else {"error": "not found / unavailable"})
    return {"latest": out}


@mcp.tool()
def fed_funds() -> dict:
    """Current Fed Funds rate (effective, daily) plus the recent monthly path."""
    dff = _latest_obs("DFF")
    path = []
    try:
        rows = _fetch_series("FEDFUNDS", start=_recent_window(420))
        path = [{"date": d, "value": v} for d, v in rows[-12:]]
    except FredError:
        pass
    out = {"effectiveRate": ({"date": dff[0], "value": dff[1]} if dff else None),
           "recentMonthlyPath": path}
    if len(path) >= 2:
        out["last12moChange"] = round(path[-1]["value"] - path[0]["value"], 2)
    return out


@mcp.tool()
def yield_curve() -> dict:
    """Current US Treasury yield curve (3M, 2Y, 5Y, 10Y, 30Y), key spreads, and inversion flags."""
    pts = [("DGS3MO", "3M"), ("DGS2", "2Y"), ("DGS5", "5Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]
    curve = []
    vals = {}
    for sid, label in pts:
        obs = _latest_obs(sid)
        if obs:
            vals[label] = obs[1]
            curve.append({"tenor": label, "yield": obs[1], "date": obs[0]})
    s_10_2 = _latest_obs("T10Y2Y")
    s_10_3m = _latest_obs("T10Y3M")
    out = {"curve": curve,
           "spreads": {
               "10Y_2Y": (s_10_2[1] if s_10_2 else None),
               "10Y_3M": (s_10_3m[1] if s_10_3m else None)},
           "inverted_2s10s": (s_10_2[1] < 0 if s_10_2 else None),
           "inverted_3m10s": (s_10_3m[1] < 0 if s_10_3m else None)}
    if s_10_2:
        out["shape"] = ("inverted (recession signal)" if s_10_2[1] < 0
                        else "flat" if s_10_2[1] < 0.25 else "normal upward")
    return out


@mcp.tool()
def inflation() -> dict:
    """Headline & core CPI and PCE (latest level + YoY) plus 5Y/10Y market-implied breakevens."""
    out = {"cpi": _yoy("CPIAUCSL"), "coreCpi": _yoy("CPILFESL"),
           "pce": _yoy("PCEPI"), "corePce": _yoy("PCEPILFE")}
    be5 = _latest_obs("T5YIE")
    be10 = _latest_obs("T10YIE")
    out["breakevens"] = {
        "5Y": ({"date": be5[0], "value": be5[1]} if be5 else None),
        "10Y": ({"date": be10[0], "value": be10[1]} if be10 else None)}
    return out


@mcp.tool()
def labor_market() -> dict:
    """Unemployment rate, nonfarm payroll change (MoM), participation, wages (YoY), and initial claims."""
    unrate = _latest_obs("UNRATE")
    return {
        "unemploymentRate": ({"date": unrate[0], "value": unrate[1]} if unrate else None),
        "nonfarmPayrolls_MoM_thousands": _change("PAYEMS", 1),
        "participationRate": (lambda o: {"date": o[0], "value": o[1]} if o else None)(_latest_obs("CIVPART")),
        "avgHourlyEarnings_YoY": _yoy("CES0500000003"),
        "initialClaims": (lambda o: {"date": o[0], "value": int(o[1])} if o else None)(_latest_obs("ICSA")),
    }


@mcp.tool()
def growth() -> dict:
    """Real GDP (latest + QoQ change), industrial production (YoY), and retail sales (YoY)."""
    return {
        "realGDP_chained_bn": _change("GDPC1", 1),
        "industrialProduction_YoY": _yoy("INDPRO"),
        "retailSales_YoY": _yoy("RSAFS"),
    }


@mcp.tool()
def financial_conditions() -> dict:
    """Financial-conditions snapshot: Chicago Fed NFCI, HY & IG credit spreads, dollar index, and VIX."""
    def g(sid):
        o = _latest_obs(sid)
        return {"date": o[0], "value": o[1]} if o else None
    nfci = _latest_obs("NFCI")
    out = {
        "nfci": g("NFCI"),
        "nfciRead": (None if not nfci else
                     ("tighter than average" if nfci[1] > 0 else "looser than average")),
        "highYieldOAS_pct": g("BAMLH0A0HYM2"),
        "investmentGradeOAS_pct": g("BAMLC0A0CM"),
        "dollarIndexBroad": g("DTWEXBGS"),
        "vix": g("VIXCLS"),
    }
    return out


@mcp.tool()
def recession_indicators() -> dict:
    """Recession watch: Sahm Rule, 2s10s and 3m10s curve spreads, with a simple composite read."""
    sahm = _latest_obs("SAHMREALTIME")
    s2s10 = _latest_obs("T10Y2Y")
    s3m10 = _latest_obs("T10Y3M")
    flags = []
    if sahm and sahm[1] >= 0.50:
        flags.append("Sahm Rule triggered (>=0.50)")
    if s2s10 and s2s10[1] < 0:
        flags.append("2s10s inverted")
    if s3m10 and s3m10[1] < 0:
        flags.append("3m10s inverted")
    return {
        "sahmRule": ({"date": sahm[0], "value": sahm[1],
                      "triggered": sahm[1] >= 0.50} if sahm else None),
        "spread_2s10s": (s2s10[1] if s2s10 else None),
        "spread_3m10s": (s3m10[1] if s3m10 else None),
        "activeFlags": flags,
        "read": ("elevated signals" if len(flags) >= 2 else
                 "some signal" if flags else "no classic recession flags active"),
    }


@mcp.tool()
def series_search(query: str, limit: Annotated[int, Field(ge=1, le=50)] = 12) -> dict:
    """Search the FRED catalog by keyword for series IDs. Requires FRED_API_KEY (free) to be set."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return {"error": "Set FRED_API_KEY (free at https://fred.stlouisfed.org/docs/api/api_key.html) "
                         "to enable catalog search. Observation tools work without a key."}
    try:
        r = _fred_get_client().get(f"{FRED_API}/series/search", params={
            "search_text": query, "api_key": key, "file_type": "json",
            "limit": limit, "order_by": "popularity", "sort_order": "desc"})
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"FRED search failed: {str(exc)[:160]}"}
    out = [{"id": s.get("id"), "title": s.get("title"),
            "frequency": s.get("frequency_short"), "units": s.get("units_short"),
            "lastUpdated": s.get("last_updated")} for s in data.get("seriess", [])]
    return {"query": query, "count": len(out), "results": out}

# ======================================================================
# Cross-broker risk / Greeks aggregator (Alpaca-live + positions file)
# ======================================================================

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_ENV_DEFAULT = "/Users/jsconiers/Claude/mcp/alpaca/.env"
POSITIONS_FILE_DEFAULT = os.path.expanduser("~/.trading/positions.json")
CONC_THRESHOLD = float(os.environ.get("CONCENTRATION_PCT", "25"))   # flag underlyings above this % of gross
DAILY_TARGET_DEFAULT = float(os.environ.get("DAILY_TARGET", "524"))  # default daily profit target ($)


def _alpaca_creds() -> tuple:
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    paper = os.environ.get("ALPACA_PAPER_TRADE", "true")
    envf = os.environ.get("ALPACA_ENV_FILE", ALPACA_ENV_DEFAULT)
    if not (key and sec) and os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if line.startswith("ALPACA_API_KEY="):
                key = line.split("=", 1)[1].strip()
            elif line.startswith("ALPACA_SECRET_KEY="):
                sec = line.split("=", 1)[1].strip()
            elif line.startswith("ALPACA_PAPER_TRADE="):
                paper = line.split("=", 1)[1].strip()
    return key, sec, str(paper).lower() != "false"


async def _alpaca_get(path: str) -> Any:
    key, sec, paper = _alpaca_creds()
    if not (key and sec):
        raise EdgeError("Alpaca credentials not found (set ALPACA_API_KEY/ALPACA_SECRET_KEY or ALPACA_ENV_FILE).")
    base = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
    r = await _get_client().get(base + path,
                                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    if r.status_code in (401, 403):
        raise EdgeError(f"Alpaca auth failed ({r.status_code}); check keys and paper-vs-live.")
    r.raise_for_status()
    return r.json()


def _rh_prices_sync(symbols: list) -> dict:
    """(#11d) Batch last-trade prices for equity symbols via the Robinhood session in ONE get_quotes
    call -- replaces the anonymous, rate-limit-prone Yahoo endpoint that was the stack's only keyless
    third-party dependency. Missing symbols fall back to a per-symbol get_latest_price, else None."""
    syms = [s for s in symbols if s]
    if not syms:
        return {}
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    out: dict = {}
    try:
        for qr in (rh.get_quotes(syms) or []):
            if qr and qr.get("symbol"):
                out[qr["symbol"]] = _to_float(qr.get("last_trade_price")
                                               or qr.get("last_extended_hours_trade_price"))
    except Exception:  # noqa: BLE001
        pass
    for s in syms:
        if out.get(s) is None:
            try:
                lp = rh.get_latest_price(s)
                out[s] = _to_float(lp[0]) if lp else None
            except Exception:  # noqa: BLE001
                out[s] = None
    return out


def _greeks_risk(S: float, K: float, T: float, sigma: float, is_call: bool, r: float = RISK_FREE):
    """Scalar (delta, gamma, theta_per_day, vega_per_1pct_vol) for one option."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = _norm_pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    vega = S * pdf * sqrtT / 100.0
    if is_call:
        theta = (-(S * pdf * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * _norm_cdf(d2)) / 365.0
    else:
        theta = (-(S * pdf * sigma) / (2 * sqrtT) + r * K * np.exp(-r * T) * _norm_cdf(-d2)) / 365.0
    return float(delta), float(gamma), float(theta), float(vega)


def _sign_qty(qty, side: Optional[str]) -> float:
    q = float(qty)
    if side and side.lower() == "short" and q > 0:
        q = -q
    return q


def _normalize_alpaca(rows: list) -> list:
    out = []
    for p in rows or []:
        sym = p.get("symbol", "")
        ac = p.get("asset_class", "us_equity")
        qty = _sign_qty(p.get("qty", 0), p.get("side"))
        mv = float(p.get("market_value") or 0.0)
        if ac == "us_option":
            parsed = _parse_occ(sym)
            und = parsed[0] if parsed else sym
            out.append({"broker": "alpaca", "symbol": sym, "qty": qty, "type": "option",
                        "underlying": und, "mv": mv,
                        "unrealizedPL": float(p.get("unrealized_pl") or 0.0)})
        else:
            out.append({"broker": "alpaca", "symbol": sym, "qty": qty, "type": "equity",
                        "underlying": sym, "price": float(p.get("current_price") or 0.0),
                        "mv": mv, "unrealizedPL": float(p.get("unrealized_pl") or 0.0)})
    return out


def _read_positions_file() -> tuple:
    path = os.environ.get("POSITIONS_FILE", POSITIONS_FILE_DEFAULT)
    if not os.path.exists(path):
        return [], path
    import json as _json
    try:
        data = _json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        raise EdgeError(f"Could not parse positions file {path}: {str(exc)[:120]}")
    raw = data.get("positions", data) if isinstance(data, dict) else data
    out = []
    for p in raw or []:
        out.append({"broker": p.get("broker", "file"), "symbol": p.get("symbol", ""),
                    "qty": float(p.get("qty", 0)), "type": p.get("type", "equity"),
                    "underlying": p.get("underlying") or p.get("symbol", ""),
                    "iv": p.get("iv"), "price": p.get("price"), "beta": p.get("beta"),
                    "strike": p.get("strike"), "expiry": p.get("expiry"), "cp": p.get("cp"),
                    "delta": p.get("delta"), "mv": p.get("mv") or p.get("cost_basis")})
    return out, path


# ---- Beta map for SPX-weighting auto-pulled equities ----
# Editable ~5yr betas; override via BETA_OVERRIDES="ICE:1.05,NVDA:1.7" or BETA_MAP_FILE=<json path>.
_BETA_DEFAULTS = {
    "VOO": 1.00, "SPY": 1.00, "IVV": 1.00, "QQQ": 1.15,
    "ICE": 1.00, "AAPL": 1.20, "MSFT": 1.10, "NVDA": 1.65,
    "SCHD": 0.82, "TOPT": 1.10, "EPR": 1.35, "LTC": 0.85, "CHPY": 0.65,
}
_BETA_OVERRIDES = None


def _load_beta_overrides() -> dict:
    ov = {}
    for part in os.environ.get("BETA_OVERRIDES", "").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                ov[k.strip().upper()] = float(v)
            except ValueError:
                pass
    f = os.environ.get("BETA_MAP_FILE")
    if f and os.path.exists(f):
        import json as _j
        try:
            for k, v in (_j.load(open(f)) or {}).items():
                ov[str(k).strip().upper()] = float(v)
        except Exception:  # noqa: BLE001
            pass
    return ov


def _beta_for(symbol: str) -> float:
    global _BETA_OVERRIDES
    if _BETA_OVERRIDES is None:
        _BETA_OVERRIDES = _load_beta_overrides()
    s = (symbol or "").upper()
    return _BETA_OVERRIDES.get(s, _BETA_DEFAULTS.get(s, 1.0))


def _beta_of(p: dict) -> float:
    b = p.get("beta")
    if b is not None:
        try:
            return float(b)
        except (ValueError, TypeError):
            pass
    return _beta_for(p.get("underlying") or "")


def _position_risk(p: dict, chain_by_sym: dict, spot: float, prices: dict) -> dict:
    typ = p["type"]
    qty = p["qty"]
    if typ == "equity":
        px = p.get("price") or prices.get(p["underlying"]) or 0.0
        beta = _beta_of(p)
        dd = qty * px
        return {"broker": p["broker"], "symbol": p["symbol"], "underlying": p["underlying"],
                "type": "equity", "qty": qty, "price": round(px, 2), "beta": beta,
                "delta$": dd, "gamma$_1pct": 0.0, "theta$_day": 0.0, "vega$_1pct": 0.0,
                "betaDelta$": dd * beta,
                "mv": (p.get("mv") if p.get("mv") is not None else dd), "greeksSource": "equity"}
    parsed = _parse_occ(p["symbol"])
    if parsed:
        root, expiry, cp, strike = parsed
    else:
        root, expiry, cp, strike = (p.get("underlying") or ""), p.get("expiry"), p.get("cp"), p.get("strike")
    cpC = (cp == "C")
    mult = 100
    if p.get("delta") is not None and any(p.get(k) is not None for k in ("gamma", "theta", "vega")):
        undpx = float(p.get("price") or prices.get(p.get("underlying") or root) or 0.0)
        if not undpx and (p.get("underlying") or root) in ("SPX", "SPXW"):
            # B1: index underlyings have no equity quote (get_latest_price('SPXW') fails and the
            # Yahoo map deliberately excludes index roots), which silently zeroed delta$/gamma$
            # on exactly the 0DTE legs that matter most. Use the SPX chain spot instead.
            undpx = float(spot or 0.0)
        d = float(p["delta"]); gg = float(p.get("gamma") or 0.0)
        th = float(p.get("theta") or 0.0); vg = float(p.get("vega") or 0.0)
        beta = _beta_of(p)
        dd = d * qty * mult * undpx
        src = "broker" if undpx else "broker (NO UNDERLYING PRICE - delta$/gamma$ zeroed)"
        return {"broker": p["broker"], "symbol": p["symbol"], "underlying": (p.get("underlying") or root),
                "type": "option", "qty": qty, "strike": strike, "expiry": expiry, "delta$": dd,
                "gamma$_1pct": gg * qty * mult * undpx * undpx * 0.01,
                "theta$_day": th * qty * mult, "vega$_1pct": vg * qty * mult,
                "betaDelta$": dd * beta, "mv": p.get("mv"), "greeksSource": src}
    if root in ("SPX", "SPXW") and expiry and strike:
        o = chain_by_sym.get(p["symbol"])
        iv = float(o["iv"]) if (o and o.get("iv")) else 0.0
        T = _year_frac(expiry)
        if iv > 0 and T > 0:
            d, g, th, vg = _greeks_risk(spot, float(strike), T, iv, cpC)
            dd = d * qty * mult * spot
            return {"broker": p["broker"], "symbol": p["symbol"], "underlying": root, "type": "option",
                    "qty": qty, "strike": strike, "expiry": expiry, "delta$": dd,
                    "gamma$_1pct": g * qty * mult * spot * spot * 0.01,
                    "theta$_day": th * qty * mult, "vega$_1pct": vg * qty * mult,
                    "betaDelta$": dd, "mv": p.get("mv"), "greeksSource": "CBOE"}
    undpx = p.get("price") or prices.get(p.get("underlying") or root) or 0.0
    iv = p.get("iv")
    beta = _beta_of(p)
    if iv and undpx and strike and expiry and cp:
        T = _year_frac(expiry)
        d, g, th, vg = _greeks_risk(float(undpx), float(strike), T, float(iv), cpC)
        dd = d * qty * mult * undpx
        return {"broker": p["broker"], "symbol": p["symbol"], "underlying": (p.get("underlying") or root),
                "type": "option", "qty": qty, "strike": strike, "expiry": expiry, "delta$": dd,
                "gamma$_1pct": g * qty * mult * undpx * undpx * 0.01,
                "theta$_day": th * qty * mult, "vega$_1pct": vg * qty * mult,
                "betaDelta$": dd * beta, "mv": p.get("mv"), "greeksSource": "computed(file iv)"}
    if p.get("delta") is not None:
        d = float(p["delta"])
        dd = d * qty * mult * (undpx or 0.0)
        return {"broker": p["broker"], "symbol": p["symbol"], "underlying": (p.get("underlying") or root),
                "type": "option", "qty": qty, "delta$": dd, "gamma$_1pct": 0.0,
                "theta$_day": 0.0, "vega$_1pct": 0.0, "betaDelta$": dd * beta,
                "mv": p.get("mv"), "greeksSource": "file delta"}
    return {"broker": p["broker"], "symbol": p["symbol"], "underlying": (p.get("underlying") or root),
            "type": "option", "qty": qty, "delta$": 0.0, "gamma$_1pct": 0.0, "theta$_day": 0.0,
            "vega$_1pct": 0.0, "betaDelta$": 0.0, "mv": p.get("mv"), "greeksSource": "unavailable"}


# ---- Robinhood source (robin_stocks + cached session pickle) ----
RH_ENV_DEFAULT = "/Users/jsconiers/Claude/MCP/robin-hood/.env"
RH_PICKLE_DIR_DEFAULT = os.path.expanduser("~/.robinhood")
RH_CACHE_TTL = 60.0
_rh_logged_in = False
_rh_cache = {"ts": 0.0, "data": None}


def _occ_symbol(root: str, expiry: str, cp: str, strike) -> str:
    yy, mm, dd = expiry[2:4], expiry[5:7], expiry[8:10]
    return f"{root}{yy}{mm}{dd}{cp}{int(round(float(strike) * 1000)):08d}"


def _rh_login_sync() -> None:
    global _rh_logged_in
    if _rh_logged_in:
        return
    try:
        import robin_stocks.robinhood as rh
        from dotenv import load_dotenv
    except ImportError:
        raise EdgeError("robin_stocks not installed in this environment.")
    envf = os.environ.get("RH_ENV_FILE", RH_ENV_DEFAULT)
    if os.path.exists(envf):
        load_dotenv(envf)
    u, p = os.environ.get("RH_USERNAME"), os.environ.get("RH_PASSWORD")
    if not (u and p):
        raise EdgeError("Robinhood credentials not found (set RH_USERNAME/RH_PASSWORD or RH_ENV_FILE).")
    rh.login(u, p, store_session=True,
             pickle_path=os.environ.get("RH_PICKLE_PATH", RH_PICKLE_DIR_DEFAULT),
             pickle_name=os.environ.get("RH_PICKLE_NAME", ""), expiresIn=86400 * 7)
    _rh_logged_in = True


def _rh_collect_sync() -> list:
    import time as _t
    if _rh_cache["data"] is not None and (_t.time() - _rh_cache["ts"]) < RH_CACHE_TTL:
        return _rh_cache["data"]
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood.helper import request_get
    _rh_login_sync()

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    out = []
    for sym, d in (rh.build_holdings() or {}).items():
        q = _f(d.get("quantity")) or 0.0
        if q == 0:
            continue
        px = _f(d.get("price")) or 0.0
        out.append({"broker": "robinhood", "symbol": sym, "qty": q, "type": "equity",
                    "underlying": sym, "price": px,
                    "mv": (_f(d.get("equity")) if d.get("equity") else q * px), "beta": None})
    try:
        opos = rh.get_open_option_positions() or []
    except Exception:  # noqa: BLE001
        opos = []

    # B14: this used to fire THREE sequential RH calls per option position (instrument detail,
    # market data, latest underlying price) -- O(3N) round trips that made get_portfolio crawl.
    # Collect the ids once, fetch instruments + market data in batches of 40 (the same ids= shape
    # _rh_chain_sync uses) and all underlying quotes in a single get_quotes call; per-item calls
    # remain only as a fallback when a batch misses an id.
    live = []
    for pos in opos:
        q = _f(pos.get("quantity")) or 0.0
        if q == 0:
            continue
        opt_url = pos.get("option", "") or ""
        oid = opt_url.rstrip("/").split("/")[-1] if opt_url else None
        live.append((pos, oid, q))
    ids = [oid for (_p, oid, _q) in live if oid]

    inst_by_id, md_by_id = {}, {}
    for i in range(0, len(ids), 40):
        batch = ids[i:i + 40]
        try:
            for c in (request_get("https://api.robinhood.com/options/instruments/", "results",
                                  {"ids": ",".join(batch)}) or []):
                if c and c.get("id"):
                    inst_by_id[c["id"]] = c
        except Exception:  # noqa: BLE001
            pass
        try:
            for m in (request_get("https://api.robinhood.com/marketdata/options/", "results",
                                  {"ids": ",".join(batch)}) or []):
                if not m:
                    continue
                mid = (m.get("instrument") or "").rstrip("/").split("/")[-1]
                if mid:
                    md_by_id[mid] = m
        except Exception:  # noqa: BLE001
            pass

    chains = sorted({(pos.get("chain_symbol") or "")
                     for (pos, _o, _q) in live if pos.get("chain_symbol")})
    px_by_sym = {}
    if chains:
        try:
            for qr in (rh.get_quotes(chains) or []):
                if qr and qr.get("symbol"):
                    px_by_sym[qr["symbol"]] = _f(qr.get("last_trade_price"))
        except Exception:  # noqa: BLE001
            px_by_sym = {}

    for pos, oid, q in live:
        try:
            qty = q if (pos.get("type") or "").lower() != "short" else -q
            chain = pos.get("chain_symbol", "")
            opt_url = pos.get("option", "")
            od = inst_by_id.get(oid) or {}
            if not od and opt_url:                 # batch miss -> per-item fallback
                try:
                    od = rh.helper.request_get(opt_url) or {}
                except Exception:  # noqa: BLE001
                    od = {}
            oid2 = od.get("id") or oid
            strike = _f(od.get("strike_price"))
            expiry = od.get("expiration_date")
            cp = {"call": "C", "put": "P"}.get(od.get("type"))
            md = md_by_id.get(oid) or (md_by_id.get(oid2) if oid2 else None) or {}
            if not md and oid2:
                try:
                    raw = rh.get_option_market_data_by_id(oid2)
                    md = raw[0] if isinstance(raw, list) and raw else (raw or {})
                except Exception:  # noqa: BLE001
                    md = {}
            undpx = px_by_sym.get(chain)
            if undpx is None and chain:
                try:
                    lp = rh.get_latest_price(chain)
                    undpx = _f(lp[0]) if lp else None
                except Exception:  # noqa: BLE001
                    undpx = None
            mark = _f(md.get("mark_price")) or _f(md.get("adjusted_mark_price"))
            occ = (_occ_symbol(chain, expiry, cp, strike)
                   if (chain and expiry and cp and strike) else (opt_url or chain))
            out.append({"broker": "robinhood", "symbol": occ, "qty": qty, "type": "option",
                        "underlying": chain, "strike": strike, "expiry": expiry, "cp": cp,
                        "price": undpx, "iv": _f(md.get("implied_volatility")),
                        "delta": _f(md.get("delta")), "gamma": _f(md.get("gamma")),
                        "theta": _f(md.get("theta")), "vega": _f(md.get("vega")),
                        "mv": (mark * abs(qty) * 100 if mark else None),
                        "avgPrice": _f(pos.get("average_price")),
                        "mult": _f(pos.get("trade_value_multiplier")) or 100.0,
                        "mark": mark})
        except Exception:  # noqa: BLE001
            continue
    _rh_cache["data"], _rh_cache["ts"] = out, _t.time()
    return out


async def _robinhood_positions() -> list:
    import asyncio
    return await asyncio.to_thread(_rh_collect_sync)


# ---- E*TRADE source (pyetrade + cached OAuth token pickle) ----
ET_ENV_DEFAULT = "/Users/jsconiers/Claude/MCP/Etrade-MCP/.env"
ET_TOKEN_FILE_DEFAULT = os.path.expanduser("~/.etrade/tokens.pickle")
ET_CACHE_TTL = 60.0
_et_cache = {"ts": 0.0, "data": None}


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _et_dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def _etrade_clients():
    import pickle
    import pyetrade
    from dotenv import load_dotenv
    envf = os.environ.get("ET_ENV_FILE", ET_ENV_DEFAULT)
    if os.path.exists(envf):
        load_dotenv(envf)
    ck = os.environ.get("ETRADE_CONSUMER_KEY", "")
    cs = os.environ.get("ETRADE_CONSUMER_SECRET", "")
    if not (ck and cs):
        raise EdgeError("E*TRADE consumer key/secret not found (ETRADE_CONSUMER_KEY/SECRET or ET_ENV_FILE).")
    tf = os.environ.get("ET_TOKEN_FILE", ET_TOKEN_FILE_DEFAULT)
    if not os.path.exists(tf):
        raise EdgeError("E*TRADE token not found; authorize via the etrade MCP (setup_etrade_auth.py).")
    tk = pickle.load(open(tf, "rb"))
    ot, osec = tk.get("oauth_token", ""), tk.get("oauth_token_secret", "")
    dev = os.environ.get("ETRADE_SANDBOX", "false").lower() == "true"
    return (pyetrade.ETradeAccounts(ck, cs, ot, osec, dev=dev),
            pyetrade.ETradeAccessManager(ck, cs, ot, osec))


def _etrade_probe_sync() -> dict:
    """Lightweight liveness probe for the E*TRADE session: reactivate an idle token, then list
    accounts (a read-only GET). Confirms the token actually works without pulling full portfolios."""
    accounts, access = _etrade_clients()
    try:
        access.renew_access_token()  # reactivate an idle (not expired) token; harmless if active
    except Exception:  # noqa: BLE001
        pass
    a = accounts.list_accounts(resp_format="json")
    al = _et_dig(a, "AccountListResponse", "Accounts", "Account", default=[])
    if isinstance(al, dict):
        al = [al]
    return {"reachable": True, "accounts": len(al)}


# ---- E*TRADE market data: an INDEPENDENT live price feed -------------------
# Why this exists: every "is my data fresh?" check in this server ultimately leaned on ONE live
# SPX print (Robinhood's undocumented index endpoint). When that endpoint moved (2026-07-16 it
# returned nothing at all), _auto_basis fell through to its "uncalibrated" branch, which computes
# basis = chainSpot - spy*mult -- making spxLiveEst IDENTICAL to chainSpot and gapVsChainPts
# identically 0.0. The staleness detector silently became incapable of detecting staleness, and
# reported a 15-min-old price as live. E*TRADE is a second, independent, genuinely real-time feed
# (verified 2026-07-16: its VIX matched RH's print exactly at 16.30, and its SPX option-chain
# parity matched RH's SPX index to ~0.1pt). It also self-reports quoteStatus=REALTIME|DELAYED.
ET_QUOTE_TTL = 5.0
_et_quote_cache: dict = {"ts": 0.0, "key": None, "data": None}


def _etrade_market():
    """pyetrade ETradeMarket built from the same cached OAuth session as _etrade_clients()."""
    import pickle
    import pyetrade
    from dotenv import load_dotenv
    envf = os.environ.get("ET_ENV_FILE", ET_ENV_DEFAULT)
    if os.path.exists(envf):
        load_dotenv(envf)
    ck = os.environ.get("ETRADE_CONSUMER_KEY", "")
    cs = os.environ.get("ETRADE_CONSUMER_SECRET", "")
    if not (ck and cs):
        raise EdgeError("E*TRADE consumer key/secret not found (ETRADE_CONSUMER_KEY/SECRET or ET_ENV_FILE).")
    tf = os.environ.get("ET_TOKEN_FILE", ET_TOKEN_FILE_DEFAULT)
    if not os.path.exists(tf):
        raise EdgeError("E*TRADE token not found; authorize via the etrade MCP (setup_etrade_auth.py).")
    tk = pickle.load(open(tf, "rb"))
    dev = os.environ.get("ETRADE_SANDBOX", "false").lower() == "true"
    return pyetrade.ETradeMarket(ck, cs, tk.get("oauth_token", ""),
                                 tk.get("oauth_token_secret", ""), dev=dev)


def _etrade_index_quote_sync(symbols: str = "SPX,VIX") -> dict:
    """Live index/equity levels from E*TRADE. Same shape as _rh_index_quote_sync:
    {SYMBOL: {value, asof, source, realtime}}.

    Returns {} on ANY failure (expired token, network, unknown symbol) so callers fall through
    rather than blow up. `realtime` carries E*TRADE's own quoteStatus so a DELAYED quote can
    never be mistaken for a live one -- the whole point of this feed.
    """
    import time as _t
    syms = [s.strip().upper() for s in str(symbols).split(",") if s.strip()]
    if not syms:
        return {}
    key = ",".join(sorted(syms))
    if _et_quote_cache["key"] == key and _et_quote_cache["data"] is not None \
            and (_t.time() - _et_quote_cache["ts"]) < ET_QUOTE_TTL:
        return _et_quote_cache["data"]
    try:
        raw = _etrade_market().get_quote(syms, resp_format="json")
    except Exception as exc:  # noqa: BLE001 -- caller falls through to another source
        log.info("E*TRADE quote unavailable (%s)", str(exc)[:140])
        return {}
    rows = _et_dig(raw, "QuoteResponse", "QuoteData", default=[])
    if isinstance(rows, dict):
        rows = [rows]
    out = {}
    for r in rows or []:
        sym = str(_et_dig(r, "Product", "symbol") or "").upper()
        val = _to_float(_et_dig(r, "All", "lastTrade"))
        if not sym or not val:
            continue
        ts = r.get("dateTimeUTC")
        asof = (_dt.datetime.fromtimestamp(ts, _dt.timezone.utc).astimezone(ET).isoformat()
                if isinstance(ts, (int, float)) else r.get("dateTime"))
        status = str(r.get("quoteStatus") or "").upper()
        out[sym] = {"value": val, "asof": asof, "source": "etrade_market",
                    "realtime": (status == "REALTIME"), "quoteStatus": status or None}
    _et_quote_cache.update(ts=_t.time(), key=key, data=out)
    return out


async def _live_spx_print() -> tuple:
    """An INDEPENDENT live SPX print, or (None, None). Tries Robinhood's index endpoint first, then
    E*TRADE. Never derive SPX from the chain here -- that is exactly the circularity this avoids.
    Returns (value, source_label)."""
    import asyncio
    for fn, label in ((_rh_index_quote_sync, "robinhood_index"),
                      (_etrade_index_quote_sync, "etrade_market")):
        try:
            q = await asyncio.to_thread(fn, "SPX")
        except Exception:  # noqa: BLE001
            continue
        row = (q or {}).get("SPX") or {}
        if row.get("realtime") is False:      # E*TRADE explicitly told us it is delayed
            continue
        if row.get("value"):
            return row["value"], label
    return None, None


def _etrade_collect_sync() -> list:
    import time as _t
    if _et_cache["data"] is not None and (_t.time() - _et_cache["ts"]) < ET_CACHE_TTL:
        return _et_cache["data"]
    accounts, access = _etrade_clients()
    try:
        access.renew_access_token()  # reactivate an idle (not expired) token; harmless if already active
    except Exception:  # noqa: BLE001
        pass
    try:
        a = accounts.list_accounts(resp_format="json")
    except Exception as exc:  # noqa: BLE001
        raise EdgeError(f"E*TRADE session not usable ({type(exc).__name__}); re-authorize via the etrade MCP.")
    al = _et_dig(a, "AccountListResponse", "Accounts", "Account", default=[])
    if isinstance(al, dict):
        al = [al]
    out = []
    for acct0 in al:
        aidk = acct0.get("accountIdKey", "")
        if not aidk:
            continue
        try:
            port = accounts.get_account_portfolio(aidk, resp_format="json")
        except Exception:  # noqa: BLE001 -- empty portfolio returns HTTP 204 / empty body
            continue
        aps = _et_dig(port, "PortfolioResponse", "AccountPortfolio", default=[])
        if isinstance(aps, dict):
            aps = [aps]
        for ap in aps:
            plist = _et_dig(ap, "Position", default=[])
            if isinstance(plist, dict):
                plist = [plist]
            for pos in plist:
                prod = _et_dig(pos, "Product", default={}) or {}
                quick = _et_dig(pos, "Quick", default={}) or {}
                sym = prod.get("symbol", "")
                qty = _to_float(pos.get("quantity")) or 0.0
                if qty == 0:
                    continue
                mv = _to_float(pos.get("marketValue"))
                if prod.get("securityType") == "OPTN":
                    strike = _to_float(prod.get("strikePrice"))
                    cp = {"CALL": "C", "PUT": "P"}.get((prod.get("callPut") or "").upper())
                    ey, em, ed = prod.get("expiryYear"), prod.get("expiryMonth"), prod.get("expiryDay")
                    expiry = (f"{int(ey):04d}-{int(em):02d}-{int(ed):02d}" if (ey and em and ed) else None)
                    occ = (_occ_symbol(sym, expiry, cp, strike)
                           if (sym and expiry and cp and strike) else sym)
                    out.append({"broker": "etrade", "symbol": occ, "qty": qty, "type": "option",
                                "underlying": sym, "strike": strike, "expiry": expiry, "cp": cp,
                                "mv": mv})
                else:
                    out.append({"broker": "etrade", "symbol": sym, "qty": qty, "type": "equity",
                                "underlying": sym, "price": _to_float(quick.get("lastTrade")) or 0.0,
                                "mv": mv, "beta": None})
    _et_cache["data"], _et_cache["ts"] = out, _t.time()
    return out


async def _etrade_positions() -> list:
    import asyncio
    return await asyncio.to_thread(_etrade_collect_sync)


# ---- TastyTrade source (OAuth2 personal grant; shares the mcp-tastytrade .env) ----
# Read-only by design: this process requests the "read" scope only, so a token minted
# here cannot place an order even if the grant itself carries "trade". Order placement
# lives in the mcp-tastytrade server, separately gated by TASTYTRADE_ALLOW_TRADING.
TT_BASE_PROD = "https://api.tastyworks.com"
TT_BASE_CERT = "https://api.cert.tastyworks.com"
TT_ENV_DEFAULT = "/Users/jsconiers/Claude/mcp/mcp-tastytrade/.env"
TT_CACHE_TTL = float(os.environ.get("TT_CACHE_TTL", "20"))

_tt_token: dict = {"value": None, "exp": 0.0, "refreshes": 0}
_tt_cache: dict = {"ts": 0.0, "data": None}


def _tt_creds() -> tuple:
    """(client_secret, refresh_token, scope, base_url). Process env wins; else the shared .env."""
    sec = os.environ.get("TASTYTRADE_CLIENT_SECRET")
    ref = os.environ.get("TASTYTRADE_REFRESH_TOKEN")
    tt_env = os.environ.get("TASTYTRADE_ENV")
    envf = os.environ.get("TASTYTRADE_ENV_FILE", TT_ENV_DEFAULT)
    if not (sec and ref) and os.path.exists(envf):
        for line in open(envf):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if not v:
                continue
            if k == "TASTYTRADE_CLIENT_SECRET" and not sec:
                sec = v
            elif k == "TASTYTRADE_REFRESH_TOKEN" and not ref:
                ref = v
            elif k == "TASTYTRADE_ENV" and not tt_env:
                tt_env = v
    # Scope is deliberately NOT sourced from the .env: that file carries "read trade" for
    # the trading server. OAuth2 permits narrowing on refresh; this process only needs read.
    scope = os.environ.get("TASTYTRADE_READ_SCOPE", "read")
    base = TT_BASE_CERT if str(tt_env or "prod").lower() == "cert" else TT_BASE_PROD
    return sec, ref, scope, base


async def _tt_access_token(force: bool = False) -> str:
    sec, ref, scope, base = _tt_creds()
    if not (sec and ref):
        raise EdgeError("TastyTrade credentials not found (set TASTYTRADE_CLIENT_SECRET/"
                        "TASTYTRADE_REFRESH_TOKEN, or point TASTYTRADE_ENV_FILE at the .env).")
    if not force and _tt_token["value"] and time.time() < (_tt_token["exp"] - 30):
        return _tt_token["value"]
    r = await _get_client().post(
        base + "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": ref,
              "client_secret": sec, "scope": scope},
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    if r.status_code != 200:
        # Never echo the body: it can contain the submitted secret.
        raise EdgeError(f"TastyTrade OAuth refresh failed ({r.status_code}); check "
                        "TASTYTRADE_CLIENT_SECRET / TASTYTRADE_REFRESH_TOKEN in the .env.")
    d = r.json()
    _tt_token["value"] = d["access_token"]
    _tt_token["exp"] = time.time() + float(d.get("expires_in") or 900)
    _tt_token["refreshes"] += 1
    return _tt_token["value"]


async def _tt_get(path: str, params: Optional[dict] = None) -> Any:
    """GET a TastyTrade REST path, unwrapping the {data: ...} envelope. Retries once on 401."""
    _, _, _, base = _tt_creds()
    for attempt in (0, 1):
        tok = await _tt_access_token(force=(attempt == 1))
        r = await _get_client().get(base + path, params=params or {},
                                    headers={"Authorization": f"Bearer {tok}"})
        if r.status_code == 401 and attempt == 0:
            continue
        if r.status_code in (401, 403):
            raise EdgeError(f"TastyTrade auth failed ({r.status_code}); the refresh token or "
                            "grant scope may have been revoked.")
        r.raise_for_status()
        body = r.json()
        return body.get("data", body) if isinstance(body, dict) else body
    raise EdgeError("TastyTrade auth failed after refresh retry.")


async def _tt_account_numbers() -> list:
    envv = os.environ.get("TASTYTRADE_ACCOUNT_NUMBERS") or os.environ.get("TASTYTRADE_ACCOUNT_ID")
    if envv:
        return [a.strip() for a in envv.split(",") if a.strip()]
    data = await _tt_get("/customers/me/accounts")
    return [it["account"]["account-number"] for it in (data.get("items") or [])
            if (it.get("account") or {}).get("account-number")]


def _normalize_tastytrade(items: list, acct: str) -> tuple:
    """-> (normalized rows, list of skipped instrument-types). TT pads OCC roots to 6 chars
    with spaces ('SPXW  260717P06400000'); _parse_occ needs that whitespace gone."""
    out, skipped = [], []
    for p in items or []:
        raw = str(p.get("symbol") or "")
        itype = str(p.get("instrument-type") or "")
        direction = str(p.get("quantity-direction") or "Long")
        qty = _to_float(p.get("quantity")) or 0.0
        if not raw or qty == 0 or direction == "Zero":
            continue
        qty = -qty if direction == "Short" else qty
        px = _to_float(p.get("mark-price"))
        if px is None:
            px = _to_float(p.get("close-price"))
        mult = _to_float(p.get("multiplier")) or (100.0 if itype == "Equity Option" else 1.0)
        mv = (qty * mult * px) if px is not None else None
        if itype == "Equity Option":
            sym = "".join(raw.split())
            parsed = _parse_occ(sym)
            und = str(p.get("underlying-symbol") or "").strip() or (parsed[0] if parsed else sym)
            row = {"broker": "tastytrade", "symbol": sym, "qty": qty, "type": "option",
                   "underlying": und, "mv": mv, "account": acct}
            if parsed:
                row.update(expiry=parsed[1], cp=parsed[2], strike=parsed[3])
            out.append(row)
        elif itype in ("Equity", "Cryptocurrency"):
            und = str(p.get("underlying-symbol") or "").strip() or raw
            out.append({"broker": "tastytrade", "symbol": raw, "qty": qty, "type": "equity",
                        "underlying": und, "price": px or 0.0, "mv": mv, "beta": None,
                        "account": acct})
        else:
            # Futures / future options carry a different multiplier and symbology than
            # _position_risk assumes. Surface them rather than corrupt the Greek rollup.
            skipped.append(f"{raw} ({itype})")
    return out, skipped


async def _tastytrade_positions() -> tuple:
    """-> (rows, skipped). Cached for TT_CACHE_TTL seconds like the other broker sources."""
    if _tt_cache["data"] is not None and (time.time() - _tt_cache["ts"]) < TT_CACHE_TTL:
        return _tt_cache["data"]
    rows, skipped = [], []
    for acct in await _tt_account_numbers():
        data = await _tt_get(f"/accounts/{acct}/positions", {"include-marks": "true"})
        r, s = _normalize_tastytrade(data.get("items") or [], acct)
        rows += r
        skipped += s
    _tt_cache["data"], _tt_cache["ts"] = (rows, skipped), time.time()
    return rows, skipped


async def _collect_positions(include_alpaca: bool, include_file: bool, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True) -> tuple:
    positions, meta = [], {}
    if include_alpaca:
        try:
            rows = await _alpaca_get("/v2/positions")
            positions += _normalize_alpaca(rows)
            meta["alpaca"] = len(rows)
        except EdgeError as exc:
            meta["alpacaError"] = str(exc)
        except Exception as exc:  # noqa: BLE001 (B9: a transport error must not kill the rollup)
            meta["alpacaError"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    if include_robinhood:
        try:
            rhpos = await _robinhood_positions()
            positions += rhpos
            meta["robinhood"] = len(rhpos)
        except EdgeError as exc:
            meta["robinhoodError"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            meta["robinhoodError"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    if include_etrade:
        try:
            etpos = await _etrade_positions()
            positions += etpos
            meta["etrade"] = len(etpos)
        except EdgeError as exc:
            meta["etradeError"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            meta["etradeError"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    if include_tastytrade:
        try:
            ttpos, ttskip = await _tastytrade_positions()
            positions += ttpos
            meta["tastytrade"] = len(ttpos)
            if ttskip:
                meta["tastytradeSkipped"] = ttskip
        except EdgeError as exc:
            meta["tastytradeError"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            meta["tastytradeError"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    if include_file:
        try:
            fpos, path = _read_positions_file()
            positions += fpos
            meta["file"], meta["filePath"] = len(fpos), path
        except EdgeError as exc:
            meta["fileError"] = str(exc)
        except Exception as exc:  # noqa: BLE001 (B9)
            meta["fileError"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    # explicit per-source inclusion flags (a stale OAuth must not silently drop legs)
    _req = {"alpaca": include_alpaca, "robinhood": include_robinhood,
            "etrade": include_etrade, "tastytrade": include_tastytrade, "file": include_file}
    for _src, _on in _req.items():
        if _on:
            meta[f"{_src}Included"] = (f"{_src}Error" not in meta)
    return positions, meta


async def _price_map(positions: list, spot: float) -> dict:
    need = set()
    for p in positions:
        if p["type"] == "equity" and not p.get("price"):
            need.add(p["underlying"])
        elif p["type"] == "option":
            if p.get("price"):     # B13: broker already priced the underlying; skip Yahoo
                continue
            root = (_parse_occ(p["symbol"]) or [""])[0] or (p.get("underlying") or "")
            if root not in ("SPX", "SPXW"):
                need.add(p.get("underlying") or root)
    need = [s for s in need if s]
    if not need:
        return {}
    import asyncio
    prices = await asyncio.to_thread(_rh_prices_sync, need)   # (#11d) RH batch, was Yahoo fan-out
    return {s: prices.get(s) for s in need}


async def _aggregate(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True) -> dict:
    positions, meta = await _collect_positions(include_alpaca, include_file, include_robinhood, include_etrade, include_tastytrade)
    chain, spot = None, 0.0
    has_spx = any(p["type"] == "option" and ((_parse_occ(p["symbol"]) or [""])[0] in ("SPX", "SPXW"))
                  for p in positions)
    if has_spx:
        try:
            chain = await _load_chain()
            spot = chain["spot"]
        except EdgeError:
            pass
    chain_by_sym = {o["symbol"]: o for o in (chain["options"] if chain else [])}
    prices = await _price_map(positions, spot)
    risks = [_position_risk(p, chain_by_sym, spot, prices) for p in positions]
    chain_fresh = _staleness(chain.get("asof")) if chain else None
    return {"positions": risks, "spot": spot, "meta": meta,
            "chainSource": ("cboe_delayed" if chain else None), "chainFreshness": chain_fresh}


@mcp.tool()
async def risk_status() -> dict:
    """Health check for the cross-broker aggregator: which position sources are configured and reachable."""
    key, sec, paper = _alpaca_creds()
    out = {"alpaca": {"configured": bool(key and sec),
                      "mode": "paper" if paper else "live"}}
    if key and sec:
        try:
            acct = await _alpaca_get("/v2/account")
            out["alpaca"]["reachable"] = True
            out["alpaca"]["equity"] = float(acct.get("equity") or 0.0)
        except EdgeError as exc:
            out["alpaca"]["reachable"] = False
            out["alpaca"]["error"] = str(exc)
    rh_env = os.environ.get("RH_ENV_FILE", RH_ENV_DEFAULT)
    rh_pickle = os.path.join(os.environ.get("RH_PICKLE_PATH", RH_PICKLE_DIR_DEFAULT),
                             f"robinhood{os.environ.get('RH_PICKLE_NAME', '')}.pickle")
    try:
        import robin_stocks  # noqa: F401
        rh_lib = True
    except ImportError:
        rh_lib = False
    out["robinhood"] = {"libInstalled": rh_lib, "envExists": os.path.exists(rh_env),
                        "sessionPickleExists": os.path.exists(rh_pickle), "loggedIn": _rh_logged_in}
    et_env = os.environ.get("ET_ENV_FILE", ET_ENV_DEFAULT)
    et_tok = os.environ.get("ET_TOKEN_FILE", ET_TOKEN_FILE_DEFAULT)
    try:
        import pyetrade  # noqa: F401
        et_lib = True
    except ImportError:
        et_lib = False
    et = {"libInstalled": et_lib, "envExists": os.path.exists(et_env),
          "tokenPickleExists": os.path.exists(et_tok)}
    # "Token pickle exists" != "logged in": E*TRADE tokens expire at midnight ET, so actually probe it.
    if et_lib and et["envExists"] and et["tokenPickleExists"]:
        import asyncio
        try:
            probe = await asyncio.wait_for(asyncio.to_thread(_etrade_probe_sync), timeout=10)
            et.update(probe)
        except asyncio.TimeoutError:
            et.update({"reachable": False, "error": "probe timed out (>10s)"})
        except EdgeError as exc:
            et.update({"reachable": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            et.update({"reachable": False,
                       "error": f"{type(exc).__name__}; re-authorize via the etrade MCP"})
    else:
        et["reachable"] = False
    out["etrade"] = et
    tt_sec, tt_ref, tt_scope, tt_base = _tt_creds()
    tt = {"configured": bool(tt_sec and tt_ref),
          "envFile": os.environ.get("TASTYTRADE_ENV_FILE", TT_ENV_DEFAULT),
          "env": "cert" if tt_base == TT_BASE_CERT else "prod",
          "scope": tt_scope, "readOnly": "trade" not in tt_scope}
    # Same rule as E*TRADE: "credentials on disk" != "grant still valid" -- probe it.
    if tt["configured"]:
        try:
            accts = await _tt_account_numbers()
            tt.update(reachable=True, accounts=len(accts), tokenRefreshes=_tt_token["refreshes"])
        except EdgeError as exc:
            tt.update(reachable=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            tt.update(reachable=False, error=f"{type(exc).__name__}: {str(exc)[:140]}")
    else:
        tt["reachable"] = False
    out["tastytrade"] = tt
    _, fpath = _read_positions_file()
    out["positionsFile"] = {"path": fpath, "exists": os.path.exists(fpath)}
    return out


@mcp.tool()
async def alpaca_positions() -> dict:
    """Live positions held in the Alpaca account (equities and options), normalized."""
    try:
        rows = await _alpaca_get("/v2/positions")
    except EdgeError as exc:
        return {"error": str(exc)}
    norm = _normalize_alpaca(rows)
    return {"mode": "paper" if _alpaca_creds()[2] else "live", "count": len(norm), "positions": norm}


@mcp.tool()
async def etrade_positions() -> dict:
    """Live E*TRADE holdings (stocks + options) via the cached pyetrade OAuth session.

    Reuses the ~/.etrade/tokens.pickle session shared with the etrade MCP (auto-renews an idle token).
    Equities get current price + market value; SPX/SPXW option legs are priced from CBOE. If the daily
    E*TRADE token has expired, re-authorize via the etrade MCP (setup_etrade_auth.py).
    """
    try:
        pos = await _etrade_positions()
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    stocks = [x for x in pos if x["type"] == "equity"]
    opts = [x for x in pos if x["type"] == "option"]
    return {"count": len(pos), "stocks": len(stocks), "options": len(opts), "positions": pos}


@mcp.tool()
async def robinhood_positions() -> dict:
    """Live Robinhood holdings (stocks + options) via the cached robin_stocks session.

    Stocks come from build_holdings; option legs include broker-provided Greeks (delta/gamma/theta/vega)
    and IV. Auth reuses the ~/.robinhood session pickle (shared with the robinhood-local server); if it
    has expired you'll get a one-time device-approval prompt in the Robinhood app on first use.
    """
    try:
        pos = await _robinhood_positions()
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    stocks = [x for x in pos if x["type"] == "equity"]
    opts = [x for x in pos if x["type"] == "option"]
    return {"count": len(pos), "stocks": len(stocks), "options": len(opts), "positions": pos}


@mcp.tool()
async def tastytrade_positions() -> dict:
    """Live tastytrade holdings (stocks + equity/index options) via the OAuth2 personal grant.

    Shares the mcp-tastytrade .env (TASTYTRADE_CLIENT_SECRET / TASTYTRADE_REFRESH_TOKEN), but mints
    its own read-scoped token, so this path can never place an order. SPX/SPXW legs are re-priced and
    Greeked from CBOE by the risk rollup; equities are marked from the broker. Futures and future
    options are reported under `skipped` rather than mixed into the Greek totals.
    """
    try:
        pos, skipped = await _tastytrade_positions()
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    stocks = [x for x in pos if x["type"] == "equity"]
    opts = [x for x in pos if x["type"] == "option"]
    out = {"count": len(pos), "stocks": len(stocks), "options": len(opts),
           "env": "cert" if _tt_creds()[3] == TT_BASE_CERT else "prod", "positions": pos}
    if skipped:
        out["skipped"] = skipped
    return out


@mcp.tool()
async def load_positions() -> dict:
    """Show the broker-agnostic positions file (for holdings not in Alpaca, e.g. Robinhood/E*TRADE).

    Path defaults to ~/.trading/positions.json (override with POSITIONS_FILE). Each entry:
    {broker, symbol, qty (negative=short), type: equity|option, and optional underlying/strike/expiry/
    cp/iv/price/beta/delta/mv}. SPX/SPXW options are auto-priced from CBOE; others use the optional fields.
    """
    try:
        pos, path = _read_positions_file()
    except EdgeError as exc:
        return {"error": str(exc)}
    if not pos:
        return {"path": path, "exists": os.path.exists(path), "count": 0,
                "template": {"positions": [
                    {"broker": "robinhood", "symbol": "ICE", "qty": 500, "type": "equity",
                     "beta": 1.05, "mv": 75000},
                    {"broker": "robinhood", "symbol": "SPXW260620P07400000", "qty": -2, "type": "option"}]}}
    return {"path": path, "count": len(pos), "positions": pos}


def _sum(risks: list, key: str) -> float:
    return float(sum(r.get(key) or 0.0 for r in risks))


def _exposure(r: dict) -> float:
    mv = r.get("mv")
    return abs(float(mv)) if mv is not None else abs(float(r.get("delta$") or 0.0))


@mcp.tool()
async def net_greeks(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True) -> dict:
    """Net portfolio Greeks aggregated across Alpaca and the positions file.

    Sums dollar delta, dollar gamma (per 1% move), theta (per day), and vega (per 1% vol). SPX/SPXW
    options get full Black-Scholes Greeks off CBOE; equities contribute delta only (beta-weighted);
    other instruments use whatever the positions file provides. Delta is also expressed in SPX points.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade, include_tastytrade)
    risks, spot = agg["positions"], agg["spot"]
    if not risks:
        return {"note": "No positions found.", "sources": agg["meta"]}
    netd = _sum(risks, "delta$")
    netbd = _sum(risks, "betaDelta$")
    netg = _sum(risks, "gamma$_1pct")
    nett = _sum(risks, "theta$_day")
    netv = _sum(risks, "vega$_1pct")
    cov = {}
    for r in risks:
        cov[r["greeksSource"]] = cov.get(r["greeksSource"], 0) + 1
    out = {
        "spot": round(spot, 2) if spot else None,
        "source": agg.get("chainSource"), "freshness": agg.get("chainFreshness"),   # (#1)
        "positions": len(risks),
        "netDelta$": round(netd, 0),
        "netDelta_betaWeighted$": round(netbd, 0),
        "netDelta_SPXpoints": (round(netbd / spot, 1) if spot else None),
        "netGamma$_per_1pct": round(netg, 0),
        "netTheta$_per_day": round(nett, 0),
        "netVega$_per_1pct_vol": round(netv, 0),
        "greeksCoverage": cov,
        "chainSource": agg.get("chainSource"), "chainFreshness": agg.get("chainFreshness"),
        "feedWarnings": _feed_warnings(agg["meta"]),
        "sources": agg["meta"],
    }
    return out


@mcp.tool()
async def risk_summary(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True) -> dict:
    """Portfolio risk overview: beta-weighted SPX exposure, gross/long/short notional, and breakdowns.

    Groups exposure by broker and by underlying, and lists the largest directional contributors.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade, include_tastytrade)
    risks, spot = agg["positions"], agg["spot"]
    if not risks:
        return {"note": "No positions found.", "sources": agg["meta"]}
    gross = sum(_exposure(r) for r in risks)
    longn = sum(_exposure(r) for r in risks if (r.get("delta$") or 0) >= 0)
    shortn = sum(_exposure(r) for r in risks if (r.get("delta$") or 0) < 0)
    by_broker, by_under = {}, {}
    for r in risks:
        by_broker.setdefault(r["broker"], 0.0)
        by_broker[r["broker"]] += r.get("betaDelta$") or 0.0
        by_under.setdefault(r["underlying"], 0.0)
        by_under[r["underlying"]] += r.get("betaDelta$") or 0.0
    top = sorted(risks, key=lambda r: abs(r.get("delta$") or 0.0), reverse=True)[:6]
    netbd = _sum(risks, "betaDelta$")
    return {
        "spot": round(spot, 2) if spot else None,
        "source": agg.get("chainSource"), "freshness": agg.get("chainFreshness"),   # (#1)
        "netBetaDelta$": round(netbd, 0),
        "netBetaDelta_SPXpoints": (round(netbd / spot, 1) if spot else None),
        "grossNotional$": round(gross, 0),
        "longNotional$": round(longn, 0), "shortNotional$": round(shortn, 0),
        "byBroker_betaDelta$": {k: round(v, 0) for k, v in sorted(by_broker.items())},
        "byUnderlying_betaDelta$": {k: round(v, 0) for k, v in
                                    sorted(by_under.items(), key=lambda kv: -abs(kv[1]))},
        "topContributors": [{"symbol": r["symbol"], "underlying": r["underlying"],
                             "delta$": round(r.get("delta$") or 0.0, 0),
                             "source": r["greeksSource"]} for r in top],
        "sources": agg["meta"],
    }


@mcp.tool()
async def concentration(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True) -> dict:
    """Exposure concentration by underlying, flagging any name above the concentration threshold.

    Threshold defaults to 25% of gross exposure (override with CONCENTRATION_PCT).
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade, include_tastytrade)
    risks = agg["positions"]
    if not risks:
        return {"note": "No positions found.", "sources": agg["meta"]}
    by_under = {}
    for r in risks:
        by_under[r["underlying"]] = by_under.get(r["underlying"], 0.0) + _exposure(r)
    gross = sum(by_under.values()) or 1.0
    rows = sorted(({"underlying": k, "exposure$": round(v, 0), "pctOfGross": round(100 * v / gross, 1)}
                   for k, v in by_under.items()), key=lambda x: -x["pctOfGross"])
    flagged = [r for r in rows if r["pctOfGross"] >= CONC_THRESHOLD]
    return {"grossExposure$": round(gross, 0), "thresholdPct": CONC_THRESHOLD,
            "byUnderlying": rows, "flagged": flagged,
            "note": (f"{flagged[0]['underlying']} is {flagged[0]['pctOfGross']}% of gross exposure."
                     if flagged else "No single name exceeds the concentration threshold."),
            "sources": agg["meta"]}


@mcp.tool()
async def scenario_shock(
    moves_pct: Optional[str] = None,
    include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True, include_tastytrade: bool = True,
) -> dict:
    """Estimate portfolio P&L under a set of SPX % moves using net beta-delta + net gamma convexity.

    `moves_pct` is a comma-separated list (e.g. '-2,-1,-0.5,0.5,1,2'); defaults to that set. SPX/SPXW
    positions use delta + gamma; everything else is beta-weighted linear. A quick risk read, not a
    full revaluation.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade, include_tastytrade)
    risks, spot = agg["positions"], agg["spot"]
    if not risks:
        return {"note": "No positions found.", "sources": agg["meta"]}
    netbd = _sum(risks, "betaDelta$")
    netg = _sum(risks, "gamma$_1pct")
    if moves_pct:
        try:
            moves = [float(x) / 100.0 for x in moves_pct.split(",") if x.strip()]
        except ValueError:
            return {"error": "moves_pct must be comma-separated numbers, e.g. '-2,-1,1,2'."}
    else:
        moves = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
    scen = []
    for m in moves:
        pnl = netbd * m + 50.0 * netg * m * m
        scen.append({"spxMovePct": round(m * 100, 2),
                     "spxLevel": (round(spot * (1 + m), 2) if spot else None),
                     "estPnL$": round(pnl, 0)})
    return {"spot": round(spot, 2) if spot else None,
            "source": agg.get("chainSource"), "freshness": agg.get("chainFreshness"),   # (#1)
            "netBetaDelta$": round(netbd, 0), "netGamma$_per_1pct": round(netg, 0),
            "scenarios": scen,
            "method": "P&L = betaDelta$*move + 50*gamma$1pct*move^2 (gamma applies to SPX/SPXW legs)",
            "sources": agg["meta"]}


@mcp.tool()
async def daily_target(
    target: Optional[float] = None,
    realized_pl: Optional[float] = None,
) -> dict:
    """Track today's realized P&L against your daily target, with a post-target discipline check.

    `target` defaults to DAILY_TARGET ($524). `realized_pl` can be passed directly; otherwise it is
    estimated from Alpaca as account equity minus prior-day equity. If you're already past target, this
    flags it - your logged history shows post-target trades have generated most of your losses.
    """
    tgt = float(target) if target is not None else _target()
    src = "provided"
    rpl = realized_pl
    if rpl is None:
        feed = await _rh_realized_today()
        if feed is not None:
            rpl = feed["feeInclusiveRealized$"]
            src = "robinhood (round-trip realized, fee-inclusive)"
        else:
            try:
                acct = await _alpaca_get("/v2/account")
                eq = float(acct.get("equity") or 0.0)
                last = float(acct.get("last_equity") or 0.0)
                rpl = eq - last
                src = "alpaca (equity - last_equity) [RH feed unavailable]"
            except EdgeError as exc:
                return {"error": f"No realized_pl and neither RH fills nor Alpaca available: {exc}"}
    pct = (rpl / tgt * 100.0) if tgt else None
    over = rpl >= tgt
    status = ("TARGET HIT" if over else "below target" if rpl >= 0 else "in drawdown")
    out = {"target$": round(tgt, 2), "realizedPnL$": round(rpl, 2),
           "pctOfTarget": (round(pct, 1) if pct is not None else None),
           "source": src, "status": status}
    if over:
        out["guardrail"] = ("You're past your daily target. Your own logged pattern is that trades taken "
                            "after hitting target produce most of your losses - strongly consider closing "
                            "the platform for the day, or cutting size to 1/4 if you keep going.")
    elif rpl < 0:
        out["guardrail"] = "In drawdown - trade your plan; avoid revenge-sizing to get back to flat."
    return out


# ============================================================================
# DISCIPLINE / ANTI-OVERTRADING LAYER (Robinhood option fills)
# ============================================================================
DD_GIVEBACK_FRAC = float(os.environ.get("TE_GIVEBACK_FRAC", "0.40"))  # of target, from peak -> STOP
RAPID_REENTRY_SECS = float(os.environ.get("TE_RAPID_REENTRY_SECS", "90"))
LATE_SESSION_ET = os.environ.get("TE_LATE_SESSION_ET", "15:45")  # final-stretch caution


def _rh_recent_option_orders(stop_date: _dt.date, max_pages: int = 12, page_size: int = 100):
    """Filled+other option orders, newest-first, paginating only until we pass stop_date (ET).

    B7: returns (orders, meta). meta['truncated'] is True when we hit the page cap while more
    in-window history still remained (a 'next' page existed and the oldest row seen had not yet
    crossed stop_date), so callers can flag 'YTD'/window figures that are actually partial.
    page_size lifts RH's ~10-per-page default so a full year fits in far fewer round trips."""
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        base = rh.urls.option_orders()
    except Exception:  # noqa: BLE001
        base = "https://api.robinhood.com/options/orders/"
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}page_size={int(page_size)}"
    out, page, oldest_d = [], 0, None
    data = rh.helper.request_get(url, "regular")
    while data and isinstance(data, dict):
        results = data.get("results", []) or []
        out.extend(results)
        page += 1
        if results:
            try:
                oldest_d = _dt.datetime.fromisoformat(
                    results[-1].get("created_at", "").replace("Z", "+00:00")).astimezone(ET).date()
            except Exception:  # noqa: BLE001
                oldest_d = None
        nxt = data.get("next")
        passed = bool(oldest_d and oldest_d < stop_date)
        if not nxt or page >= max_pages or passed:
            truncated = bool(nxt) and (page >= max_pages) and not passed
            return out, {"pages": page, "truncated": truncated,
                         "oldestSeen": oldest_d.isoformat() if oldest_d else None}
        data = rh.helper.request_get(nxt, "regular")
    return out, {"pages": page, "truncated": False,
                 "oldestSeen": oldest_d.isoformat() if oldest_d else None}


def _leg_fill(o: dict, lg: dict, when, trade_date: str) -> Optional[dict]:
    """One per-leg fill record from a (multi-leg) option order leg. Cash flow is the leg's OWN
    signed premium: sell = credit (+), buy = debit (-). Carries order_id so round-trip matching
    can dedupe the several legs that share one spread/roll order."""
    exs = lg.get("executions") or []
    qty, gross, px_last = 0.0, 0.0, None
    for ex in exs:
        q = _to_float(ex.get("quantity")) or 0.0
        p = _to_float(ex.get("price"))
        if p is not None:
            px_last = p
            gross += p * q
        qty += q
    if qty <= 0:
        qty = _to_float(lg.get("ratio_quantity")) or 0.0
    side = lg.get("side")
    sign = 1.0 if side == "sell" else -1.0
    return {"time": when, "trade_date": trade_date, "chain": o.get("chain_symbol", ""),
            "n_legs": len(o.get("legs") or []), "net_cf": sign * gross * 100.0,
            "gross_premium": None, "qty": qty,
            "option_id": (lg.get("option") or "").rstrip("/").split("/")[-1],
            "side": side, "effect": lg.get("position_effect"),
            "strike": _to_float(lg.get("strike_price")),
            "cp": {"call": "C", "put": "P"}.get(lg.get("option_type")),
            "expiry": lg.get("expiration_date"),
            "price": px_last, "order_id": o.get("id")}


def _order_to_fills(o: dict) -> list:
    """B2: decompose an option order into per-leg fill records so MULTI-LEG orders (vertical
    spreads, rolls) are visible to round-trip matching -- the old _order_to_fill collapsed them
    into a single effect=None record that every P&L/discipline tool silently skipped. Single-leg
    orders yield one record identical in shape (and net_cf semantics) to the legacy output.
    Returns [] for unfilled orders."""
    if o.get("state") != "filled":
        return []
    legs = o.get("legs", []) or []
    ts, trade_date = None, None
    for lg in legs:
        for ex in (lg.get("executions") or []):
            t_ = ex.get("timestamp")
            if t_ and (ts is None or t_ > ts):
                ts = t_
            trade_date = ex.get("trade_date") or trade_date
    ts = ts or o.get("updated_at") or o.get("created_at")
    try:
        when = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)
    except Exception:  # noqa: BLE001
        return []
    trade_date = trade_date or when.date().isoformat()

    if len(legs) == 1:
        net = _to_float(o.get("net_amount")) or 0.0
        direction = o.get("net_amount_direction") or o.get("direction")
        net_cf = net if direction == "credit" else -net
        lg = legs[0]
        ex0 = (lg.get("executions") or [{}])
        return [{"time": when, "trade_date": trade_date, "chain": o.get("chain_symbol", ""),
                 "n_legs": 1, "net_cf": net_cf,
                 "gross_premium": _to_float(o.get("processed_premium")),
                 "qty": _to_float(o.get("processed_quantity")) or _to_float(o.get("quantity")) or 0.0,
                 "option_id": (lg.get("option") or "").rstrip("/").split("/")[-1],
                 "side": lg.get("side"), "effect": lg.get("position_effect"),
                 "strike": _to_float(lg.get("strike_price")),
                 "cp": {"call": "C", "put": "P"}.get(lg.get("option_type")),
                 "expiry": lg.get("expiration_date"),
                 "price": _to_float(ex0[0].get("price")) if ex0 else None,
                 "order_id": o.get("id")}]

    fills = [rec for lg in legs if (rec := _leg_fill(o, lg, when, trade_date)) is not None]
    if not fills:
        net = _to_float(o.get("net_amount")) or 0.0
        direction = o.get("net_amount_direction") or o.get("direction")
        net_cf = net if direction == "credit" else -net
        fills.append({"time": when, "trade_date": trade_date, "chain": o.get("chain_symbol", ""),
                      "n_legs": len(legs), "net_cf": net_cf, "gross_premium": None,
                      "qty": _to_float(o.get("processed_quantity")) or 0.0,
                      "option_id": "multi:" + (o.get("id") or ""), "side": None, "effect": None,
                      "strike": None, "cp": None, "expiry": None, "price": None,
                      "order_id": o.get("id"), "legFallback": True})
    return fills


def _day_fills_sync(date_iso: str):
    """B2/B7: returns (fills, meta). fills now include per-leg records for multi-leg orders."""
    target = _dt.date.fromisoformat(date_iso)
    orders, ometa = _rh_recent_option_orders(target)
    fills, leg_fallbacks = [], 0
    for o in orders:
        for f in _order_to_fills(o):
            if f.get("trade_date") == date_iso:
                if f.get("legFallback"):
                    leg_fallbacks += 1
                fills.append(f)
    fills.sort(key=lambda r: r["time"])
    meta = dict(ometa)
    meta["legFallback"] = leg_fallbacks
    return fills, meta


async def _day_fills(date_iso: str):
    import asyncio
    return await asyncio.to_thread(_day_fills_sync, date_iso)


def _fill_data_warnings(fmeta: dict, tstats: dict) -> list:
    """B2/B7: human-readable warnings when the realized-P&L picture may be incomplete."""
    w = []
    if fmeta.get("truncated"):
        w.append("Order history hit the page cap; some older fills for this window may be missing.")
    if fmeta.get("legFallback"):
        w.append(f"{fmeta['legFallback']} multi-leg order(s) could not be split into legs; "
                 f"their round trips may be approximate.")
    if tstats.get("unmatchedCloses"):
        w.append(f"{tstats['unmatchedCloses']} closing fill(s) had no matching open in this "
                 f"window (${tstats['unmatchedCloseCF$']:.2f} cash) - the opens likely predate it.")
    if tstats.get("orderLevelFallback"):
        w.append(f"{tstats['orderLevelFallback']} order(s) counted at order level only "
                 f"(${tstats['fallbackCF$']:.2f} cash).")
    return w


def _fmt_et(d) -> str:
    return d.astimezone(ET).strftime("%H:%M:%S")


def _build_curve(trips: list, target: float) -> dict:
    """Cumulative REALIZED-P&L curve stepped by completed round trips (sorted by close time)."""
    trips = sorted(trips, key=lambda t: t["close"])
    cum = peak = 0.0
    max_dd = 0.0
    cross_i = cross_cum = cross_time = None
    rows = []
    for i, t in enumerate(trips):
        cum += t["pnl"]
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
        if cross_i is None and target > 0 and cum >= target:
            cross_i, cross_cum, cross_time = i, cum, t["close"]
        rows.append({"t": _fmt_et(t["close"]), "chain": t.get("chain"),
                     "strike": t.get("strike"), "cp": t.get("cp"),
                     "pnl$": round(t["pnl"], 2), "cum$": round(cum, 2)})
    return {"rows": rows, "total": cum, "peak": peak, "maxDrawdownFromPeak": abs(max_dd),
            "crossIdx": cross_i, "crossCum": cross_cum, "crossTime": cross_time}


def _round_trips_full(fills: list):
    """B2: FIFO round-trip matcher that ALSO reports unmatched closes (a close with no open lot
    in-window -- e.g. the open was truncated out of the window, or was a leg we couldn't
    decompose) and order-level fallbacks. Each trip carries open_order_id/close_order_id so
    callers can dedupe spread legs. Returns (trips, stats)."""
    from collections import defaultdict, deque
    lots = defaultdict(deque)
    trips = []
    unmatched_n, unmatched_cf, fallback_n, fallback_cf = 0, 0.0, 0, 0.0
    for f in fills:
        oid, eff, qty, net = f.get("option_id"), f.get("effect"), (f.get("qty") or 0.0), f["net_cf"]
        if isinstance(oid, str) and oid.startswith("multi:"):
            fallback_n += 1
            fallback_cf += net
            continue
        if eff == "open" and qty > 0:
            lots[oid].append([qty, net, f["time"], f.get("order_id")])
        elif eff == "close" and qty > 0:
            remaining = qty
            close_per = net / qty if qty else 0.0
            while remaining > 1e-9 and lots[oid]:
                lot = lots[oid][0]
                lot_qty, lot_cost, lot_time, lot_oid = lot
                take = min(remaining, lot_qty)
                open_per = lot_cost / lot_qty if lot_qty else 0.0
                trips.append({"open": lot_time, "close": f["time"], "chain": f.get("chain"),
                              "strike": f.get("strike"), "cp": f.get("cp"), "qty": take,
                              "pnl": take * (open_per + close_per),
                              "holdSec": (f["time"] - lot_time).total_seconds(),
                              "expiry": f.get("expiry"), "option_id": oid,
                              "open_order_id": lot_oid, "close_order_id": f.get("order_id")})
                lot_qty -= take
                lot[0], lot[1] = lot_qty, lot_cost - open_per * take
                remaining -= take
                if lot_qty <= 1e-9:
                    lots[oid].popleft()
            if remaining > 1e-9:
                unmatched_n += 1
                unmatched_cf += close_per * remaining
    stats = {"unmatchedCloses": unmatched_n, "unmatchedCloseCF$": round(unmatched_cf, 2),
             "orderLevelFallback": fallback_n, "fallbackCF$": round(fallback_cf, 2)}
    return trips, stats


def _round_trips(fills: list) -> list:
    """Back-compat wrapper (trips only)."""
    trips, _ = _round_trips_full(fills)
    return trips


@mcp.tool()
async def daily_pnl_curve(date: Optional[str] = None, target: Optional[float] = None,
                          full: bool = False) -> dict:
    """Reconstruct today's realized P&L trade-by-trade from your Robinhood option fills.

    Builds the running cumulative-P&L curve (net of fees), marks the moment you crossed your daily
    target, and quantifies what happened AFTER that point - the single number your logged history says
    costs you money. `date` (YYYY-MM-DD ET) defaults to today; `full=True` returns every fill row.
    """
    d = date or _today_et().isoformat()
    tgt = float(target) if target is not None else _target()
    try:
        fills, fmeta = await _day_fills(d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not fills:
        out0 = {"date": d, "note": "No filled option orders for this date.", "realized$": 0.0,
                "orders": 0}
        if fmeta.get("truncated"):
            out0["dataWarning"] = "Order history hit the page cap before this date; older fills may be missing."
        return out0
    trips, tstats = _round_trips_full(fills)   # B2: multi-leg spreads/rolls now matched
    cash = round(sum(f["net_cf"] for f in fills), 2)
    data_warnings = _fill_data_warnings(fmeta, tstats)
    if not trips:
        out1 = {"date": d, "orders": len(fills), "netCashFlow$": cash,
                "note": "No completed round trips - positions may still be open or expired by assignment."}
        if data_warnings:
            out1["dataWarnings"] = data_warnings
        return out1
    cur = _build_curve(trips, tgt)
    by_chain = {}
    for tr in trips:
        by_chain[tr["chain"]] = round(by_chain.get(tr["chain"], 0.0) + tr["pnl"], 2)
    after = (round(cur["total"] - cur["crossCum"], 2) if cur["crossCum"] is not None else None)
    out = {"date": d, "target$": round(tgt, 2), "realized$": round(cur["total"], 2),
           "orders": len(fills), "roundTrips": len(trips),
           "peak$": round(cur["peak"], 2), "maxDrawdownFromPeak$": round(cur["maxDrawdownFromPeak"], 2),
           "byUnderlying$": by_chain,
           "firstFill": _fmt_et(fills[0]["time"]), "lastFill": _fmt_et(fills[-1]["time"])}
    if abs(cash - cur["total"]) > 1.0:
        out["netCashFlow$"] = cash
        out["reconNote"] = ("Net cash flow differs from round-trip realized P&L. Usual causes: a "
                            "position opened on an earlier date and closed today (or opened today and "
                            "still open), a same-day expiring/assigned leg, or an unmatched close - "
                            "see reconStats.")
        out["reconStats"] = tstats
    if data_warnings:
        out["dataWarnings"] = data_warnings
    if cur["crossIdx"] is not None:
        out["targetCross"] = {"time": _fmt_et(cur["crossTime"]),
                              "tradesAfter": len(trips) - 1 - cur["crossIdx"],
                              "pnlSinceTarget$": after}
        if after is not None and after < 0:
            out["leak"] = (f"You hit ${tgt:.0f} at {_fmt_et(cur['crossTime'])}, then gave back "
                           f"${abs(after):.2f} over {out['targetCross']['tradesAfter']} more trades. "
                           f"This is the pattern - stopping at target would have left you ${cur['crossCum']:.2f}.")
    out["curve"] = cur["rows"] if full else cur["rows"][-12:]
    return out


@mcp.tool()
async def daily_review(date: Optional[str] = None, target: Optional[float] = None) -> dict:
    """End-of-day scorecard from your Robinhood fills: win rate, expectancy, P&L by hour, and the
    killer split - performance BEFORE vs AFTER you hit your daily target.

    Pairs opening and closing fills (FIFO) into round trips to compute per-trade stats. `date`
    (YYYY-MM-DD ET) defaults to today.
    """
    d = date or _today_et().isoformat()
    tgt = float(target) if target is not None else _target()
    try:
        fills, fmeta = await _day_fills(d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    trips, tstats = _round_trips_full(fills)
    data_warnings = _fill_data_warnings(fmeta, tstats)
    if not trips:
        out0 = {"date": d, "note": "No completed round trips for this date.",
                "orders": len(fills)}
        if data_warnings:
            out0["dataWarnings"] = data_warnings
        return out0
    cur = _build_curve(trips, tgt)
    cross_time = cur["crossTime"]
    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] < 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    total = sum(t["pnl"] for t in trips)
    n = len(trips)

    def _hhmm(secs):
        return f"{int(secs // 60)}m{int(secs % 60):02d}s"

    by_hour = {}
    for t in trips:
        hr = t["close"].astimezone(ET).strftime("%H:00")
        h = by_hour.setdefault(hr, {"pnl": 0.0, "n": 0})
        h["pnl"] += t["pnl"]; h["n"] += 1
    by_hour = {k: {"pnl$": round(v["pnl"], 2), "trades": v["n"]} for k, v in sorted(by_hour.items())}

    out = {"date": d, "target$": round(tgt, 2), "realized$": round(total, 2), "roundTrips": n,
           "winRate%": round(100.0 * len(wins) / n, 1),
           "avgWin$": round(gross_win / len(wins), 2) if wins else 0.0,
           "avgLoss$": round(-gross_loss / len(losses), 2) if losses else 0.0,
           "profitFactor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
           "expectancyPerTrade$": round(total / n, 2),
           "largestWin$": round(max((t["pnl"] for t in trips), default=0.0), 2),
           "largestLoss$": round(min((t["pnl"] for t in trips), default=0.0), 2),
           "avgHoldTime": _hhmm(sum(t["holdSec"] for t in trips) / n),
           "pnlByHour": by_hour}
    if cross_time is not None:
        before = [t for t in trips if t["close"] <= cross_time]
        after = [t for t in trips if t["close"] > cross_time]
        def _blk(ts):
            if not ts:
                return {"trades": 0, "pnl$": 0.0, "winRate%": None}
            w = sum(1 for t in ts if t["pnl"] > 0)
            return {"trades": len(ts), "pnl$": round(sum(t["pnl"] for t in ts), 2),
                    "winRate%": round(100.0 * w / len(ts), 1)}
        out["beforeVsAfterTarget"] = {"targetHitAt": _fmt_et(cross_time),
                                      "beforeTarget": _blk(before), "afterTarget": _blk(after)}
        a = out["beforeVsAfterTarget"]["afterTarget"]
        if a["trades"] and a["pnl$"] < 0:
            out["verdict"] = (f"After hitting target you took {a['trades']} more trades for "
                              f"${a['pnl$']:.2f} ({a['winRate%']}% win). The discipline rule pays you "
                              f"${abs(a['pnl$']):.2f}/session here.")
    else:
        out["note2"] = "Target not reached this session."
    if data_warnings:
        out["dataWarnings"] = data_warnings
    return out


@mcp.tool()
async def should_i_trade(date: Optional[str] = None, target: Optional[float] = None,
                         strict: bool = False) -> dict:
    """Real-time GO / CAUTION / STOP gate before your next 0DTE entry.

    Combines past-target status, give-back from your intraday peak, consecutive losses, rapid re-entry
    (churning), and time-of-session into one call. This is your agreed-on target procedure, made
    queryable mid-session. Time-based signals assume `date` is today (the default).
    """
    d = date or _today_et().isoformat()
    tgt = float(target) if target is not None else _target()
    giveback_frac = float(_cfg("giveback_frac"))
    rapid_secs = float(_cfg("rapid_reentry_secs"))
    late_et = str(_cfg("late_session_et"))
    is_today = (d == _today_et().isoformat())
    try:
        fills = await _day_fills(d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    now = _dt.datetime.now(ET)
    trips, tstats = _round_trips_full(fills)
    cur = _build_curve(trips, tgt)
    total = cur["total"]
    peak = cur["peak"]
    reasons, flags = [], []

    # B2: if the fill picture is incomplete, the discipline math (realized$, peak, streaks) can't
    # be trusted -- so this gate must never clear you to GO on partial data.
    data_warnings = _fill_data_warnings(fmeta, tstats)
    data_incomplete = bool(data_warnings)
    # (#5) Optional integrity cross-check: reconcile the gate's round-trip realized against the
    # independent recon reconstruction; a >$1 disagreement means the two views of the tape differ,
    # so the gate must not clear you. Off by default because it re-pages RH (latency on the hot path).
    if strict and is_today:
        import asyncio
        try:
            _recon = await asyncio.to_thread(_rh_realized_recon_sync, d)
            _rr = _recon.get("roundTripRealized$")
            if _rr is not None and abs(_rr - round(total, 2)) > 1.0:
                data_incomplete = True
                data_warnings.append(f"Reconstruction cross-check disagrees: gate ${total:.2f} vs "
                                     f"recon ${_rr:.2f} - treating inputs as incomplete.")
        except Exception:  # noqa: BLE001
            pass

    past_target = total >= tgt and tgt > 0
    giveback = peak >= tgt and (peak - total) >= giveback_frac * tgt
    last3 = trips[-3:]
    consec_losses = len(last3) >= 3 and all(t["pnl"] < 0 for t in last3)
    consec2 = len(trips) >= 2 and all(t["pnl"] < 0 for t in trips[-2:])
    deep_dd = total <= -0.5 * tgt
    # rapid re-entry: tight gaps between the last few DISTINCT opening ORDERS. Dedupe by order_id
    # (B2) so the several legs of one spread -- which fill at the same instant -- don't look like
    # churning and trip a false RAPID_REENTRY.
    open_times = {}
    for f in fills:
        if f.get("effect") == "open":
            oid = f.get("order_id") or id(f)
            t_ = f["time"]
            if oid not in open_times or t_ < open_times[oid]:
                open_times[oid] = t_
    opens = sorted(open_times.values())
    rapid = False
    if len(opens) >= 3:
        gaps = [(opens[i] - opens[i - 1]).total_seconds() for i in range(-2, 0)]
        rapid = all(g < rapid_secs for g in gaps)
    # late-session gate keyed to the ACTUAL session close (B8): the configured cutoff, or 15
    # minutes before an early close, whichever is earlier -- and only while the market is open.
    _dd = _dt.date.fromisoformat(d)
    close_dt = _dt.datetime.combine(_dd, _session_close_et(_dd), tzinfo=ET)
    lh, lm = (int(x) for x in late_et.split(":"))
    cfg_cut = _dt.datetime.combine(_dd, _dt.time(lh, lm), tzinfo=ET)
    gate_dt = min(cfg_cut, close_dt - _dt.timedelta(minutes=15))
    late = is_today and now >= gate_dt and now < close_dt

    if past_target:
        flags.append("PAST_TARGET")
        reasons.append(f"You're at ${total:.2f} vs ${tgt:.0f} target. Post-target trades are your "
                       f"documented main source of losses.")
    if giveback:
        flags.append("GIVING_BACK")
        reasons.append(f"You peaked at ${peak:.2f} and are now ${total:.2f} - given back "
                       f"${peak - total:.2f} from the high.")
    if consec_losses:
        flags.append("CONSEC_LOSSES")
        reasons.append("Last 3 round trips were all losers - classic tilt setup.")
    if rapid:
        flags.append("RAPID_REENTRY")
        reasons.append(f"Last entries were <{int(rapid_secs)}s apart - you're churning, not "
                       f"waiting for setups.")
    if deep_dd:
        flags.append("DEEP_DRAWDOWN")
        reasons.append(f"Down ${abs(total):.2f} (>0.5x target) - high revenge-sizing risk.")
    if late:
        flags.append("LATE_SESSION")
        reasons.append(f"It's {now.strftime('%H:%M')} ET - final-stretch 0DTE gamma/pin risk into the bell.")
    if data_incomplete:
        flags.append("DATA_INCOMPLETE")
        reasons.extend(data_warnings)

    if past_target or consec_losses or giveback:
        verdict = "STOP"
    elif data_incomplete:
        verdict = "UNKNOWN"     # B2: partial fills -> cannot clear you to trade
    elif late or rapid or deep_dd or consec2:
        verdict = "CAUTION"
    else:
        verdict = "GO"
    if not reasons:
        reasons.append("No discipline flags: within target, no tilt signals, normal pacing.")
    out = {"date": d, "verdict": verdict, "flags": flags, "reasons": reasons,
           "realized$": round(total, 2), "peak$": round(peak, 2), "target$": round(tgt, 2),
           "roundTrips": len(trips), "asof": now.strftime("%H:%M:%S ET") if is_today else "EOD review"}
    if data_incomplete:
        out["dataWarnings"] = data_warnings
    return out


# ============================================================================
# 0DTE DECISION SUPPORT (chain-derived)
# ============================================================================
def _atm_iv(spot: float, opts: list):
    from collections import defaultdict
    by = defaultdict(dict)
    for o in opts:
        if o.get("iv", 0) > 0:
            by[o["strike"]][o["cp"]] = o["iv"]
    for k in sorted(by.keys(), key=lambda x: abs(x - spot)):
        cp = by[k]
        if "C" in cp and "P" in cp:
            return k, (cp["C"] + cp["P"]) / 2.0
        if "C" in cp or "P" in cp:
            return k, list(cp.values())[0]
    return None, None


def _em_levels(spot: float, opts: list, expiry: Optional[str]) -> dict:
    em = _expected_move(spot, opts)
    atm_k, atm_iv = _atm_iv(spot, opts)
    iv_em = None
    if atm_iv and expiry:
        T = _year_frac(expiry)
        iv_em = spot * atm_iv * (T ** 0.5)
    pts = (em or {}).get("expectedMovePts") or iv_em or 0.0
    out = {"atmStrike": (em or {}).get("atmStrike", atm_k),
           "straddle$": (em or {}).get("straddle"),
           "expectedMovePts": round(pts, 1) if pts else None,
           "expectedMovePct": round(100.0 * pts / spot, 2) if pts else None,
           "atmIV": round(atm_iv, 4) if atm_iv else None,
           "ivBasedMovePts": round(iv_em, 1) if iv_em else None}
    if pts:
        out["levels"] = {"upper1sigma": round(spot + pts, 2), "lower1sigma": round(spot - pts, 2),
                         "upper2sigma": round(spot + 2 * pts, 2), "lower2sigma": round(spot - 2 * pts, 2)}
    return out


@mcp.tool()
async def expected_move(expiration: Optional[str] = None, zero_dte: bool = True,
                        root: str = "SPXW") -> dict:
    """Today's implied trading range from the ATM straddle - the single most useful number for 0DTE
    strike selection.

    Returns the ATM straddle (~1-sigma move for the session), the IV-based 1-sigma, and the +/-1 and
    +/-2 sigma price levels. Defaults to today's SPXW expiry (`zero_dte=True`).
    """
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte,     # B3: RH-live primary, was CBOE-only
                                     expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if zero_dte else (expiration or _nearest_expiry(ch["options"], root))
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    resolved = "today"
    if not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        resolved = exp
    elif not zero_dte:
        resolved = exp
    if not opts:
        return {"error": "No contracts for that selection."}
    # B11: on a 0DTE ("today") request pass today's date -- not None -- so the IV-based 1-sigma
    # (ivBasedMovePts) still computes instead of silently disabling.
    exp_for_T = _today_et().isoformat() if resolved == "today" else resolved
    em = _em_levels(spot, opts, exp_for_T)
    return {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
            "asof": ch.get("asof"), "source": ch.get("source"), "freshness": ch.get("freshness"), **em}


@mcp.tool()
async def strike_probabilities(expiration: Optional[str] = None, zero_dte: bool = True,
                               root: str = "SPXW",
                               width_pct: Annotated[float, Field(ge=0.2, le=10.0)] = 2.0,
                               strikes: Optional[str] = None) -> dict:
    """Risk-neutral probability that each strike finishes in-the-money, plus an estimated probability
    of touching it before expiry - for sizing short-strike risk on 0DTE.

    Computes prob-ITM = N(d2) (calls) / N(-d2) (puts) from each strike's IV, and prob-touch ~= 2x the
    finish-OTM probability. Defaults to a grid within +/-`width_pct` of spot for today's SPXW expiry;
    pass `strikes` as a comma list to target specific ones.
    """
    try:
        ch = await _load_chain_smart(zero_dte=zero_dte,     # B3: RH-live primary, was CBOE-only
                                     expiration=(None if zero_dte else expiration), root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if zero_dte else (expiration or _nearest_expiry(ch["options"], root))
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
    resolved = "today"
    if not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        resolved = exp
    elif not zero_dte:
        resolved = exp
    if not opts:
        return {"error": "No contracts for that selection."}
    exp_for_T = _today_et().isoformat() if resolved == "today" else resolved
    T = _year_frac(exp_for_T)
    want = None
    if strikes:
        want = set()
        for s in strikes.split(","):
            try:
                want.add(round(float(s.strip()), 2))
            except ValueError:
                pass
    lo, hi = spot * (1 - width_pct / 100.0), spot * (1 + width_pct / 100.0)
    from collections import defaultdict
    by = defaultdict(dict)
    for o in opts:
        if o.get("iv", 0) > 0:
            by[o["strike"]][o["cp"]] = o
    rows = []
    for k in sorted(by.keys()):
        if want is not None:
            if round(k, 2) not in want:
                continue
        elif not (lo <= k <= hi):
            continue
        row = {"strike": k, "vsSpot": round(k - spot, 1)}
        sqrtT = T ** 0.5
        # one touch probability per strike (depends on strike vs spot, not call/put)
        ivk = None
        for side in ("C", "P"):
            if by[k].get(side):
                ivk = by[k][side]["iv"]
                break
        if ivk and abs(k - spot) > 1e-6:
            d2k = (np.log(spot / k) + (RISK_FREE - 0.5 * ivk * ivk) * T) / (ivk * sqrtT)
            p_beyond = float(_norm_cdf(np.array([d2k if k >= spot else -d2k]))[0])  # P(S_T past K)
            row["probTouch%"] = round(min(1.0, 2.0 * p_beyond) * 100.0, 1)
        elif abs(k - spot) <= 1e-6:
            row["probTouch%"] = 100.0     # spot is AT this strike -> already touched
        else:
            row["probTouch%"] = None      # B15: no IV for this strike -> unknown, not a bogus 100%
        for side in ("C", "P"):
            o = by[k].get(side)
            if not o:
                continue
            iv = o["iv"]
            d2 = (np.log(spot / k) + (RISK_FREE - 0.5 * iv * iv) * T) / (iv * sqrtT)
            p_itm = float(_norm_cdf(np.array([d2 if side == "C" else -d2]))[0])
            # (#9) BS delta from the strike's IV, not the feed's delta -- one code path, consistent
            # across RH/CBOE and immune to the B4 zero-delta feed case.
            d1 = d2 + iv * sqrtT
            bs_delta = float(_norm_cdf(np.array([d1]))[0]) - (1.0 if side == "P" else 0.0)
            row[side] = {"probITM%": round(100.0 * p_itm, 1),
                         "delta": round(bs_delta, 3), "iv": round(iv, 4)}
        rows.append(row)
    return {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
            "asof": ch.get("asof"), "source": ch.get("source"), "freshness": ch.get("freshness"),
            "strikes": rows}


@mcp.tool()
async def daily_game_plan(root: str = "SPXW") -> dict:
    """One call for today's 0DTE map: spot, expected-move bands, gamma regime + flip, call/put walls,
    high-OI pins, and max-pain - assembled into support/resistance you can trade against.

    Resistance = call wall / +sigma / high call OI; support = put wall / -sigma / high put OI; pivots =
    max-pain, gamma flip, spot. SPX pins toward max-pain and gamma walls into the close on long-gamma days.
    """
    try:
        ch = await _load_chain_smart(zero_dte=True, root=root)
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    opts = _filter(ch["options"], root=root, zero_dte=True)
    resolved, note = "today", None
    if not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        resolved = exp
        note = (f"*** NOT TODAY'S MAP *** Nothing expires today, so EVERY level below -- gamma "
                f"flip, call/put walls, max pain, expected move -- belongs to the {exp} expiry, "
                f"not today's. Do NOT trade 0DTE against these levels.")
    comp = _gex_components(spot, opts)
    if not comp:
        return {"error": "No valid contracts (need OI and IV)."}
    flip = _gamma_flip(spot, comp)
    call, put, net = _per_strike(comp)
    cw = max(({k: v for k, v in call.items() if k >= spot}).items(), key=lambda kv: kv[1], default=None)
    pw = max(({k: v for k, v in put.items() if k <= spot}).items(), key=lambda kv: kv[1], default=None)
    mp = _max_pain(opts)
    # B11: pass today's date on a 0DTE map so ivBasedMovePts still computes (was None -> disabled).
    em = _em_levels(spot, opts, _today_et().isoformat() if resolved == "today" else resolved)
    # high open-interest strikes by side
    oi_c = sorted(((o["strike"], o["oi"]) for o in opts if o["cp"] == "C" and o["oi"] > 0),
                  key=lambda x: x[1], reverse=True)[:3]
    oi_p = sorted(((o["strike"], o["oi"]) for o in opts if o["cp"] == "P" and o["oi"] > 0),
                  key=lambda x: x[1], reverse=True)[:3]
    total = comp["total"]
    em_lv = em.get("levels") or {}
    resistance = sorted({x for x in [cw[0] if cw else None, em_lv.get("upper1sigma"),
                                     oi_c[0][0] if oi_c else None] if x and x > spot}, )
    support = sorted({x for x in [pw[0] if pw else None, em_lv.get("lower1sigma"),
                                  oi_p[0][0] if oi_p else None] if x and x < spot}, reverse=True)
    out = {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2), "asof": ch.get("asof"),
           "source": ch.get("source"), "freshness": ch.get("freshness"),
           "regime": "long gamma (pin / mean-revert)" if total > 0 else "short gamma (trend / amplify)",
           "totalGEX_$mm_per_1pct": _mm(total),
           "gammaFlip": round(flip, 2) if flip else None,
           "expectedMove": {"pts": em.get("expectedMovePts"), "pct": em.get("expectedMovePct"),
                            "upper": em_lv.get("upper1sigma"), "lower": em_lv.get("lower1sigma")},
           "callWall": cw[0] if cw else None, "putWall": pw[0] if pw else None,
           "maxPain": mp,
           "highOI_calls": [{"strike": k, "oi": int(v)} for k, v in oi_c],
           "highOI_puts": [{"strike": k, "oi": int(v)} for k, v in oi_p],
           "map": {"resistance": resistance, "pivots": sorted({x for x in [mp, round(flip, 0) if flip else None,
                                                                           round(spot, 0)] if x}),
                   "support": support}}
    if note:
        out["note"] = note
    return out


# ============================================================================
# COVERED-CALL MANAGER + SINGLE-NAME EARNINGS CALENDAR + REGIME CLASSIFIER
# ============================================================================
_ETF_HINTS = {"SCHD", "TOPT", "VOO", "CHPY", "SPY", "QQQ", "IVV", "DIA", "IWM", "VTI"}


def _dte_days(expiry: Optional[str]) -> Optional[int]:
    if not expiry:
        return None
    try:
        return (_dt.date.fromisoformat(expiry[:10]) - _today_et()).days
    except Exception:  # noqa: BLE001
        return None


def _next_earnings_sync(symbol: str) -> Optional[dict]:
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        with _RH_SYNC_SEM:                      # (#11e) cap concurrent RH fan-out
            e = rh.stocks.get_earnings(symbol) or []
    except Exception:  # noqa: BLE001
        return None
    today = _today_et()
    best = None
    for x in e:
        rep = x.get("report") or {}
        ds = rep.get("date")
        if not ds:
            continue
        try:
            d = _dt.date.fromisoformat(ds[:10])
        except Exception:  # noqa: BLE001
            continue
        if d < today:
            continue
        if best is None or d < best[0]:
            best = (d, rep.get("timing"), rep.get("verified"), x.get("year"), x.get("quarter"))
    if best is None:
        return None
    d, timing, verified, yr, q = best
    tmap = {"am": "before open (BMO)", "pm": "after close (AMC)"}
    return {"date": d.isoformat(), "daysAway": (d - today).days,
            "session": tmap.get(timing, timing or "unknown"), "confirmed": bool(verified),
            "fiscal": (f"Q{q} {yr}" if q and yr else None)}


async def _next_earnings(symbol: str) -> Optional[dict]:
    import asyncio
    return await asyncio.to_thread(_next_earnings_sync, symbol)


@mcp.tool()
async def covered_call_manager(
    roll_delta: Annotated[float, Field(ge=0.1, le=0.9)] = 0.45,
    roll_dte: Annotated[int, Field(ge=0, le=60)] = 7,
) -> dict:
    """Manage your short (covered) calls: DTE, assignment probability (delta), premium captured vs
    extrinsic left, annualized yield, share-coverage check, earnings-before-expiry risk, and roll signals.

    Scans Robinhood option positions for short calls and matches them to your share holdings to confirm
    coverage. A roll is flagged when the call is ITM into expiry (DTE <= `roll_dte`), delta is deep, or
    most of the premium has already decayed. Assignment probability is approximated by the option delta.
    """
    import asyncio
    try:
        pos = await _robinhood_positions()
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    shares: dict = {}
    for p in pos:
        if p.get("type") == "equity":
            shares[p["symbol"]] = shares.get(p["symbol"], 0.0) + (p.get("qty") or 0.0)
    shorts = [p for p in pos if p.get("type") == "option" and p.get("cp") == "C"
              and (p.get("qty") or 0) < 0]
    if not shorts:
        return {"note": "No short call positions found on Robinhood.",
                "equityHoldings": sorted(shares.keys())}
    unders = sorted({p["underlying"] for p in shorts if p.get("underlying")})
    earn_list = await asyncio.gather(*[_next_earnings(u) for u in unders])
    earn = dict(zip(unders, earn_list))
    rows, tot_prem = [], 0.0
    for p in shorts:
        u = p.get("underlying")
        contracts = abs(p.get("qty") or 0.0)
        strike = p.get("strike")
        undpx = p.get("price")
        delta = p.get("delta")
        avg = p.get("avgPrice")
        mult = p.get("mult") or 100.0
        mark = p.get("mark")
        dte = _dte_days(p.get("expiry"))
        prem_total = (avg * contracts * mult) if (avg is not None) else None
        if prem_total:
            tot_prem += prem_total
        intrinsic = (max(0.0, undpx - strike) if (undpx is not None and strike is not None) else None)
        extrinsic = (max(0.0, mark - intrinsic) if (mark is not None and intrinsic is not None) else None)
        assign_prob = (abs(delta) * 100.0) if delta is not None else None
        is_itm = (undpx is not None and strike is not None and undpx > strike)
        covered = shares.get(u, 0.0) >= 100.0 * contracts
        ann_yield = ((avg / undpx) * (365.0 / dte) * 100.0
                     if (avg is not None and undpx and dte and dte > 0) else None)
        # management / roll signal (priority order)
        if is_itm and dte is not None and dte <= roll_dte:
            sig = "ITM into expiry - roll up/out to defend shares, or accept assignment"
        elif delta is not None and abs(delta) >= 0.70:
            sig = "deep ITM (high assignment prob) - roll out/up if keeping shares"
        elif avg and mark is not None and mark <= 0.20 * avg and (dte is None or dte > roll_dte):
            sig = "most premium captured - consider buy-to-close to free shares / re-strike"
        elif dte is not None and dte <= roll_dte and assign_prob is not None and assign_prob < 30:
            sig = "low assignment risk near expiry - let expire, then re-write"
        elif delta is not None and abs(delta) >= roll_delta and dte is not None and dte <= max(roll_dte * 2, 14):
            sig = "approaching roll zone - watch delta/DTE"
        else:
            sig = "hold; manage at thresholds"
        e = earn.get(u)
        earn_flag = None
        if e and p.get("expiry"):
            try:
                ed = _dt.date.fromisoformat(e["date"])
                if _today_et() <= ed <= _dt.date.fromisoformat(p["expiry"][:10]):
                    earn_flag = f"earnings {e['date']} ({e['session']}) BEFORE expiry - gap/assignment risk"
            except Exception:  # noqa: BLE001
                pass
        rows.append({
            "underlying": u, "contracts": int(contracts), "strike": strike, "expiry": p.get("expiry"),
            "dte": dte, "underlyingPx": round(undpx, 2) if undpx else None,
            "moneyness": "ITM" if is_itm else "OTM",
            "assignmentProb%": round(assign_prob, 1) if assign_prob is not None else None,
            "delta": round(delta, 3) if delta is not None else None,
            "premiumCaptured$": round(prem_total, 2) if prem_total is not None else None,
            "mark$/sh": round(mark, 2) if mark is not None else None,
            "extrinsicLeft$/sh": round(extrinsic, 2) if extrinsic is not None else None,
            "annYieldOnPremium%": round(ann_yield, 1) if ann_yield is not None else None,
            "covered": covered, "sharesHeld": int(shares.get(u, 0.0)),
            "signal": sig, "earningsRisk": earn_flag,
        })
    rows.sort(key=lambda r: (r["dte"] if r["dte"] is not None else 9999))
    out = {"shortCalls": len(shorts), "totalPremiumCaptured$": round(tot_prem, 2),
           "rollThresholds": {"delta": roll_delta, "dte": roll_dte}, "positions": rows}
    if any((r["dte"] is not None and r["dte"] <= 2 and r["annYieldOnPremium%"] is not None)
           for r in rows):
        out["yieldNote"] = ("annYieldOnPremium% annualizes by 365/DTE, so it balloons on very "
                            "short-dated calls (<=2 DTE) - read it as a comparison ratio, not an "
                            "achievable annual return.")     # B15
    return out


@mcp.tool()
async def earnings_calendar(symbols: Optional[str] = None,
                            days: Annotated[int, Field(ge=1, le=400)] = 90) -> dict:
    """Next single-name earnings dates for your holdings (or a provided symbol list), sorted by
    proximity, with the report session (BMO/AMC), days away, and whether it falls within `days`.

    Earnings are binary-risk events; this flags which positions report soon. Holdings with no earnings
    (ETFs/funds) are listed separately. Dates come from Robinhood's earnings data (the upcoming report
    is the entry with no actual EPS yet).
    """
    import asyncio
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        try:
            pos = await _robinhood_positions()
        except EdgeError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        syms = sorted({p["symbol"] for p in pos if p.get("type") == "equity"})
    if not syms:
        return {"note": "No equity symbols to check."}
    res = await asyncio.gather(*[_next_earnings(s) for s in syms])
    upcoming, none_found = [], []
    today = _today_et()
    for s, e in zip(syms, res):
        if e is None:
            none_found.append(s)
        else:
            upcoming.append({"symbol": s, **e, "withinWindow": e["daysAway"] <= days})
    upcoming.sort(key=lambda r: r["daysAway"])
    within = [r for r in upcoming if r["withinWindow"]]
    return {"windowDays": days, "asof": today.isoformat(),
            "withinWindowCount": len(within), "nextEarnings": upcoming,
            "noEarnings_ETFsFunds": none_found}


def _score_band(value: float, bands: list) -> tuple:
    """bands: ascending [(upper_threshold, score, label), ...]; first whose threshold >= value wins."""
    for thr, sc, lab in bands:
        if value <= thr:
            return sc, lab
    return bands[-1][1], bands[-1][2]


@mcp.tool()
async def regime_classifier() -> dict:
    """One-call market-regime read for 0DTE posture: folds VIX level + VIX term structure + financial
    conditions (NFCI) + credit spreads (HY OAS) + the yield curve + the Sahm rule into a single
    risk-on / constructive / neutral / caution / risk-off score with a suggested trading posture.

    Combines your existing siloed CBOE/FRED gauges into one verdict. VIX & term structure are ~15-min
    delayed; the FRED series update daily/weekly.
    """
    import asyncio
    vix, v9, v3m = await asyncio.gather(_vol_index("VIX"), _vol_index("VIX9D"), _vol_index("VIX3M"))
    nfci, hy, c2s10, sahm = await asyncio.gather(
        asyncio.to_thread(_latest_obs, "NFCI"),
        asyncio.to_thread(_latest_obs, "BAMLH0A0HYM2"),
        asyncio.to_thread(_latest_obs, "T10Y2Y"),
        asyncio.to_thread(_latest_obs, "SAHMREALTIME"),
    )
    factors, score, n = [], 0, 0

    def add(name, val, sc, lab):
        nonlocal score, n
        if sc is not None:
            score += sc
            n += 1
        factors.append({"factor": name, "value": val, "score": sc, "read": lab})

    if vix is not None:
        sc, lab = _score_band(vix, [(13, 1, "calm"), (20, 0, "normal"), (30, -1, "elevated"),
                                    (1e9, -2, "stress")])
        add("VIX", round(vix, 2), sc, lab)
    else:
        add("VIX", None, None, "unavailable")

    if v9 and v3m and v3m > 0:
        ratio = v9 / v3m
        if ratio <= 0.95:
            sc, lab = 1, "steep contango (calm)"
        elif ratio <= 1.0:
            sc, lab = 0, "contango"
        elif ratio <= 1.05:
            sc, lab = -1, "mild backwardation (stress)"
        else:
            sc, lab = -2, "deep backwardation (acute stress)"
        add("VIX term 9D/3M", round(ratio, 3), sc, lab)
    else:
        add("VIX term 9D/3M", None, None, "unavailable")

    if nfci is not None:
        sc, lab = _score_band(nfci[1], [(-0.2, 1, "loose"), (0.2, 0, "neutral"), (0.5, -1, "tight"),
                                        (1e9, -2, "very tight")])
        add("NFCI (financial conditions)", round(nfci[1], 3), sc, lab)
    else:
        add("NFCI (financial conditions)", None, None, "unavailable")

    if hy is not None:
        sc, lab = _score_band(hy[1], [(3.5, 1, "tight credit"), (5.0, 0, "normal credit"),
                                      (7.0, -1, "wide credit"), (1e9, -2, "stressed credit")])
        add("HY OAS %", round(hy[1], 2), sc, lab)
    else:
        add("HY OAS %", None, None, "unavailable")

    if c2s10 is not None:
        if c2s10[1] < 0:
            sc, lab = -1, "inverted (late-cycle caution)"
        else:
            sc, lab = 0, "positive slope"
        add("2s10s curve %", round(c2s10[1], 2), sc, lab)
    else:
        add("2s10s curve %", None, None, "unavailable")

    if sahm is not None:
        if sahm[1] < 0.3:
            sc, lab = 0, "no recession signal"
        elif sahm[1] < 0.5:
            sc, lab = -1, "warming (watch)"
        else:
            sc, lab = -2, "recession trigger (>=0.5)"
        add("Sahm rule", round(sahm[1], 2), sc, lab)
    else:
        add("Sahm rule", None, None, "unavailable")

    if score >= 3:
        regime = "RISK-ON"
    elif score >= 1:
        regime = "CONSTRUCTIVE"
    elif score >= -1:
        regime = "NEUTRAL"
    elif score >= -3:
        regime = "CAUTION"
    else:
        regime = "RISK-OFF / STRESS"
    posture = {
        "RISK-ON": "Calm/long-gamma backdrop: pinning & mean-reversion favored; fading extremes and premium-selling work; normal size.",
        "CONSTRUCTIVE": "Mostly calm: lean range/mean-revert but keep stops; normal-to-slightly-reduced size.",
        "NEUTRAL": "Mixed signals: trade levels both ways, no strong edge; standard risk.",
        "CAUTION": "Stress building (vol/credit): expect trend and gamma flips; cut size, respect levels, avoid fading strength/weakness.",
        "RISK-OFF / STRESS": "Acute stress: large ranges, negative gamma; trade small or stand aside, no counter-trend fades.",
    }[regime]
    return {"asof": "CBOE ~15-min + FRED daily", "regime": regime, "compositeScore": score,
            "factorsScored": n, "avgFactor": round(score / n, 2) if n else 0.0,
            "posture": posture, "factors": factors}


# ============================================================================
# USER CONFIG (goals / discipline tunables): JSON file, env overrides, live reload
# ============================================================================
TE_CONFIG_PATH = os.environ.get(
    "TE_CONFIG_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))

_CONFIG_DEFAULTS = {
    "daily_target": 524.0,        # $ profit target per trading day
    "weekly_target": None,        # optional $ weekly target (informational)
    "giveback_frac": 0.40,        # give-back from intraday peak (x target) -> STOP
    "rapid_reentry_secs": 90.0,   # entries closer than this flag churning
    "late_session_et": "15:45",   # final-stretch CAUTION after this ET time
    "max_trades_per_day": None,   # optional round-trip cap (informational)
    "roll_delta": 0.45,           # covered-call roll trigger: |delta| >=
    "roll_dte": 7,                # covered-call roll trigger: DTE <=
}
_CONFIG_ENV = {
    "daily_target": "DAILY_TARGET",
    "giveback_frac": "TE_GIVEBACK_FRAC",
    "rapid_reentry_secs": "TE_RAPID_REENTRY_SECS",
    "late_session_et": "TE_LATE_SESSION_ET",
}
_config_cache = {"mtime": None, "data": {}}


def _load_config() -> dict:
    """Read config.json, cached by mtime. Returns {} if missing/invalid."""
    import json
    try:
        st = os.stat(TE_CONFIG_PATH)
    except OSError:
        _config_cache["mtime"], _config_cache["data"] = None, {}
        return {}
    if _config_cache["mtime"] == st.st_mtime:
        return _config_cache["data"]
    try:
        with open(TE_CONFIG_PATH) as f:
            data = json.load(f)
        data = data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        data = {}
    _config_cache["mtime"], _config_cache["data"] = st.st_mtime, data
    return data


def _coerce(raw, base):
    if isinstance(base, bool):
        return str(raw).lower() in ("1", "true", "yes", "on")
    try:
        if isinstance(base, float):
            return float(raw)
        if isinstance(base, int):
            return int(raw)
    except (TypeError, ValueError):
        return base
    return raw


def _cfg(key, default=None):
    """Resolve a setting. Precedence: env var > config.json > built-in default."""
    base = _CONFIG_DEFAULTS.get(key, default)
    env = _CONFIG_ENV.get(key)
    if env:
        raw = os.environ.get(env)
        if raw not in (None, ""):
            return _coerce(raw, base if base is not None else default)
    cfg = _load_config()
    if cfg.get(key) is not None:
        return cfg[key]
    return base


def _cfg_source(key) -> str:
    env = _CONFIG_ENV.get(key)
    if env and os.environ.get(env) not in (None, ""):
        return f"env:{env}"
    if _load_config().get(key) is not None:
        return "config.json"
    return "default"


def _target() -> float:
    try:
        return float(_cfg("daily_target", 524.0))
    except (TypeError, ValueError):
        return 524.0


@mcp.tool()
async def trading_config(action: str = "show", key: Optional[str] = None,
                         value: Optional[str] = None) -> dict:
    """View or change your Traders Edge goals/discipline settings (config.json), live - no restart.

    `action='show'` (default) lists every setting, its effective value, and source (env / config /
    default). `action='set'` with `key` and `value` writes to config.json (e.g. key='daily_target',
    value='550'). `action='reset'` with `key` removes it. Env vars, if set, always win over the file.
    Editable keys: daily_target, weekly_target, giveback_frac, rapid_reentry_secs, late_session_et,
    max_trades_per_day, roll_delta, roll_dte.
    """
    import json
    keys = list(_CONFIG_DEFAULTS.keys())
    if action == "show":
        eff = {k: {"value": _cfg(k), "default": _CONFIG_DEFAULTS[k], "source": _cfg_source(k)}
               for k in keys}
        return {"configFile": TE_CONFIG_PATH, "exists": os.path.exists(TE_CONFIG_PATH),
                "settings": eff}
    if action in ("set", "reset"):
        if not key or key not in _CONFIG_DEFAULTS:
            return {"error": f"Unknown key '{key}'. Editable keys: {keys}"}
        cfg = dict(_load_config())
        note = None
        if action == "set":
            if value is None:
                return {"error": "Provide `value` to set."}
            base = _CONFIG_DEFAULTS[key]
            cfg[key] = (None if str(value).lower() in ("null", "none", "")
                        else _coerce(value, base if base is not None else 0.0))
            if _CONFIG_ENV.get(key) and os.environ.get(_CONFIG_ENV[key]) not in (None, ""):
                note = (f"Note: env var {_CONFIG_ENV[key]} is set and overrides config.json for this "
                        f"key until unset.")
        else:
            cfg.pop(key, None)
        try:
            os.makedirs(os.path.dirname(TE_CONFIG_PATH), exist_ok=True)
            with open(TE_CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
            _config_cache["mtime"] = None
        except OSError as exc:
            return {"error": f"Could not write config: {exc}"}
        out = {"action": action, "key": key, "newEffectiveValue": _cfg(key),
               "source": _cfg_source(key), "configFile": TE_CONFIG_PATH}
        if note:
            out["note"] = note
        return out
    return {"error": "action must be 'show', 'set', or 'reset'."}


# ============================================================================
# DISCIPLINE BACKTEST - replay fills through the stop-at-target rule
# ============================================================================
_WD_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@mcp.tool()
async def discipline_backtest(lookback_days: Annotated[int, Field(ge=5, le=400)] = 90,
                              target: Optional[float] = None) -> dict:
    """Replay your real fills through the stop-at-target rule: how much would stopping each day at your
    target have changed realized P&L vs what you actually did?

    For every trading day in the window it rebuilds round trips, finds the target cross, and compares
    actual P&L to 'stopped at target' P&L. Aggregates the after-target leak, win rate, expectancy, an
    equity curve, and by-day-of-week / by-hour breakdowns. The headline is the dollar value of discipline.
    """
    import asyncio
    from collections import defaultdict
    tgt = float(target) if target is not None else _target()
    stop = _today_et() - _dt.timedelta(days=lookback_days)
    stop_iso = stop.isoformat()
    try:
        fills_by_day, ometa = await _fills_window(stop, _today_et())   # (#3) DB-first
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not fills_by_day:
        return {"note": f"No fills in the last {lookback_days} days.", "target$": tgt}

    days, equity, all_trips = [], [], []
    dow = defaultdict(lambda: {"pnl": 0.0, "days": 0})
    hour = defaultdict(lambda: {"pnl": 0.0, "n": 0})
    cum_actual = 0.0
    for date in sorted(fills_by_day.keys()):
        fills = sorted(fills_by_day[date], key=lambda r: r["time"])
        trips = _round_trips(fills)
        if not trips:
            continue
        all_trips.extend(trips)
        cur = _build_curve(trips, tgt)
        actual = cur["total"]
        crossed = cur["crossIdx"] is not None
        hypo = cur["crossCum"] if crossed else actual
        after = (actual - cur["crossCum"]) if crossed else 0.0
        cum_actual += actual
        equity.append({"date": date, "pnl$": round(actual, 2), "cum$": round(cum_actual, 2)})
        wd = _dt.date.fromisoformat(date).strftime("%a")
        dow[wd]["pnl"] += actual
        dow[wd]["days"] += 1
        for tr in trips:
            h = tr["close"].astimezone(ET).strftime("%H:00")
            hour[h]["pnl"] += tr["pnl"]
            hour[h]["n"] += 1
        days.append({"date": date, "actual$": round(actual, 2), "hitTarget": crossed,
                     "targetTime": (_fmt_et(cur["crossTime"]) if crossed else None),
                     "afterTarget$": round(after, 2), "stopAtTarget$": round(hypo, 2),
                     "trips": len(trips)})

    actual_total = sum(d["actual$"] for d in days)
    hypo_total = sum(d["stopAtTarget$"] for d in days)
    hit = [d for d in days if d["hitTarget"]]
    gave_back = [d for d in hit if d["afterTarget$"] < 0]
    added = [d for d in hit if d["afterTarget$"] > 0]
    wins = [t for t in all_trips if t["pnl"] > 0]
    losses = [t for t in all_trips if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    n = len(all_trips)
    delta = hypo_total - actual_total
    out = {
        "window": f"{stop_iso} -> {_today_et().isoformat()} ({lookback_days}d)",
        "target$": round(tgt, 2), "tradingDays": len(days),
        "actualRealized$": round(actual_total, 2),
        "stopAtTargetRealized$": round(hypo_total, 2),
        "disciplineDelta$": round(delta, 2),
        "verdict": (f"Stopping at ${tgt:.0f} each day would have changed realized P&L by ${delta:+.2f} "
                    f"over {len(days)} days "
                    f"({'discipline wins' if delta > 0 else 'after-target trades were net positive here'})."),
        "daysHitTarget": len(hit), "daysContinuedAndLost": len(gave_back),
        "daysContinuedAndGained": len(added),
        "afterTargetNet$": round(sum(d["afterTarget$"] for d in hit), 2),
        "afterTargetLeakLosingDays$": round(sum(d["afterTarget$"] for d in gave_back), 2),
        "roundTrips": n, "winRate%": round(100.0 * len(wins) / n, 1) if n else None,
        "expectancyPerTrade$": round((gw - gl) / n, 2) if n else None,
        "profitFactor": round(gw / gl, 2) if gl > 0 else None,
        "bestDay": max(days, key=lambda d: d["actual$"]) if days else None,
        "worstDay": min(days, key=lambda d: d["actual$"]) if days else None,
        "byDayOfWeek": {k: {"pnl$": round(dow[k]["pnl"], 2), "days": dow[k]["days"],
                            "avg$": round(dow[k]["pnl"] / dow[k]["days"], 2)}
                        for k in _WD_ORDER if k in dow},
        "byHour": {k: {"pnl$": round(hour[k]["pnl"], 2), "trades": hour[k]["n"]}
                   for k in sorted(hour)},
        "equityCurve": equity,
    }
    if ometa.get("truncated"):
        out["dataWarning"] = (f"Order history hit the page cap (~{ometa.get('pages')} pages); this "
                              f"window may not reach the full {lookback_days} days. Oldest seen: "
                              f"{ometa.get('oldestSeen')}.")
    return out


# ============================================================================
# TAX SUMMARY - realized options P&L + wash-sale candidates (CPA hand-off)
# ============================================================================
@mcp.tool()
async def tax_summary(year: Optional[int] = None) -> dict:
    """Realized options P&L for the year plus wash-sale candidates - a hand-off for your CPA.

    Reconstructs closed round trips (FIFO) from your Robinhood option fills: total realized, short-term
    vs long-term, by month, gross gains/losses. Flags identical-contract wash-sale candidates (a
    realized loss where the same contract was re-opened within 30 days after the loss). Options only;
    not tax advice - verify against your 1099-B.
    """
    import asyncio
    from collections import defaultdict
    yr = int(year) if year else _today_et().year
    jan1 = _dt.date(yr, 1, 1)
    try:
        orders, ometa = await asyncio.to_thread(_rh_recent_option_orders, jan1, 40)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    fills = []
    for o in orders:
        fills.extend(_order_to_fills(o))    # B2: include multi-leg spread/roll legs
    fills.sort(key=lambda r: r["time"])
    trips = _round_trips(fills)
    yr_trips = [t for t in trips if t["close"].astimezone(ET).year == yr]
    if not yr_trips:
        return {"year": yr, "note": "No closed option round trips this year."}
    wins = [t for t in yr_trips if t["pnl"] > 0]
    losses = [t for t in yr_trips if t["pnl"] < 0]
    st = [t for t in yr_trips if (t["close"] - t["open"]).days <= 365]
    lt = [t for t in yr_trips if (t["close"] - t["open"]).days > 365]
    by_month = defaultdict(lambda: {"pnl": 0.0, "n": 0})
    for t in yr_trips:
        m = t["close"].astimezone(ET).strftime("%Y-%m")
        by_month[m]["pnl"] += t["pnl"]
        by_month[m]["n"] += 1
    opens_by_contract = defaultdict(list)
    for f in fills:
        if f.get("effect") == "open" and f.get("option_id"):
            opens_by_contract[f["option_id"]].append(f["time"])
    wash = []
    for t in losses:
        oid, cd = t.get("option_id"), t["close"]
        for ot in opens_by_contract.get(oid, []):
            if cd < ot and (ot - cd).days <= 30:
                wash.append({"contract": f"{t['chain']} {t.get('strike')}{t.get('cp')} {t.get('expiry')}",
                             "lossCloseDate": cd.astimezone(ET).date().isoformat(),
                             "loss$": round(t["pnl"], 2),
                             "reopenDate": ot.astimezone(ET).date().isoformat()})
                break
    out = {
        "year": yr, "asof": _today_et().isoformat(), "scope": "Robinhood options only (not tax advice)",
        "realizedTotal$": round(sum(t["pnl"] for t in yr_trips), 2),
        "shortTerm$": round(sum(t["pnl"] for t in st), 2), "shortTermTrips": len(st),
        "longTerm$": round(sum(t["pnl"] for t in lt), 2), "longTermTrips": len(lt),
        "roundTrips": len(yr_trips), "wins": len(wins), "losses": len(losses),
        "grossGains$": round(sum(t["pnl"] for t in wins), 2),
        "grossLosses$": round(sum(t["pnl"] for t in losses), 2),
        "byMonth": {k: {"pnl$": round(by_month[k]["pnl"], 2), "trips": by_month[k]["n"]}
                    for k in sorted(by_month)},
        "washSaleCandidates": wash,
        "washSaleNote": ("Identical-contract re-opens within 30 days after a realized loss. For 0DTE this "
                         "is usually rare (an expired contract can't be re-opened); a near-zero count is "
                         "expected. Different strikes/expiries may still warrant CPA review under the "
                         "substantially-identical rule. Stock lots are not included."),
    }
    if ometa.get("truncated"):
        out["WARNING_TRUNCATED"] = ("Order history hit the page cap before reaching Jan 1, so this "
                                    "year's realized P&L is INCOMPLETE - do not file from it as-is.")
        out["dataBeginsAt"] = ometa.get("oldestSeen")
    return out


# ============================================================================
# SNAPSHOT LOGGER (SQLite) - intraday GEX / level migration
# ============================================================================
TE_DB_PATH = os.environ.get("TE_DB_PATH",
                            os.path.join(os.path.expanduser("~"), ".trading", "traders_edge.db"))


def _db_conn():
    import sqlite3
    os.makedirs(os.path.dirname(TE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(TE_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        ts TEXT, date TEXT, spot REAL, total_gex REAL, gamma_flip REAL, call_wall REAL,
        put_wall REAL, max_pain REAL, expected_move REAL, vix REAL, vix1d REAL,
        regime TEXT, regime_score INTEGER)""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_snap_date ON snapshots(date)")
    # (#3) Local persistence of option fills + daily session scorecards, so weekly_review /
    # discipline_backtest / tax_summary can read history from disk and page RH only for days not yet
    # ingested (idempotent by fid). 'ingested_days' records which PAST dates are fully captured.
    conn.execute("""CREATE TABLE IF NOT EXISTS fills (
        fid TEXT PRIMARY KEY, order_id TEXT, trade_date TEXT, time_iso TEXT, chain TEXT,
        option_id TEXT, side TEXT, effect TEXT, strike REAL, cp TEXT, expiry TEXT, price REAL,
        qty REAL, net_cf REAL, n_legs INTEGER, leg_fallback INTEGER)""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_fills_date ON fills(trade_date)")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        date TEXT PRIMARY KEY, realized REAL, fee_inclusive REAL, fees REAL, trips INTEGER,
        cross_time TEXT, verdict TEXT, updated TEXT)""")
    conn.execute("CREATE TABLE IF NOT EXISTS ingested_days (date TEXT PRIMARY KEY)")
    return conn


def _snapshot_write_sync(row: dict) -> None:
    conn = _db_conn()
    with conn:
        conn.execute(
            "INSERT INTO snapshots VALUES (:ts,:date,:spot,:total_gex,:gamma_flip,:call_wall,"
            ":put_wall,:max_pain,:expected_move,:vix,:vix1d,:regime,:regime_score)", row)
    conn.close()


def _snapshot_read_sync(date_iso: str) -> list:
    conn = _db_conn()
    cur = conn.execute(
        "SELECT ts,spot,total_gex,gamma_flip,call_wall,put_wall,max_pain,expected_move,"
        "vix,vix1d,regime,regime_score FROM snapshots WHERE date=? ORDER BY ts", (date_iso,))
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


# ---- (#3) fills / sessions persistence -------------------------------------------------------
def _fid(f: dict) -> str:
    return f"{f.get('order_id')}|{f.get('option_id')}|{f.get('effect')}|{f.get('side')}"


def _fill_to_row(f: dict) -> tuple:
    return (_fid(f), f.get("order_id"), f.get("trade_date"),
            (f["time"].isoformat() if f.get("time") is not None else None),
            f.get("chain"), f.get("option_id"), f.get("side"), f.get("effect"),
            f.get("strike"), f.get("cp"), f.get("expiry"), f.get("price"),
            f.get("qty"), f.get("net_cf"), f.get("n_legs"),
            1 if f.get("legFallback") else 0)


def _row_to_fill(r: dict) -> dict:
    t = None
    if r.get("time_iso"):
        try:
            t = _dt.datetime.fromisoformat(r["time_iso"])
        except Exception:  # noqa: BLE001
            t = None
    out = {"time": t, "trade_date": r.get("trade_date"), "chain": r.get("chain"),
           "option_id": r.get("option_id"), "side": r.get("side"), "effect": r.get("effect"),
           "strike": r.get("strike"), "cp": r.get("cp"), "expiry": r.get("expiry"),
           "price": r.get("price"), "qty": r.get("qty"), "net_cf": r.get("net_cf"),
           "n_legs": r.get("n_legs"), "order_id": r.get("order_id")}
    if r.get("leg_fallback"):
        out["legFallback"] = True
    return out


def _ingest_fills_and_mark_sync(fills: list, complete_dates: list) -> int:
    """Idempotent upsert of fills (INSERT OR REPLACE by fid) plus marking fully-captured PAST days."""
    conn = _db_conn()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [_fill_to_row(f) for f in fills])
        if complete_dates:
            conn.executemany("INSERT OR IGNORE INTO ingested_days VALUES (?)",
                             [(d,) for d in complete_dates])
    conn.close()
    return len(fills)


def _fills_read_range_sync(start_iso: str, end_iso: str) -> list:
    conn = _db_conn()
    cur = conn.execute("SELECT * FROM fills WHERE trade_date BETWEEN ? AND ? ORDER BY time_iso",
                       (start_iso, end_iso))
    cols = [c[0] for c in cur.description]
    rows = [_row_to_fill(dict(zip(cols, r))) for r in cur.fetchall()]
    conn.close()
    return rows


def _ingested_days_sync() -> set:
    conn = _db_conn()
    cur = conn.execute("SELECT date FROM ingested_days")
    out = {r[0] for r in cur.fetchall()}
    conn.close()
    return out


def _session_upsert_sync(row: dict) -> None:
    conn = _db_conn()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES "
            "(:date,:realized,:fee_inclusive,:fees,:trips,:cross_time,:verdict,:updated)", row)
    conn.close()


def _sessions_read_range_sync(start_iso: str, end_iso: str) -> list:
    conn = _db_conn()
    cur = conn.execute("SELECT * FROM sessions WHERE date BETWEEN ? AND ? ORDER BY date",
                       (start_iso, end_iso))
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


async def _fills_window(start: "_dt.date", end: "_dt.date"):
    """(#3) Return ({trade_date: [fills]}, meta) for [start,end]: read the local store for days already
    ingested, and page RH ONCE for the window only when some day isn't captured yet (or the range
    includes today, which is never marked complete). Any DB error degrades to a single live pull, so
    this never behaves worse than the pre-cache path."""
    import asyncio
    from collections import defaultdict
    today = _today_et()
    s_iso, e_iso = start.isoformat(), end.isoformat()
    all_days = [start + _dt.timedelta(days=i) for i in range((end - start).days + 1)]
    past_days = [d for d in all_days if d < today]
    by = defaultdict(list)
    try:
        ingested = await asyncio.to_thread(_ingested_days_sync)
    except Exception:  # noqa: BLE001
        ingested = set()
    need_live = (end >= today) or any(d.isoformat() not in ingested for d in past_days)
    if not need_live:
        try:
            for f in await asyncio.to_thread(_fills_read_range_sync, s_iso, e_iso):
                by[f["trade_date"]].append(f)
            return by, {"source": "cache", "truncated": False}
        except Exception:  # noqa: BLE001
            by = defaultdict(list)   # fall through to a live pull
    orders, ometa = await asyncio.to_thread(_rh_recent_option_orders, start, 40)
    fills = []
    for o in orders:
        for f in _order_to_fills(o):
            if s_iso <= f["trade_date"] <= e_iso:
                fills.append(f)
                by[f["trade_date"]].append(f)
    try:
        # only mark past days complete when the whole window was covered (not truncated)
        complete = [] if ometa.get("truncated") else [d.isoformat() for d in past_days]
        await asyncio.to_thread(_ingest_fills_and_mark_sync, fills, complete)
    except Exception:  # noqa: BLE001
        pass
    return by, {"source": "live", "truncated": ometa.get("truncated")}


@mcp.tool()
async def snapshot_log() -> dict:
    """Capture the current 0DTE state (spot, total GEX, gamma flip, call/put walls, max-pain, expected
    move, VIX/VIX1D, regime) to a local SQLite log for intraday trend tracking.

    Call periodically through the session; `snapshot_history` then shows how dealer gamma and the key
    levels drifted (GEX migration). Stored at ~/.trading/traders_edge.db.
    """
    import asyncio
    # NOTE (B15): this calls the @mcp.tool()-decorated coroutines directly. Under FastMCP >= 3 the
    # decorator returns the underlying function unchanged, so this works; on FastMCP 2.x it returned
    # a Tool wrapper and these calls would raise. requirements.txt pins fastmcp>=3.4.0 -- keep it.
    # freshness-lint: exempt -- the "spot" written below is a PERSISTED DB row, not tool output.
    z, vx, rg = await asyncio.gather(zero_dte_exposure(), vix_complex(), regime_classifier(),
                                     return_exceptions=True)
    if isinstance(z, Exception) or (isinstance(z, dict) and "error" in z):
        await asyncio.sleep(0.8)
        try:
            z2 = await zero_dte_exposure()
            if not (isinstance(z2, dict) and "error" in z2):
                z = z2
        except Exception:  # noqa: BLE001
            pass
    degraded = isinstance(z, Exception) or not isinstance(z, dict) or "error" in z
    chain_err = None
    if degraded:
        chain_err = str(z) if isinstance(z, Exception) else (z.get("error") if isinstance(z, dict) else "unknown")
        z = {}
    vx = vx if isinstance(vx, dict) else {}
    rg = rg if isinstance(rg, dict) else {}
    em = z.get("expectedMove") or {}
    idx = vx.get("indices") or {}
    cw, pw = z.get("callWall"), z.get("putWall")
    now = _dt.datetime.now(ET)
    row = {
        "ts": now.isoformat(), "date": now.date().isoformat(),
        "spot": z.get("spot"), "total_gex": z.get("totalGEX"), "gamma_flip": z.get("gammaFlip"),
        "call_wall": cw.get("strike") if isinstance(cw, dict) else None,
        "put_wall": pw.get("strike") if isinstance(pw, dict) else None,
        "max_pain": z.get("maxPainPin"), "expected_move": em.get("expectedMovePts"),
        "vix": (idx.get("VIX") or {}).get("value"), "vix1d": (idx.get("VIX1D") or {}).get("value"),
        "regime": rg.get("regime") if isinstance(rg, dict) else None,
        "regime_score": rg.get("compositeScore") if isinstance(rg, dict) else None,
    }
    try:
        await asyncio.to_thread(_snapshot_write_sync, row)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"DB write failed: {str(exc)[:160]}"}
    out = {"logged": True, "db": TE_DB_PATH,
            "snapshot": {k: row[k] for k in ("ts", "spot", "total_gex", "gamma_flip", "call_wall",
                                             "put_wall", "max_pain", "expected_move", "vix", "vix1d",
                                             "regime", "regime_score")}}
    if degraded:
        out["degraded"] = True
        out["note"] = (f"Chain-derived levels unavailable ({str(chain_err)[:120]}); logged vol/regime "
                       "only to preserve the intraday timeline.")
    return out


@mcp.tool()
async def snapshot_history(date: Optional[str] = None) -> dict:
    """Read back the day's logged snapshots and summarize how key levels drifted - the intraday GEX
    migration and where dealer positioning moved.

    Returns each snapshot plus first/last/min/max and net change for spot, total GEX, gamma flip, and
    call/put walls. `date` (YYYY-MM-DD ET) defaults to today. Requires prior `snapshot_log` calls.
    """
    import asyncio
    d = date or _today_et().isoformat()
    try:
        rows = await asyncio.to_thread(_snapshot_read_sync, d)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"DB read failed: {str(exc)[:160]}"}
    if not rows:
        return {"date": d, "note": "No snapshots logged for this date. Call snapshot_log during the session."}

    def drift(field):
        nums = [r[field] for r in rows if r[field] is not None]
        if not nums:
            return None
        return {"first": nums[0], "last": nums[-1], "min": min(nums), "max": max(nums),
                "change": round(nums[-1] - nums[0], 2)}

    # freshness-lint: exempt -- these are HISTORICAL snapshot rows; freshness is each row's own ts.
    fmt = [{"t": (r["ts"][11:19] if r["ts"] else None), "spot": r["spot"], "gex": r["total_gex"],
            "flip": r["gamma_flip"], "callWall": r["call_wall"], "putWall": r["put_wall"],
            "vix": r["vix"], "regime": r["regime"]} for r in rows]
    return {"date": d, "snapshots": len(rows),
            "drift": {k: drift(k) for k in ("spot", "total_gex", "gamma_flip", "call_wall",
                                            "put_wall", "vix")},
            "rows": fmt}


# ============================================================================
# ROLL-CANDIDATE FINDER - roll up-and-out targets for covered calls
# ============================================================================
def _roll_scan_sync(symbol: str, cur_strike: float, cur_expiry: str, min_dte: int,
                    max_dte: int, target_delta: float, n_per_expiry: int) -> dict:
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    px = None
    try:
        lp = rh.get_latest_price(symbol)
        px = float(lp[0]) if lp else None
    except Exception:  # noqa: BLE001
        px = None
    try:
        ch = rh.options.get_chains(symbol)
        exps = ch.get("expiration_dates", []) if isinstance(ch, dict) else []
    except Exception:  # noqa: BLE001
        exps = []
    today = _today_et()
    try:
        cur_exp_d = _dt.date.fromisoformat(cur_expiry)
    except Exception:  # noqa: BLE001
        cur_exp_d = today
    targets = []
    for e in exps:
        try:
            ed = _dt.date.fromisoformat(e)
        except Exception:  # noqa: BLE001
            continue
        dte = (ed - today).days
        if ed > cur_exp_d and min_dte <= dte <= max_dte:
            targets.append((e, dte))
    targets = targets[:2]
    cur_mark = None
    try:
        md = rh.options.find_options_by_expiration_and_strike(
            symbol, cur_expiry, str(cur_strike), "call")
        if md:
            cur_mark = (float(md[0].get("mark_price") or md[0].get("adjusted_mark_price") or 0)
                        or None)
    except Exception:  # noqa: BLE001
        cur_mark = None
    cands = []
    for e, dte in targets:
        try:
            calls = rh.options.find_options_by_expiration(
                symbol, expirationDate=e, optionType="call") or []
        except Exception:  # noqa: BLE001
            continue
        rows = []
        for o in calls:
            try:
                k = float(o.get("strike_price") or 0)
            except (TypeError, ValueError):
                continue
            if k < cur_strike:
                continue
            mark = float(o.get("mark_price") or o.get("adjusted_mark_price") or 0)
            delta = float(o.get("delta") or 0)
            rows.append((k, mark, delta, float(o.get("implied_volatility") or 0),
                         float(o.get("open_interest") or 0)))
        rows.sort(key=lambda r: abs(r[2] - target_delta))
        for k, mark, delta, iv, oi in rows[:n_per_expiry]:
            net_credit = (mark - cur_mark) if cur_mark is not None else None
            ann = ((mark / px) * (365.0 / dte) * 100.0) if (px and dte > 0) else None
            cands.append({"expiry": e, "dte": dte, "strike": k, "mark$/sh": round(mark, 2),
                          "delta": round(delta, 3), "iv": round(iv, 4), "oi": int(oi),
                          "netCreditVsClose$/sh": (round(net_credit, 2) if net_credit is not None else None),
                          "annYield%": round(ann, 1) if ann is not None else None})
    cands.sort(key=lambda c: (-(c["netCreditVsClose$/sh"] or -1e9), c["dte"]))
    return {"underlying": symbol, "underlyingPx": round(px, 2) if px else None,
            "currentCall": {"strike": cur_strike, "expiry": cur_expiry,
                            "markToClose$/sh": round(cur_mark, 2) if cur_mark else None},
            "candidates": cands}


@mcp.tool()
async def roll_candidates(underlying: Optional[str] = None,
                          current_strike: Optional[float] = None,
                          current_expiry: Optional[str] = None,
                          target_delta: Annotated[float, Field(ge=0.05, le=0.6)] = 0.30,
                          min_dte: Annotated[int, Field(ge=1, le=120)] = 20,
                          max_dte: Annotated[int, Field(ge=1, le=180)] = 45) -> dict:
    """Suggest roll-up-and-out targets for a covered call: candidate strikes/expiries with mark, delta,
    net credit vs closing the current call, and annualized yield.

    With no args it scans your open short calls and proposes rolls for each. Or pass `underlying`,
    `current_strike`, `current_expiry` to evaluate a specific call. Targets are expiries after the
    current one within `min_dte`-`max_dte` days, strikes at/above current, ranked toward `target_delta`
    then by net credit.
    """
    import asyncio
    jobs = []
    if underlying and current_strike and current_expiry:
        jobs.append((underlying.upper(), float(current_strike), current_expiry))
    else:
        try:
            pos = await _robinhood_positions()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        for p in pos:
            if (p.get("type") == "option" and p.get("cp") == "C" and (p.get("qty") or 0) < 0
                    and p.get("underlying") and p.get("strike") and p.get("expiry")):
                jobs.append((p["underlying"], p["strike"], p["expiry"]))
        if not jobs:
            return {"note": "No short calls open and no explicit call specified. Pass underlying, "
                            "current_strike, current_expiry to evaluate a hypothetical roll."}
    # Roll scans are independent per position, so run them concurrently instead of one-at-a-time.
    sel = jobs[:5]
    scans = await asyncio.gather(
        *[asyncio.to_thread(_roll_scan_sync, sym, k, e, min_dte, max_dte, target_delta, 4)
          for sym, k, e in sel],
        return_exceptions=True)
    results = []
    for (sym, k, e), r in zip(sel, scans):
        if isinstance(r, Exception):
            results.append({"underlying": sym, "error": str(r)[:140]})
        else:
            results.append(r)
    return {"rollTargets": {"minDte": min_dte, "maxDte": max_dte, "targetDelta": target_delta},
            "results": results}


# ============================================================================
# v0.8.0 - daily workflow, wheel/income, risk analytics, 0DTE execution
# ============================================================================
import statistics as _stats


def _safe_dict(x):
    return x if isinstance(x, dict) else {"error": str(x)[:160]}


async def _recent_session_summary():
    import asyncio
    from collections import defaultdict
    orders, _ometa = await asyncio.to_thread(
        _rh_recent_option_orders, _today_et() - _dt.timedelta(days=7), 8)
    by = defaultdict(list)
    for o in orders:
        for f in _order_to_fills(o):        # B2
            by[f["trade_date"]].append(f)
    if not by:
        return None
    last = max(by.keys())
    trips = _round_trips(sorted(by[last], key=lambda r: r["time"]))
    if not trips:
        return None
    total = sum(tr["pnl"] for tr in trips)
    wins = sum(1 for tr in trips if tr["pnl"] > 0)
    return {"date": last, "pnl$": round(total, 2), "trips": len(trips),
            "winRate%": round(100 * wins / len(trips), 1)}


@mcp.tool()
async def morning_brief() -> dict:
    """Pre-open command center: regime + posture, today's key 0DTE levels (spot, expected move, gamma
    flip, call/put walls, max-pain), the vol complex, high-impact economic events and holdings reporting
    earnings within ~7 days, your last session result, and the discipline reset. One call, not five.
    """
    import asyncio
    z, rg, vx, ec, ear = await asyncio.gather(
        zero_dte_exposure(), regime_classifier(), vix_complex(), economic_calendar(),
        earnings_calendar(days=10), return_exceptions=True)
    z, rg, vx, ec, ear = (_safe_dict(z), _safe_dict(rg), _safe_dict(vx), _safe_dict(ec), _safe_dict(ear))
    today = _today_et()
    ev = []
    for e in (ec.get("events") or []):
        try:
            d = _dt.date.fromisoformat(e["date"][:10])
        except Exception:
            continue
        if today <= d <= today + _dt.timedelta(days=2) and e.get("importance") in ("high", "medium"):
            ev.append({"date": e["date"], "time": e.get("time"), "name": e.get("name"),
                       "importance": e.get("importance")})
    earnings_soon = [r for r in (ear.get("nextEarnings") or [])
                     if r.get("daysAway") is not None and r["daysAway"] <= 7]
    idx = (vx.get("indices") or {})
    em = z.get("expectedMove") or {}
    cw, pw = (z.get("callWall") or {}), (z.get("putWall") or {})
    out = {
        "date": today.isoformat(),
        "regime": {"label": rg.get("regime"), "score": rg.get("compositeScore"),
                   "posture": rg.get("posture")},
        "levels": {"spot": z.get("spot"), "asof": z.get("asof"),
                   "source": z.get("source"), "freshness": z.get("freshness"),   # (#1)
                   "expectedMovePts": em.get("expectedMovePts"), "expectedMovePct": em.get("expectedMovePct"),
                   "gammaFlip": z.get("gammaFlip"), "spotVsFlip": z.get("spotVsFlip"),
                   "callWall": cw.get("strike"), "putWall": pw.get("strike"),
                   "maxPain": z.get("maxPainPin"), "gexRegime": z.get("regime")},
        "vol": {"vix": (idx.get("VIX") or {}).get("value"), "vix1d": (idx.get("VIX1D") or {}).get("value"),
                "vix9d": (idx.get("VIX9D") or {}).get("value")},
        "economicEventsNext2d": ev,
        "earningsNext7d": [{"symbol": r["symbol"], "date": r["date"], "session": r.get("session"),
                            "daysAway": r["daysAway"]} for r in earnings_soon],
    }
    try:
        ls = await _recent_session_summary()
        if ls:
            out["lastSession"] = ls
    except Exception:
        pass
    out["disciplineReset"] = "Fresh day - target and give-back limits reset. Run should_i_trade first."
    return out


@mcp.tool()
async def eod_wrap() -> dict:
    """End-of-day wrap: today realized vs target, discipline adherence (stopped at target vs gave back),
    where the key 0DTE levels closed, and a snapshot logged to SQLite history. Run after the close to
    score the day and capture closing state.
    """
    import asyncio
    # daily_review / should_i_trade / snapshot_log each `import robin_stocks` inside their own
    # to_thread worker. On a COLD process three simultaneous first-imports trip CPython's
    # module-lock deadlock detector (_DeadlockError) -- and eod_wrap is a plausible first call
    # right after a server restart. Warm the import on one thread so the gather finds it cached.
    try:
        await asyncio.to_thread(__import__, "robin_stocks.robinhood")
    except Exception:  # noqa: BLE001 -- if it's genuinely missing, surfaces downstream as EdgeError
        pass
    dr, sit, snap = await asyncio.gather(daily_review(), should_i_trade(), snapshot_log(),
                                         return_exceptions=True)
    dr, sit, snap = _safe_dict(dr), _safe_dict(sit), _safe_dict(snap)
    out = {"date": _today_et().isoformat()}
    # daily_review emits realized$ / roundTrips and nests the target split under
    # beforeVsAfterTarget. Read THOSE names: pnl$ / trades / targetHitAt are a stale
    # daily_review schema, so "pnl$" was never present -- every session fell through to
    # "no fills today" and the give-back leak note below was unreachable.
    bva = dr.get("beforeVsAfterTarget") or {}
    hit_at = bva.get("targetHitAt")
    after = bva.get("afterTarget") or {}
    realized = dr.get("realized$")
    if dr.get("error"):
        # An upstream failure (dropped RH session, etc.) must never look like a flat day.
        out["result"] = {"error": dr["error"],
                         "note": "daily_review failed - P&L UNKNOWN, not a no-trade day."}
    elif realized is not None:
        out["result"] = {"realized$": realized, "trips": dr.get("roundTrips"),
                         "winRate%": dr.get("winRate%"), "targetHitAt": hit_at,
                         "beforeTarget": bva.get("beforeTarget"), "afterTarget": after or None}
    else:
        out["result"] = {"note": dr.get("note") or "no fills today"}
    tgt = _target()
    notes = []
    if hit_at:
        ap = after.get("pnl$")
        if ap is not None and ap < 0:
            notes.append(f"Hit target at {hit_at}, then gave back ${abs(ap):.2f} after - the leak pattern.")
        elif ap is not None and ap > 0:
            notes.append(f"Hit target at {hit_at} and added ${ap:.2f} after.")
        else:
            notes.append(f"Hit target at {hit_at}.")
    elif realized is not None:
        notes.append(f"Target ${tgt:.0f} not reached (realized ${realized:.2f}).")
    out["discipline"] = {"verdict": sit.get("verdict"), "notes": notes}
    if isinstance(snap, dict) and snap.get("logged"):
        s = snap.get("snapshot") or {}
        out["closingLevels"] = {"spot": s.get("spot"), "gammaFlip": s.get("gamma_flip"),
                                "callWall": s.get("call_wall"), "putWall": s.get("put_wall"),
                                "vix": s.get("vix"), "regime": s.get("regime"),
                                "source": "closing snapshot",                     # (#1)
                                "freshness": {"asof": out["date"], "note": "session close"}}
        out["snapshotLogged"] = True
    else:
        out["snapshotLogged"] = False
    # (#3) Persist the session scorecard + today's fills so weekly_review / discipline_backtest can
    # read history from disk instead of re-paging RH (and it survives RH endpoint drift). Today is
    # NOT marked "complete" here -- it gets ingested as a past day on the next window pull.
    try:
        import asyncio
        await asyncio.to_thread(_session_upsert_sync, {
            "date": out["date"], "realized": realized, "fee_inclusive": None, "fees": None,
            "trips": dr.get("roundTrips"), "cross_time": hit_at, "verdict": sit.get("verdict"),
            "updated": _dt.datetime.now(ET).isoformat()})
        _by, _m = await _fills_window(_today_et(), _today_et())
        out["persisted"] = {"session": True, "fillDays": len(_by), "source": _m.get("source")}
    except Exception as _exc:  # noqa: BLE001
        out["persisted"] = {"error": str(_exc)[:120]}
    return out


@mcp.tool()
async def session_db(action: str = "status", start: Optional[str] = None,
                     end: Optional[str] = None) -> dict:
    """(#3) Inspect / backfill the local SQLite session store that weekly_review and
    discipline_backtest read from. action='status' (ingested-day count + range), 'backfill' (page RH
    once for [start,end] and upsert, marking complete past days), or 'sessions' (read persisted daily
    scorecards). Dates YYYY-MM-DD ET; default range = last 30 days. The store lives at TE_DB_PATH.
    """
    import asyncio
    today = _today_et()
    try:
        s = _dt.date.fromisoformat(start) if start else (today - _dt.timedelta(days=30))
        e = _dt.date.fromisoformat(end) if end else today
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD."}
    act = (action or "status").lower()
    if act == "status":
        try:
            ing = sorted(await asyncio.to_thread(_ingested_days_sync))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:140]}"}
        return {"dbPath": TE_DB_PATH, "ingestedDayCount": len(ing),
                "earliest": (ing[0] if ing else None), "latest": (ing[-1] if ing else None)}
    if act == "backfill":
        by, meta = await _fills_window(s, e)
        return {"backfilled": {"from": s.isoformat(), "to": e.isoformat()},
                "tradingDaysWithFills": len(by), "fills": sum(len(v) for v in by.values()),
                "source": meta.get("source"), "truncated": meta.get("truncated")}
    if act == "sessions":
        try:
            rows = await asyncio.to_thread(_sessions_read_range_sync, s.isoformat(), e.isoformat())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:140]}"}
        return {"from": s.isoformat(), "to": e.isoformat(), "sessions": rows}
    return {"error": f"Unknown action '{action}'. Use status | backfill | sessions."}


@mcp.tool()
async def weekly_review(week_offset: Annotated[int, Field(ge=0, le=12)] = 0) -> dict:
    """This week realized P&L vs your weekly target: Mon-Fri daily breakdown, best/worst day, win rate,
    and progress to goal. week_offset looks back N weeks (0 = current). Set a goal with
    trading_config(action=set, key=weekly_target, value=2500).
    """
    import asyncio
    from collections import defaultdict
    today = _today_et()
    monday = today - _dt.timedelta(days=today.weekday()) - _dt.timedelta(weeks=week_offset)
    friday = monday + _dt.timedelta(days=4)
    by, ometa = await _fills_window(monday, friday)   # (#3) DB-first; pages RH only for new days
    days, total, alltrips = [], 0.0, []
    for d in sorted(by.keys()):
        trips = _round_trips(sorted(by[d], key=lambda r: r["time"]))
        if not trips:
            continue
        p = sum(tr["pnl"] for tr in trips)
        total += p
        alltrips += trips
        days.append({"date": d, "day": _dt.date.fromisoformat(d).strftime("%a"),
                     "pnl$": round(p, 2), "trips": len(trips)})
    wt = _cfg("weekly_target")
    wins = sum(1 for tr in alltrips if tr["pnl"] > 0)
    out = {"week": f"{monday.isoformat()} to {friday.isoformat()}", "realized$": round(total, 2),
           "tradingDays": len(days), "roundTrips": len(alltrips),
           "winRate%": round(100 * wins / len(alltrips), 1) if alltrips else None, "byDay": days,
           "bestDay": max(days, key=lambda d: d["pnl$"]) if days else None,
           "worstDay": min(days, key=lambda d: d["pnl$"]) if days else None}
    if wt:
        out["weeklyTarget$"] = float(wt)
        out["pctOfWeeklyTarget"] = round(100 * total / float(wt), 1) if float(wt) else None
        out["overUnder$"] = round(total - float(wt), 2)
    else:
        out["weeklyTargetNote"] = "No weekly_target set. Use trading_config to set one."
    if ometa.get("truncated"):
        out["dataWarning"] = "Order history hit the page cap; this week's totals may be incomplete."
    return out


@mcp.tool()
async def tilt_detector(date: Optional[str] = None) -> dict:
    """Scan a session trade sequence for tilt: revenge sizing (size up after losses), rushing (shrinking
    time between entries), intraday win-rate decay, and trading after a give-back from peak. Flags tilt
    while there is still time to stop. date (ET) defaults to today.
    """
    import asyncio
    d = date or _today_et().isoformat()
    orders, ometa = await asyncio.to_thread(_rh_recent_option_orders, _dt.date.fromisoformat(d), 10)
    fills = [f for o in orders for f in _order_to_fills(o) if f["trade_date"] == d]  # B2
    if not fills:
        return {"date": d, "note": "No fills for this session."}
    fills.sort(key=lambda r: r["time"])
    trips, tstats = _round_trips_full(fills)
    data_warnings = _fill_data_warnings(ometa, tstats)
    if len(trips) < 4:
        out = {"date": d, "roundTrips": len(trips),
               "note": f"Only {len(trips)} round trips - too few to assess tilt."}
        if data_warnings:
            out["dataWarnings"] = data_warnings
        return out
    flags, ev = [], {}
    after_loss, after_win = [], []
    for i in range(1, len(trips)):
        (after_loss if trips[i - 1]["pnl"] < 0 else after_win).append(abs(trips[i]["qty"]))
    al = _stats.mean(after_loss) if after_loss else 0.0
    aw = _stats.mean(after_win) if after_win else 0.0
    ev["avgQtyAfterLoss"], ev["avgQtyAfterWin"] = round(al, 2), round(aw, 2)
    if aw > 0 and al > 1.25 * aw:
        flags.append(f"Revenge sizing: avg {al:.1f} contracts after a loss vs {aw:.1f} after a win.")
    # dedupe opens by opening ORDER (B2) so the several legs of one spread -- which share an open
    # time -- don't read as zero-gap "rushing".
    opens = sorted({(tr.get("open_order_id") or tr["open"]): tr["open"] for tr in trips}.values())
    gaps = [(opens[i] - opens[i - 1]).total_seconds() for i in range(1, len(opens))]
    if len(gaps) >= 6:
        third = len(gaps) // 3
        early, late = _stats.mean(gaps[:third]), _stats.mean(gaps[-third:])
        ev["avgGapEarlyMin"], ev["avgGapLateMin"] = round(early / 60, 1), round(late / 60, 1)
        if early > 0 and late < 0.6 * early:
            flags.append(f"Rushing: entries ~{late / 60:.0f}m apart late vs ~{early / 60:.0f}m early.")
    half = len(trips) // 2
    fw = sum(1 for tr in trips[:half] if tr["pnl"] > 0) / half
    sw = sum(1 for tr in trips[half:] if tr["pnl"] > 0) / (len(trips) - half)
    ev["winRateFirstHalf%"], ev["winRateSecondHalf%"] = round(100 * fw, 0), round(100 * sw, 0)
    if sw < fw - 0.20:
        flags.append(f"Win-rate decay: {100 * fw:.0f}% first half -> {100 * sw:.0f}% second half.")
    cur = _build_curve(trips, _target())
    if cur.get("crossIdx") is not None:
        aft = cur["total"] - cur["crossCum"]
        if aft < 0:
            flags.append(f"Gave back ${aft:.2f} after hitting target - traded past the stop signal.")
    cum = peak = ddmax = 0.0
    for tr in trips:
        cum += tr["pnl"]
        peak = max(peak, cum)
        ddmax = min(ddmax, cum - peak)
    ev["maxDrawdownFromPeak$"] = round(ddmax, 2)
    level = ("HIGH" if len(flags) >= 3 else "ELEVATED" if len(flags) == 2
             else "MILD" if len(flags) == 1 else "NONE")
    advice = ("Step away - multiple tilt signals present." if level in ("HIGH", "ELEVATED")
              else "One soft signal; stay disciplined." if level == "MILD"
              else "No tilt signatures detected.")
    out = {"date": d, "roundTrips": len(trips), "tiltLevel": level, "flags": flags,
           "evidence": ev, "advice": advice}
    if data_warnings:
        out["dataWarnings"] = data_warnings
    return out


def _holding_sync(sym: str):
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    h = rh.build_holdings() or {}
    d = h.get(sym)
    if not d:
        return {"shares": 0.0, "avgCost": None, "price": None}
    return {"shares": _to_float(d.get("quantity")) or 0.0, "avgCost": _to_float(d.get("average_buy_price")),
            "price": _to_float(d.get("price"))}


@mcp.tool()
async def wheel_tracker(symbol: str = "ICE",
                        lookback_days: Annotated[int, Field(ge=30, le=1500)] = 365) -> dict:
    """Lifetime wheel scorecard for a symbol: net option premium collected (calls + puts), contracts
    sold to open, buy-to-close cost, expiry cycles traded, your share position and average cost, and the
    effective cost basis after premium. lookback_days bounds the order history scanned.
    """
    import asyncio
    sym = symbol.upper()
    stop = _today_et() - _dt.timedelta(days=lookback_days)
    orders, ometa = await asyncio.to_thread(_rh_recent_option_orders, stop, 40)
    calls_credit = puts_credit = btc_debit = 0.0
    sto_calls = sto_puts = 0.0
    expiries = set()
    for o in orders:
        for f in _order_to_fills(o):        # B2: multi-leg rolls now counted leg-by-leg
            if f.get("chain") != sym:
                continue
            expiries.add(f.get("expiry"))
            eff, side, cp = f.get("effect"), f.get("side"), f.get("cp")
            ncf, q = (f.get("net_cf") or 0.0), abs(f.get("qty") or 0)
            if eff == "open" and side == "sell":
                if cp == "C":
                    calls_credit += ncf
                    sto_calls += q
                elif cp == "P":
                    puts_credit += ncf
                    sto_puts += q
            elif eff == "close" and side == "buy":
                btc_debit += ncf
    net_prem = calls_credit + puts_credit + btc_debit
    hold = await asyncio.to_thread(_holding_sync, sym)
    shares = hold.get("shares") or 0.0
    avg = hold.get("avgCost")
    eff_basis = (avg - net_prem / shares) if (avg and shares > 0) else None
    out = {"symbol": sym, "lookbackDays": lookback_days,
           "netOptionPremium$": round(net_prem, 2),
           "callPremium$": round(calls_credit, 2), "putPremium$": round(puts_credit, 2),
           "buyToCloseCost$": round(-btc_debit, 2),   # B15: report as positive dollars paid to close
           "shortCallsSold": int(sto_calls), "shortPutsSold": int(sto_puts),
           "expiryCyclesTraded": len([e for e in expiries if e]),
           "shares": round(shares, 4), "avgCost$": round(avg, 2) if avg else None,
           "currentPrice$": hold.get("price"),
           "effectiveBasisAfterPremium$": round(eff_basis, 2) if eff_basis is not None else None,
           "basisReductionPerShare$": round(net_prem / shares, 2) if shares > 0 else None,
           "note": ("Net premium = call+put credits minus buy-to-close debits over the window. "
                    "buyToCloseCost$ is the positive dollars paid to close. Assignment/exercise legs "
                    "may post separately; verify against statements.")}
    if ometa.get("truncated"):
        out["dataWarning"] = "Order history hit the page cap; the lookback may be incomplete."
    return out


def _write_scan_sync(symbol, opt_type, target_delta, min_dte, max_dte, n_per_expiry):
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        lp = rh.get_latest_price(symbol)
        px = _to_float(lp[0]) if lp else None
    except Exception:
        px = None
    try:
        ch = rh.options.get_chains(symbol)
        exps = ch.get("expiration_dates", []) if isinstance(ch, dict) else []
    except Exception:
        exps = []
    today = _today_et()
    targets = []
    for e in exps:
        try:
            dte = (_dt.date.fromisoformat(e) - today).days
        except Exception:
            continue
        if min_dte <= dte <= max_dte:
            targets.append((e, dte))
    targets = targets[:2]
    out = []
    for e, dte in targets:
        try:
            opts = rh.options.find_options_by_expiration(symbol, expirationDate=e, optionType=opt_type) or []
        except Exception:
            continue
        rows = []
        for o in opts:
            k = _to_float(o.get("strike_price"))
            if k is None:
                continue
            if opt_type == "call" and px and k < px:
                continue
            if opt_type == "put" and px and k > px:
                continue
            mark = _to_float(o.get("mark_price")) or _to_float(o.get("adjusted_mark_price")) or 0.0
            delta = _to_float(o.get("delta")) or 0.0
            rows.append((k, mark, delta, _to_float(o.get("implied_volatility")) or 0.0,
                         _to_float(o.get("open_interest")) or 0.0))
        rows.sort(key=lambda r: abs(abs(r[2]) - target_delta))
        for k, mark, delta, iv, oi in rows[:n_per_expiry]:
            base = px if opt_type == "call" else k
            ann = (mark / base) * (365.0 / dte) * 100.0 if (base and dte > 0) else None
            out.append({"expiry": e, "dte": dte, "strike": k, "markPerSh$": round(mark, 2),
                        "delta": round(delta, 3), "iv": round(iv, 4), "oi": int(oi),
                        "premiumPerContract$": round(mark * 100, 2),
                        "annYield%": round(ann, 1) if ann is not None else None})
    out.sort(key=lambda c: abs(abs(c["delta"]) - target_delta))
    return {"underlying": symbol, "underlyingPx": round(px, 2) if px else None, "candidates": out}


@mcp.tool()
async def covered_call_writer(symbol: str = "ICE", target_delta: float = 0.30,
                              min_dte: int = 25, max_dte: int = 45) -> dict:
    """Fresh covered calls to write on a symbol you hold: OTM call strikes near your target delta across
    the next expiries, ranked by annualized yield, with how many contracts your shares cover and a flag
    for any earnings or ex-dividend date before expiry (early-assignment risk).
    """
    import asyncio
    sym = symbol.upper()
    scan, hold, ear, div = await asyncio.gather(
        asyncio.to_thread(_write_scan_sync, sym, "call", target_delta, min_dte, max_dte, 3),
        asyncio.to_thread(_holding_sync, sym), _next_earnings(sym),
        asyncio.to_thread(_next_ex_div_sync, sym), return_exceptions=True)
    scan = _safe_dict(scan)
    hold = hold if isinstance(hold, dict) else {}
    shares = hold.get("shares") or 0.0
    for c in scan.get("candidates", []):
        flags = []
        if isinstance(ear, dict) and ear.get("date") and ear["date"] <= c["expiry"]:
            flags.append(f"earnings {ear['date']} ({ear.get('session')})")
        if isinstance(div, dict) and div.get("nextExDate") and div["nextExDate"] <= c["expiry"]:
            flags.append(f"ex-div {div['nextExDate']}")
        c["riskBeforeExpiry"] = flags or None
    return {"symbol": sym, "underlyingPx": scan.get("underlyingPx"), "sharesHeld": round(shares, 4),
            "contractsCovered": int(shares // 100), "targetDelta": target_delta,
            "candidates": scan.get("candidates", []),
            "note": "Yield annualized on stock price. Writing more than contractsCovered is uncovered."}


@mcp.tool()
async def csp_finder(symbol: str = "ICE", target_delta: float = 0.30,
                     min_dte: int = 25, max_dte: int = 45) -> dict:
    """Cash-secured puts to sell on a symbol: OTM put strikes near your target delta across the next
    expiries, ranked by annualized yield on the cash secured (strike x100), with cash required per
    contract and an earnings-before-expiry flag.
    """
    import asyncio
    sym = symbol.upper()
    scan, ear = await asyncio.gather(
        asyncio.to_thread(_write_scan_sync, sym, "put", target_delta, min_dte, max_dte, 3),
        _next_earnings(sym), return_exceptions=True)
    scan = _safe_dict(scan)
    for c in scan.get("candidates", []):
        c["cashSecured$"] = round(c["strike"] * 100, 2)
        flags = []
        if isinstance(ear, dict) and ear.get("date") and ear["date"] <= c["expiry"]:
            flags.append(f"earnings {ear['date']} ({ear.get('session')})")
        c["riskBeforeExpiry"] = flags or None
    return {"symbol": sym, "underlyingPx": scan.get("underlyingPx"), "targetDelta": target_delta,
            "candidates": scan.get("candidates", []),
            "note": "Yield annualized on cash secured (strike x 100); assignment buys stock at the strike."}


def _div_info_sync(symbols):
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    ff = rh.stocks.get_fundamentals(symbols) or []
    try:
        prices = rh.stocks.get_latest_price(symbols) or []
    except Exception:
        prices = [None] * len(symbols)
    out = {}
    for sym, f, p in zip(symbols, ff, prices):
        if not f:
            out[sym] = None
            continue
        out[sym] = {"dps": _to_float(f.get("dividend_per_share")), "yieldPct": _to_float(f.get("dividend_yield")),
                    "px": _to_float(p), "lastEx": f.get("ex_dividend_date")}
    return out


def _project_ex(info):
    from calendar import monthrange
    dps, yld, px, lastex = info.get("dps"), info.get("yieldPct"), info.get("px"), info.get("lastEx")
    if not (dps and dps > 0) or not lastex:
        return {"nextExDate": None, "lastExDate": lastex, "freq": None, "dps": dps, "yieldPct": yld}
    freq = None
    if yld and px:
        per_year = ((yld / 100.0) * px) / dps
        if per_year > 0:
            freq = min([1, 2, 4, 12], key=lambda fr: abs(fr - per_year))
    try:
        d = _dt.date.fromisoformat(lastex[:10])
    except Exception:
        return {"nextExDate": None, "lastExDate": lastex, "freq": freq, "dps": dps, "yieldPct": yld}
    today = _today_et()
    nxt = d
    if freq:
        step = 12 // freq
        guard = 0
        while nxt < today and guard < 36:
            m = nxt.month - 1 + step
            y = nxt.year + m // 12
            m = m % 12 + 1
            nxt = _dt.date(y, m, min(nxt.day, monthrange(y, m)[1]))
            guard += 1
    return {"nextExDate": nxt.isoformat() if nxt >= today else None, "lastExDate": lastex,
            "freq": freq, "dps": dps, "yieldPct": yld}


def _next_ex_div_sync(symbol):
    info = _div_info_sync([symbol]).get(symbol)
    return _project_ex(info) if info else None


@mcp.tool()
async def dividend_calendar(symbols: Optional[str] = None,
                            days: Annotated[int, Field(ge=1, le=400)] = 90) -> dict:
    """Upcoming ex-dividend dates for your holdings (or a symbol list): estimated next ex-date, payout
    frequency, dividend per share, and yield. Ex-div dates drive early-assignment risk on short calls.
    Dates are projected from Robinhood fundamentals (last ex-date + frequency).
    """
    import asyncio
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        pos = await _robinhood_positions()
        syms = sorted({p["symbol"] for p in pos if p.get("type") == "equity"})
    if not syms:
        return {"note": "No symbols to check."}
    info = await asyncio.to_thread(_div_info_sync, syms)
    today, rows, none_div = _today_et(), [], []
    fmap = {1: "annual", 2: "semiannual", 4: "quarterly", 12: "monthly"}
    for s in syms:
        i = info.get(s)
        if not i or not i.get("dps"):
            none_div.append(s)
            continue
        proj = _project_ex(i)
        nx = proj.get("nextExDate")
        da = (_dt.date.fromisoformat(nx) - today).days if nx else None
        rows.append({"symbol": s, "nextExDate": nx, "daysAway": da,
                     "freq": fmap.get(proj.get("freq")), "dividendPerShare$": proj.get("dps"),
                     "yieldPct": proj.get("yieldPct"), "withinWindow": (da is not None and da <= days)})
    rows.sort(key=lambda r: (r["daysAway"] if r["daysAway"] is not None else 9999))
    return {"windowDays": days, "asof": today.isoformat(), "exDividends": rows, "noDividend": none_div,
            "note": "Next ex-date is projected (last ex-date + frequency); verify before acting."}


def _hist_closes_sync(symbols, span):
    import robin_stocks.robinhood as rh
    from collections import defaultdict
    _rh_login_sync()
    h = rh.stocks.get_stock_historicals(symbols, interval="day", span=span) or []
    series = defaultdict(dict)
    for row in h:
        s, ts, c = row.get("symbol"), (row.get("begins_at") or "")[:10], _to_float(row.get("close_price"))
        if s and ts and c:
            series[s][ts] = c
    return dict(series)


@mcp.tool()
async def correlation_matrix(lookback_days: Annotated[int, Field(ge=20, le=500)] = 90,
                             symbols: Optional[str] = None) -> dict:
    """Daily-return correlation across your holdings (or a symbol list): the pairwise matrix, each name
    average correlation to the rest, the most/least correlated pairs, and the portfolio-wide average - a
    true-diversification check (10 tickers that move together are not diversified).
    """
    import asyncio
    import numpy as np
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        pos = await _robinhood_positions()
        syms = sorted({p["symbol"] for p in pos if p.get("type") == "equity"})
    if len(syms) < 2:
        return {"note": "Need at least 2 symbols."}
    span = "year" if lookback_days > 180 else "3month" if lookback_days > 30 else "month"
    series = await asyncio.to_thread(_hist_closes_sync, syms, span)
    have = {s: series.get(s, {}) for s in syms if series.get(s)}
    syms = [s for s in syms if s in have]
    if len(syms) < 2:
        return {"note": "Insufficient history."}
    common = sorted(set.intersection(*[set(have[s].keys()) for s in syms]))
    common = common[-lookback_days:] if len(common) > lookback_days else common
    if len(common) < 10:
        return {"note": f"Only {len(common)} overlapping days of history."}
    rets = {s: np.diff(np.log([have[s][d] for d in common])) for s in syms}
    M = np.array([rets[s] for s in syms])
    C = np.corrcoef(M)
    matrix = {syms[i]: {syms[j]: round(float(C[i, j]), 2) for j in range(len(syms))}
              for i in range(len(syms))}
    avg = {s: round(float(np.mean([C[i, j] for j in range(len(syms)) if j != i])), 2)
           for i, s in enumerate(syms)}
    pairs = [(syms[i], syms[j], round(float(C[i, j]), 2))
             for i in range(len(syms)) for j in range(i + 1, len(syms))]
    pairs.sort(key=lambda x: -x[2])
    return {"lookbackDays": len(common), "symbols": syms, "matrix": matrix, "avgCorrelation": avg,
            "mostCorrelated": [{"pair": f"{a}/{b}", "corr": c} for a, b, c in pairs[:3]],
            "leastCorrelated": [{"pair": f"{a}/{b}", "corr": c} for a, b, c in pairs[-3:]],
            "portfolioAvgCorr": round(float(np.mean([p[2] for p in pairs])), 2),
            "note": "High average correlation = less true diversification than the ticker count implies."}


@mcp.tool()
async def account_growth(span: str = "year") -> dict:
    """Risk/return profile of your CURRENT holdings over the period: total return, CAGR, annualized
    volatility, max drawdown, and a rough Sharpe - computed by valuing today positions back through
    price history. Robinhood removed account-equity history, so this is the current allocation
    historical profile, not your actual past equity. span: month, 3month, year, 5year.
    """
    import asyncio
    import numpy as np
    span = span if span in ("month", "3month", "year", "5year") else "year"
    pos = await _robinhood_positions()
    eq = [(p["symbol"], p.get("qty") or 0.0) for p in pos
          if p.get("type") == "equity" and (p.get("qty") or 0) > 0]
    if not eq:
        return {"note": "No equity holdings."}
    syms = [s for s, _ in eq]
    series = await asyncio.to_thread(_hist_closes_sync, syms, span)
    have = {s: series.get(s, {}) for s in syms if series.get(s)}
    syms = [s for s in syms if s in have]
    if not syms:
        return {"note": "No price history available."}
    common = sorted(set.intersection(*[set(have[s].keys()) for s in syms]))
    if len(common) < 10:
        return {"note": f"Only {len(common)} overlapping days of history."}
    qty = {s: q for s, q in eq}
    port = np.array([sum(qty[s] * have[s][d] for s in syms) for d in common])
    rets = np.diff(np.log(port))
    yrs = len(common) / 252.0
    cagr = (port[-1] / port[0]) ** (1 / yrs) - 1 if yrs > 0 and port[0] > 0 else None
    vol = float(np.std(rets) * np.sqrt(252))
    peak = np.maximum.accumulate(port)
    maxdd = float(((port - peak) / peak).min())
    sharpe = float((np.mean(rets) * 252 - RISK_FREE) / vol) if vol > 0 else None
    return {"span": span, "days": len(common), "holdings": len(syms),
            "startValue$": round(float(port[0]), 2), "endValue$": round(float(port[-1]), 2),
            "totalReturn%": round(100 * (port[-1] / port[0] - 1), 2),
            "cagr%": round(100 * cagr, 2) if cagr is not None else None,
            "annualizedVol%": round(100 * vol, 2), "maxDrawdown%": round(100 * maxdd, 2),
            "sharpe": round(sharpe, 2) if sharpe is not None else None,
            "note": "Synthetic: current holdings valued through history; not actual account equity."}


def _spy_live_sync():
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        p = rh.stocks.get_latest_price("SPY", includeExtendedHours=True)
        return _to_float(p[0]) if p and p[0] else None
    except Exception:
        return None


@mcp.tool()
async def spot_blend(spy_mult: float = 10.0, basis: Optional[float] = None) -> dict:
    """De-stale the gamma map: compares the delayed CBOE chain spot (~15 min) to a live SPY-implied SPX
    (SPY x spy_mult + basis) and reports the gap and whether spot has likely crossed the gamma flip or a
    wall since the snapshot. SPYx10 carries a ~20-40pt dividend basis to SPX - pass basis to calibrate
    against a real SPX print.
    """
    import asyncio
    z = await zero_dte_exposure()
    if isinstance(z, dict) and "error" in z:
        return {"error": z["error"]}
    spy = await asyncio.to_thread(_spy_live_sync)
    chain_spot, flip = z.get("spot"), z.get("gammaFlip")
    cw = (z.get("callWall") or {}).get("strike")
    pw = (z.get("putWall") or {}).get("strike")
    asof = z.get("asof")
    stl = _staleness(asof)
    age_min = stl["ageMin"]
    spx_idx = None
    if basis is None:
        # Real independent print (RH index, else E*TRADE) before the circular _auto_basis fallback.
        spx_idx, spx_src = await _live_spx_print()
        if spx_idx and spy:
            basis, basis_src = round(spx_idx - spy * spy_mult, 2), f"{spx_src} (SPX print)"
        else:
            basis, basis_src = _auto_basis(chain_spot, asof, spy, spy_mult)
    else:
        basis_src = "manual"
    spx_est = (spy * spy_mult + basis) if (spy and basis is not None) else None
    gap = round(spx_est - chain_spot, 1) if (spx_est and chain_spot) else None

    def side(level):
        if not (spx_est and chain_spot and level):
            return None
        was = "above" if chain_spot > level else "below"
        now = "above" if spx_est > level else "below"
        return {"level": level, "chainSide": was, "liveSide": now, "crossed": was != now}

    return {"chainSpot": chain_spot, "chainAgeMin": age_min, "freshness": stl, "spyLive": spy,
            "spxIndexLive": spx_idx, "spyMult": spy_mult, "basis": basis, "basisSource": basis_src,
            "spxLiveEst": round(spx_est, 1) if spx_est is not None else None, "gapPts": gap,
            "vsGammaFlip": side(flip), "vsCallWall": side(cw), "vsPutWall": side(pw),
            "note": ("spxLiveEst = SPY*mult + basis; basis auto-calibrated from the chain when fresh "
                     "(see basisSource). Pass basis= to override.")}


@mcp.tool()
async def pcs_sizer(short_delta: float = 0.30, width: float = 10.0,
                    expiry: Optional[str] = None) -> dict:
    """Size an SPX put credit spread (your ASD 0DTE PCS): from the live chain it picks the short put
    nearest short_delta and the long put width points below, then reports net credit, max loss,
    breakeven, return-on-risk, and an approximate probability of profit. expiry defaults to 0DTE.
    """
    try:
        chain = await _load_chain_smart(zero_dte=(expiry is None), expiration=expiry, root="SPXW")
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = chain.get("spot")
    opts = chain.get("options") or []
    exp = expiry or _nearest_expiry(opts, "SPXW")
    puts = [o for o in _filter(opts, root="SPXW", expiration=exp)
            if o.get("cp") == "P" and o.get("strike")]
    if not puts:
        return {"error": f"No SPXW puts found for {exp}."}
    T = _year_frac(exp)

    def _put_delta(o):
        # B4: prefer a real feed delta, but if it is missing OR exactly 0.0 (RH's "unknown"),
        # recompute abs(delta) from the strike's IV. The old code accepted delta==0.0 as valid,
        # which -- when the whole feed's deltas came back 0 -- sized a nonsense short strike and
        # reported approxPOP 100%.
        d = o.get("delta")
        if d is not None and abs(d) > 1e-6:
            return abs(d)
        iv, K = o.get("iv"), o.get("strike")
        if iv and iv > 0 and K and spot:
            dd, _g, _v, _c = _greeks_vec(spot, np.array([float(K)]), np.array([float(T)]),
                                         np.array([float(iv)]), np.array([False]))
            return abs(float(dd[0]))
        return None

    scored = [(o, _put_delta(o)) for o in puts]
    with_delta = [(o, dl) for (o, dl) in scored if dl is not None]
    if not with_delta:
        return {"error": "No usable delta on puts (feed delta missing/zero and no IV to recompute)."}
    short, short_d = min(with_delta, key=lambda od: abs(od[1] - short_delta))
    ks = short["strike"]
    longs = [o for o in puts if o["strike"] <= ks - 0.01]
    if not longs:
        return {"error": "No long put strike below the short available."}
    longp = min(longs, key=lambda o: abs(o["strike"] - (ks - width)))
    long_d = next((dl for (o, dl) in scored if o is longp), None)
    actual_width = round(ks - longp["strike"], 2)
    sc, lc = (short.get("mid") or 0.0), (longp.get("mid") or 0.0)
    credit = round(sc - lc, 2)
    maxloss = round(actual_width - credit, 2)
    return {"expiry": exp, "spot": spot,
            "source": chain.get("source"), "freshness": chain.get("freshness"),
            "shortPut": {"strike": ks, "delta": round(short_d, 3), "mid": round(sc, 2)},
            "longPut": {"strike": longp["strike"], "delta": round(long_d or 0.0, 3),
                        "mid": round(lc, 2)},
            "width": actual_width, "creditPerSh$": credit, "creditPerContract$": round(credit * 100, 2),
            "maxLossPerSh$": maxloss, "maxLossPerContract$": round(maxloss * 100, 2),
            "breakeven": round(ks - credit, 2),
            "returnOnRisk%": round(100 * credit / maxloss, 1) if maxloss > 0 else None,
            "approxPOP%": round(100 * (1 - short_d), 1),
            "note": ("approxPOP ~ P(short put expires OTM) = 1-abs(delta); true POP is a bit higher. "
                     "Deltas are recomputed from IV when the feed's delta is missing or 0 (B4). "
                     "Feed in 'source'/'freshness': Robinhood live when market open, else CBOE ~15-min delayed.")}


@mcp.tool()
async def event_risk_radar(days: Annotated[int, Field(ge=1, le=60)] = 7) -> dict:
    """What can gap your book in the next N days: high-impact economic events plus any holdings
    reporting earnings, merged into one timeline flagged by what you hold. Combines the economic and
    single-name earnings calendars with your positions.
    """
    import asyncio
    ec, ear, pos = await asyncio.gather(economic_calendar(days=days), earnings_calendar(days=days),
                                        _robinhood_positions(), return_exceptions=True)
    today = _today_et()
    horizon = today + _dt.timedelta(days=days)
    items = []
    if isinstance(ec, dict):
        for e in (ec.get("events") or []):
            try:
                dd = _dt.date.fromisoformat(e["date"][:10])
            except Exception:
                continue
            if today <= dd <= horizon and e.get("importance") == "high":
                items.append({"date": e["date"], "type": "economic", "name": e.get("name"),
                              "importance": "high", "daysAway": (dd - today).days})
    # B6: an options trader's "held" names include the underlyings of option positions, not just
    # outright shares -- otherwise earnings on a name you only hold via options look "unheld".
    held = set()
    if isinstance(pos, list):
        for p in pos:
            if p.get("type") == "equity" and p.get("symbol"):
                held.add(p["symbol"])
            elif p.get("type") == "option" and p.get("underlying"):
                held.add(p["underlying"])
    if isinstance(ear, dict):
        for r in (ear.get("nextEarnings") or []):
            if r.get("withinWindow"):
                items.append({"date": r["date"], "type": "earnings", "name": f"{r['symbol']} earnings",
                              "session": r.get("session"), "held": r["symbol"] in held,
                              "daysAway": r.get("daysAway")})
    items.sort(key=lambda x: (x["daysAway"] if x.get("daysAway") is not None else 999))
    return {"windowDays": days, "asof": today.isoformat(), "eventCount": len(items), "timeline": items,
            "note": "High-impact macro + earnings in your holdings. Size down into binary events."}


@mcp.tool()
async def estimated_tax(year: Optional[int] = None, fed_rate: float = 0.35,
                        state_rate: float = 0.0499, include_niit: bool = True) -> dict:
    """Estimated tax set-aside on your realized trading gains: pulls YTD realized short/long-term options
    P&L and applies your marginal federal + Georgia rates, with a quarterly figure. Short-term is taxed
    as ordinary income. Georgia's 2026 flat rate is 4.99% (HB 463). include_niit adds the 3.8% Net
    Investment Income Tax (applies above ~$200k single / $250k MFJ MAGI). Trading gains only - excludes
    W-2/Schedule C/clergy; not tax advice.
    """
    ts = await tax_summary(year=year)
    if "realizedTotal$" not in ts:
        base = {k: ts[k] for k in ("year", "error", "note") if k in ts}
        return {**base, "note": ts.get("note") or "No realized P&L to estimate."}
    st = ts.get("shortTerm$", 0.0) or 0.0
    lt = ts.get("longTerm$", 0.0) or 0.0
    niit = 0.038 if include_niit else 0.0
    fed_st = max(0.0, st) * (fed_rate + niit)
    st_state = max(0.0, st) * state_rate
    ltcg = max(0.0, lt) * (0.15 + niit)
    lt_state = max(0.0, lt) * state_rate
    total = fed_st + st_state + ltcg + lt_state
    out = {"year": ts.get("year"), "realizedShortTerm$": round(st, 2), "realizedLongTerm$": round(lt, 2),
           "assumedFedRate": fed_rate, "assumedStateRate": state_rate,
           "niitApplied": include_niit, "niitRate": niit,
           "estShortTermTax$": round(fed_st + st_state, 2), "estLongTermTax$": round(ltcg + lt_state, 2),
           "estTotalSetAside$": round(total, 2), "quarterlySetAside$": round(total / 4, 2),
           "note": ("Trading gains only; short-term taxed as ordinary income; LTCG modeled at 15% (a 20% "
                    "bracket applies at high income). Georgia flat 4.99% (2026, HB 463). Excludes W-2 / "
                    "Schedule C / clergy housing. This set-aside covers YTD realized gains SO FAR and "
                    "does NOT cover the full year. Verify with your CPA.")}
    if ts.get("WARNING_TRUNCATED"):
        out["WARNING_TRUNCATED"] = ("Underlying order history was truncated, so realized P&L (and this "
                                    "set-aside) is INCOMPLETE. " + str(ts.get("WARNING_TRUNCATED")))
        out["dataBeginsAt"] = ts.get("dataBeginsAt")
    return out


# ============================================================================
# Feed freshness guards + live-spot calibration
#   (added: staleness detection and auto-calibrated SPY->SPX basis)
# ============================================================================
_basis_cache: dict = {}   # {"basis": float, "ts": monotonic_seconds, "spot": float}


def _market_open_et(now: Optional[_dt.datetime] = None) -> bool:
    """True if the US equity cash session is open right now (a regular trading day, 09:30 to the
    session close ET). B8: NYSE holidays are excluded and half days close at 13:00 ET."""
    n = now or _dt.datetime.now(ET)
    if n.tzinfo is None:
        n = n.replace(tzinfo=ET)
    n = n.astimezone(ET)
    if not _is_trading_day(n.date()):
        return False
    close_t = _session_close_et(n.date())
    mins = n.hour * 60 + n.minute
    return (9 * 60 + 30) <= mins < (close_t.hour * 60 + close_t.minute)


def _parse_asof_et(asof: Optional[str]) -> Optional[_dt.datetime]:
    """Parse a CBOE asof timestamp into an aware UTC datetime. CBOE's last_trade_time is ISO
    *without* a timezone and is Eastern, so naive values are interpreted as ET."""
    if not asof or not isinstance(asof, str):
        return None
    try:
        d = _dt.datetime.fromisoformat(asof.replace("Z", "+00:00"))
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=ET)
    return d.astimezone(_dt.timezone.utc)


def _staleness(asof: Optional[str], fresh_max_min: float = 16.0,
               stale_min: float = 20.0) -> dict:
    """Classify how fresh a CBOE timestamp is. CBOE is inherently ~15 min delayed, so while the cash
    session is open 'fresh' means within that normal delay; an age beyond ~20 min during market hours
    means the feed itself is lagging (stale). Outside the session the print is just the prior close."""
    d_utc = _parse_asof_et(asof)
    open_now = _market_open_et()
    if d_utc is None:
        return {"asof": asof, "ageMin": None, "marketOpen": open_now,
                "stale": None, "verdict": "unknown"}
    age_min = round((_dt.datetime.now(_dt.timezone.utc) - d_utc).total_seconds() / 60.0, 1)
    if not open_now:
        verdict, stale = "after-hours", False
    elif age_min <= fresh_max_min:
        verdict, stale = "fresh", False
    elif age_min <= stale_min:
        verdict, stale = "lagging", False
    else:
        verdict, stale = "stale", True
    return {"asof": asof, "ageMin": age_min, "marketOpen": open_now,
            "stale": stale, "verdict": verdict}


def _auto_basis(chain_spot: Optional[float], asof: Optional[str], spy_live: Optional[float],
                mult: float) -> tuple:
    """Auto-calibrate the offset in SPX = SPY*mult + basis without a second live feed.

    The SPX/SPY basis (driven by accumulated dividends) is stable intraday. We compute it as
    chain_spot - spy_live*mult while the CBOE chain is fresh (delayed SPX ~ live within the normal
    lag) and cache it; once the chain goes stale, that cached fresh basis is reused with the *live*
    SPY, yielding an accurate live SPX even though the chain spot is old.

    Returns (basis, source) with source in 'live-calibrated' / 'cached-fresh' / 'uncalibrated' /
    'unavailable'.
    """
    stl = _staleness(asof)
    if chain_spot and spy_live:
        live_basis = chain_spot - spy_live * mult
        if stl["verdict"] in ("fresh", "after-hours"):
            _basis_cache.update({"basis": live_basis, "ts": time.monotonic(), "spot": chain_spot})
            return round(live_basis, 2), "live-calibrated"
        cached = _basis_cache.get("basis")
        if cached is not None and (time.monotonic() - _basis_cache.get("ts", 0)) < 6 * 3600:
            return round(cached, 2), "cached-fresh"
        return round(live_basis, 2), "uncalibrated"
    cached = _basis_cache.get("basis")
    if cached is not None:
        return round(cached, 2), "cached-fresh"
    return None, "unavailable"


@mcp.tool()
async def feed_health() -> dict:
    """One-stop 'can I trust the data right now?' for 0DTE work. Reports CBOE chain freshness (age and a
    stale flag judged against market hours) and overlays a live SPY-implied SPX (auto-calibrated basis)
    so you can see the live level even when the chain spot is lagging. Check it before acting on gamma
    levels, max-pain, or expected-move distances.
    """
    import asyncio
    out: dict = {"source": "CBOE delayed options JSON (~15 min) + live SPY overlay"}
    ch = None
    for attempt in range(2):
        try:
            ch = await _load_chain()
            break
        except Exception as exc:  # noqa: BLE001
            out["chainError"] = str(exc)[:160]
            if attempt == 0:
                await asyncio.sleep(0.8)
    if not ch:
        out["reachable"] = False
        out["verdict"] = "DOWN"
        out["recommendation"] = ("CBOE chain unreachable - do not rely on gamma levels; price off your "
                                 "broker quote until it recovers.")
        return out
    chain_spot = ch.get("spot")
    stl = _staleness(ch.get("asof"))
    out["reachable"] = True
    out["chainSpot"] = round(chain_spot, 2) if chain_spot else None
    out["freshness"] = stl
    out["contracts"] = len(ch.get("options") or [])

    # B3: the 0DTE tools now use the Robinhood LIVE chain as PRIMARY (CBOE is only the fallback),
    # so surface RH's health here too -- otherwise feed_health can report "OK" about a feed the
    # maps no longer lean on. Probe today's RH SPXW chain with a short timeout.
    rh_probe: dict = {}
    try:
        rh_ch = await asyncio.wait_for(_rh_chain_cached(_today_et().isoformat()), timeout=12)
        if rh_ch and rh_ch.get("options"):
            rh_probe = {"up": True, "contracts": len(rh_ch.get("options") or []),
                        "paritySpot": round(rh_ch["spot"], 2) if rh_ch.get("spot") else None,
                        "verdict": "OK"}
        elif not _is_trading_day(_today_et()):
            rh_probe = {"up": False, "verdict": "CLOSED",
                        "note": "Not a trading day; RH has no live 0DTE chain."}
        else:
            rh_probe = {"up": False, "verdict": "NO_TODAY_EXPIRY",
                        "note": "RH returned no contracts expiring today (series may have rolled)."}
    except asyncio.TimeoutError:
        rh_probe = {"up": False, "verdict": "TIMEOUT", "note": "RH chain probe timed out (>12s)."}
    except Exception as exc:  # noqa: BLE001
        rh_probe = {"up": False, "verdict": "ERROR", "error": str(exc)[:160]}
    out["rhChain"] = rh_probe

    spy = None
    try:
        spy = await asyncio.to_thread(_spy_live_sync)
    except Exception:  # noqa: BLE001
        spy = None
    basis, src = _auto_basis(chain_spot, ch.get("asof"), spy, 10.0)
    spx_est = round(spy * 10.0 + basis, 1) if (spy and basis is not None) else None
    # An INDEPENDENT live SPX print is what makes this tool mean anything. Without one, the
    # "uncalibrated" branch of _auto_basis sets basis = chainSpot - spy*10, so spxLiveEst comes
    # out IDENTICAL to chainSpot and gapVsChainPts is identically 0.0 -- the staleness check
    # silently reports a stale price as live (this happened 2026-07-15). Prefer a real print
    # (Robinhood index, else E*TRADE) and be loud when neither is available.
    spx_live, spx_src = await _live_spx_print()
    if spx_live:
        out["spxLive"] = spx_live
        out["spxLiveSource"] = spx_src
        spx_est = round(spx_live, 1)
        if spy:
            basis, src = round(spx_live - spy * 10.0, 2), f"{spx_src} (live SPX print)"
    out["spyLive"] = spy
    out["basis"] = basis
    out["basisSource"] = src
    out["spxLiveEst"] = spx_est
    out["gapVsChainPts"] = round(spx_est - chain_spot, 1) if (spx_est and chain_spot) else None
    if not spx_live and src == "uncalibrated":
        out["CANNOT_VERIFY"] = True
        out["warning"] = ("No independent live SPX print (Robinhood index endpoint AND E*TRADE both "
                          "unavailable) and no cached basis. spxLiveEst is derived FROM chainSpot, so "
                          "it EQUALS chainSpot and gapVsChainPts is meaningless -- this tool cannot "
                          "detect staleness right now. Price off your broker quote.")
    v = stl["verdict"]
    if v == "stale":
        out["verdict"] = "STALE"
        out["recommendation"] = ("Chain spot is lagging >20 min while the market is open - trust "
                                 "spxLiveEst over chainSpot for distance-to-flip/walls.")
    elif v == "lagging":
        out["verdict"] = "LAGGING"
        out["recommendation"] = "Chain is slightly behind; cross-check levels against spxLiveEst."
    elif v == "after-hours":
        out["verdict"] = "CLOSED"
        out["recommendation"] = "Market closed - the chain shows the last session's close."
    elif v == "unknown":
        out["verdict"] = "UNKNOWN"
        out["recommendation"] = "Could not parse the feed timestamp; treat freshness as unverified."
    else:
        out["verdict"] = "OK"
        out["recommendation"] = "Chain is within its normal ~15-min delay."
    if spy is None:
        out["note"] = ("Live SPY overlay unavailable (broker session) - spxLiveEst may be absent and "
                       "basis fell back to cache/none.")
    # (#8) feed_health v2: a verdict per feed the desk actually stands on + a composite, so a green
    # CBOE verdict can't mask a down RH-live chain (the feed the smart 0DTE tools now use as primary).
    rhc = out.get("rhChain") or {}
    feeds = {
        "cboeChain": {"verdict": out.get("verdict"), "ageMin": (stl or {}).get("ageMin")},
        "rhChain": {"verdict": rhc.get("verdict"), "contracts": rhc.get("contracts")},
        "spxPrint": {"verdict": ("OK" if spx_live else "DOWN"), "source": spx_src, "value": spx_live},
    }
    rh_ok = rhc.get("verdict") == "OK"
    cboe_ok = out.get("verdict") in ("OK", "LAGGING")
    if rh_ok and spx_live:
        composite = "OK"            # primary 0DTE feed live AND an independent print to check it
    elif rh_ok or cboe_ok:
        composite = "DEGRADED"      # a usable chain, but verification is thin
    else:
        composite = "DOWN"          # no trustworthy chain
    out["feeds"] = feeds
    out["compositeVerdict"] = composite
    return out


def _selftest() -> int:
    """(#11b) `python traders_edge_mcp.py --selftest` -- fast OFFLINE sanity of the plumbing before you
    restart Claude Desktop. Pure math + fill decomposition; no network, no broker session touched."""
    import datetime as _d
    fails = []

    def ck(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    ck("norm_cdf(0)=.5", abs(float(_norm_cdf(np.array([0.0]))[0]) - 0.5) < 1e-6)
    ck("2026-07-03 is a holiday", not _is_trading_day(_d.date(2026, 7, 3)))
    ck("half-day close 13:00", _session_close_et(_d.date(2026, 11, 27)) == _d.time(13, 0))
    ck("redact strips api_key", "SECRET" not in _redact_url("http://x/?api_key=SECRET"))

    def _leg(oid, side, eff, k, ot, px, ts):
        return {"option": f"x/{oid}/", "side": side, "position_effect": eff, "strike_price": str(k),
                "option_type": ot, "expiration_date": "2026-07-20",
                "executions": [{"timestamp": ts, "trade_date": "2026-07-20", "price": str(px),
                                "quantity": "1"}], "ratio_quantity": "1"}

    o1 = {"id": "O1", "state": "filled", "chain_symbol": "SPXW", "updated_at": "2026-07-20T14:30:00Z",
          "created_at": "2026-07-20T14:30:00Z", "net_amount": "300", "net_amount_direction": "credit",
          "processed_premium": None, "processed_quantity": "1", "quantity": "1",
          "legs": [_leg("a", "sell", "open", 6800, "put", 5.0, "2026-07-20T14:30:00Z"),
                   _leg("b", "buy", "open", 6790, "put", 2.0, "2026-07-20T14:30:00Z")]}
    o2 = {"id": "O2", "state": "filled", "chain_symbol": "SPXW", "updated_at": "2026-07-20T15:30:00Z",
          "created_at": "2026-07-20T15:30:00Z", "net_amount": "50", "net_amount_direction": "debit",
          "processed_premium": None, "processed_quantity": "1", "quantity": "1",
          "legs": [_leg("a", "buy", "close", 6800, "put", 1.0, "2026-07-20T15:30:00Z"),
                   _leg("b", "sell", "close", 6790, "put", 0.5, "2026-07-20T15:30:00Z")]}
    f = _order_to_fills(o1) + _order_to_fills(o2)
    trips, _stats = _round_trips_full(f)
    ck("spread -> 2 trips, P&L == net cash",
       len(trips) == 2 and round(sum(t["pnl"] for t in trips), 2) == round(sum(x["net_cf"] for x in f), 2))
    ck("mi_ratio date-aligns",
       _mi_ratio({"d1": 10.0, "d2": 20.0, "d3": 99.0}, {"d1": 2.0, "d2": 5.0}) == [5.0, 4.0])
    print("SELFTEST " + ("FAILED: " + ", ".join(fails) if fails else "PASSED"))
    return 1 if fails else 0


def main() -> None:
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    log.info("Starting Traders Edge MCP server (stdio); risk-free=%.3f", RISK_FREE)
    mcp.run()


# ============================================================================
# ROBINHOOD-NATIVE ADDITIONS (v0.9.0)
#   realized_pnl        - official-vs-reconstruction realized-P&L validator + feed
#   index_quote         - live SPX/VIX/NDX index levels (feeds spot_blend basis)
#   watchlist_radar     - catalyst radar across a named Robinhood watchlist
#   earnings_results    - per-symbol EPS actual/estimate/surprise history
#   equity_fundamentals - per-symbol P/E, mkt cap, div yield, sector, 52wk range
# All reuse the existing robin_stocks session helpers (_rh_login_sync, request_get,
# _round_trips, _next_earnings, _div_info_sync, _project_ex, _to_float).
# ============================================================================

# ---- Realized P&L: authoritative reconciliation of the FIFO reconstruction ----
# Optional: Robinhood's account-scoped "PnL hub" (internally "Wormhole") realized-P&L endpoint. It is
# undocumented; when RH_PNL_HUB_URL is set (a URL template that may contain {span} and {account}) it is
# attached as the top-level official source. Capture it from the app's Realized-P&L page in browser
# devtools. Until then, the fee-inclusive round-trip figure below is the authoritative number.
RH_PNL_HUB_URL = os.environ.get("RH_PNL_HUB_URL", "").strip()

_FEE_KEYS = ("regulatory_fees", "total_regulatory_fees", "sec_fees", "orf_fees", "contract_fees")


def _rh_order_fees(o: dict) -> float:
    """Best-effort total fees on one RH option order (reg / exchange / ORF passthrough).

    RH options are commission-free but pass a small per-contract regulatory fee on sells; the FIFO
    reconstruction in daily_pnl_curve / tax_summary ignores it, which is a chief source of the drift
    this validator surfaces. Sums the known top-level fee keys plus any per-execution 'fees'.
    """
    total = 0.0
    for k in _FEE_KEYS:
        total += _to_float(o.get(k)) or 0.0
    for lg in (o.get("legs") or []):
        for ex in (lg.get("executions") or []):
            total += _to_float(ex.get("fees")) or 0.0
    return round(total, 4)


def _rh_realized_recon_sync(date_iso: str) -> dict:
    """Independent realized-P&L measures for one ET date, from the raw RH option orders.

    Returns the same round-trip realized figure daily_pnl_curve computes (fee-less), plus a
    fee-inclusive figure and the raw net cash flow, so the three can be cross-checked. The
    fee-inclusive number is the one that should reconcile to Robinhood's official PnL hub.
    """
    target = _dt.date.fromisoformat(date_iso)
    raw, ometa = _rh_recent_option_orders(target)
    fills, fees, n_orders = [], 0.0, 0
    for o in raw:
        day_legs = [f for f in _order_to_fills(o) if f["trade_date"] == date_iso]   # B2
        if day_legs:
            fills.extend(day_legs)
            fees += _rh_order_fees(o)                  # B7: fees counted ONCE per ORDER, not per leg
            n_orders += 1
    fills.sort(key=lambda r: r["time"])
    trips, tstats = _round_trips_full(fills)
    rt_realized = round(sum(t["pnl"] for t in trips), 2)   # fee-less; == daily_pnl_curve realized$
    net_cf = round(sum(f["net_cf"] for f in fills), 2)
    fee_incl = round(rt_realized - fees, 2)
    out = {"date": date_iso, "orders": n_orders, "legFills": len(fills), "roundTrips": len(trips),
           "roundTripRealized$": rt_realized, "fees$": round(fees, 2),
           "feeInclusiveRealized$": fee_incl, "netCashFlow$": net_cf,
           "openOrExpiredDelta$": round(net_cf - rt_realized, 2)}
    if tstats.get("unmatchedCloses") or tstats.get("orderLevelFallback"):
        out["reconStats"] = tstats
    if ometa.get("truncated"):
        out["truncated"] = True
    return out


def _rh_pnl_hub_sync(span: str, account_number):
    """Pull Robinhood's official realized P&L ('PnL hub' / Wormhole) if RH_PNL_HUB_URL is configured.

    The endpoint is account-scoped and undocumented; set RH_PNL_HUB_URL to the captured URL template
    (may contain {span} and {account}) to enable. Returns the parsed payload, or None when unset/failed.
    """
    if not RH_PNL_HUB_URL:
        return None
    from robin_stocks.robinhood.helper import request_get
    _rh_login_sync()
    url = RH_PNL_HUB_URL.replace("{span}", span).replace("{account}", account_number or "")
    try:
        data = request_get(url)
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


async def _rh_realized_today():
    """Fee-inclusive realized P&L for today from RH round trips - the primary daily_target feed."""
    import asyncio
    try:
        return await asyncio.to_thread(_rh_realized_recon_sync, _today_et().isoformat())
    except Exception:  # noqa: BLE001
        return None


@mcp.tool()
async def realized_pnl(date: Optional[str] = None, span: Optional[str] = None,
                       account_number: Optional[str] = None) -> dict:
    """Authoritative realized-P&L check: reconciles the FIFO reconstruction (daily_pnl_curve /
    tax_summary) against fee-inclusive round trips and, when configured, Robinhood's official PnL hub.

    Use it to (a) catch drift/bugs in the reconstruction - the diff isolates fees, expiries/assignments,
    and rounding - and (b) get a trustworthy realized figure for the session. `date` (YYYY-MM-DD ET)
    checks one day (default today). `span` (week|month|3month|ytd|all) is passed to the official hub only,
    when RH_PNL_HUB_URL is set. See the reconciliation block for exactly where the numbers disagree.
    """
    import asyncio
    d = date or _today_et().isoformat()
    try:
        recon = await asyncio.to_thread(_rh_realized_recon_sync, d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    feeless = recon["roundTripRealized$"]          # the number daily_pnl_curve/tax_summary report
    authoritative = recon["feeInclusiveRealized$"]
    out = {"date": d,
           "reconstructionSource": "robinhood option fills (FIFO round trips)",
           "reportedByReconstruction$": feeless, "feeInclusive$": authoritative,
           "reconstruction": recon}

    checks = []
    if abs(recon["fees$"]) >= 0.01:
        checks.append(f"daily_pnl_curve/tax_summary omit fees: fee-inclusive realized is "
                      f"${authoritative:.2f} vs ${feeless:.2f} reconstructed "
                      f"(${recon['fees$']:.2f} in fees).")
    if abs(recon["openOrExpiredDelta$"]) > 1.0:
        checks.append(f"Net cash flow (${recon['netCashFlow$']:.2f}) differs from round-trip realized "
                      f"by ${recon['openOrExpiredDelta$']:.2f} - a leg expired/assigned or is still open "
                      f"(not paired into a round trip).")
    if recon.get("truncated"):
        checks.append("Order history hit the page cap for this lookback - if a position opened on an "
                      "earlier date, its close today may be unmatched. Figures may be incomplete.")
    if recon.get("reconStats"):
        checks.append(f"Round-trip matching flagged data-quality issues: {recon['reconStats']}.")
    if len(checks) == 0:
        checks.append("Reconstruction reconciles cleanly (no fees, no unpaired legs) for this date.")

    hub = await asyncio.to_thread(_rh_pnl_hub_sync, (span or "day"), account_number)
    if hub is not None:
        out["officialHub"] = hub
        out["officialHubSource"] = RH_PNL_HUB_URL
        checks.append("Official RH PnL-hub payload attached under officialHub - diff it against "
                      "feeInclusive$ to validate the reconstruction end-to-end.")
    elif RH_PNL_HUB_URL:
        checks.append("RH_PNL_HUB_URL is set but the hub call returned nothing - verify the URL "
                      "template and that the account is reachable by this session.")
    else:
        checks.append("Set RH_PNL_HUB_URL to Robinhood's PnL-hub endpoint to attach the official number "
                      "as the top-level source; until then feeInclusive$ is the authoritative figure.")

    out["reconciliation"] = checks
    return out


# ---- Live index levels (SPX / VIX / NDX) via Robinhood marketdata ----
_RH_INDEX_IDS = {
    "SPX": os.environ.get("RH_SPX_ID", "432fbbb8-b82c-454a-852d-eb85382c7066"),
    "VIX": os.environ.get("RH_VIX_ID", "3b912aa2-88f9-4682-8ae3-e39520bdf4db"),
    "NDX": os.environ.get("RH_NDX_ID", "50c298f7-27a8-44a1-b049-ec153cf2892f"),
}
# RH's index-quote endpoint is not officially documented. These candidate templates are tried in order
# (first hit is cached for the process); override/pin with RH_INDEX_QUOTE_URL ({ids}=comma-joined UUIDs).
_RH_INDEX_URL_CANDIDATES = [
    "https://api.robinhood.com/marketdata/indices/quotes/?ids={ids}",
    "https://api.robinhood.com/marketdata/index/quotes/?ids={ids}",
    "https://api.robinhood.com/marketdata/quotes/{id}/",
]
_rh_index_url_cache = {"url": None}


def _rh_index_quote_sync(symbols: str) -> dict:
    """Live index levels for one or more symbols (comma-separated, e.g. 'SPX,VIX').

    Returns {SYMBOL: {value, asof, source}}. Robinhood's index marketdata endpoint is undocumented, so
    the first working URL template is discovered once and cached; set RH_INDEX_QUOTE_URL to pin it.
    """
    from robin_stocks.robinhood.helper import request_get
    _rh_login_sync()
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    id_for = {s: _RH_INDEX_IDS.get(s) for s in syms}
    ids = [i for i in id_for.values() if i]
    if not ids:
        return {}
    ids_join = ",".join(ids)
    env_url = os.environ.get("RH_INDEX_QUOTE_URL", "").strip()
    candidates = [env_url] if env_url else []
    if _rh_index_url_cache["url"]:
        candidates.append(_rh_index_url_cache["url"])
    candidates += _RH_INDEX_URL_CANDIDATES

    payload = None
    for tmpl in candidates:
        if not tmpl:
            continue
        url = tmpl.replace("{ids}", ids_join).replace("{id}", ids[0])
        try:
            data = request_get(url)
        except Exception:  # noqa: BLE001
            continue
        rows = None
        if isinstance(data, dict):
            rows = data.get("results") or data.get("quotes")
            if rows is None and data.get("value") is not None:
                rows = [data]
        elif isinstance(data, list):
            rows = data
        if rows:
            payload = rows
            _rh_index_url_cache["url"] = tmpl
            break
    if not payload:
        return {}
    by_id = {}
    for row in payload:
        by_id[row.get("instrument_id") or row.get("id")] = row
    out = {}
    for s, iid in id_for.items():
        row = by_id.get(iid)
        if not row:
            continue
        out[s] = {"value": _to_float(row.get("value") or row.get("last_trade_price")),
                  "asof": row.get("venue_timestamp") or row.get("updated_at"),
                  "source": "robinhood_index"}
    return out


@mcp.tool()
async def index_quote(symbols: str = "SPX,VIX") -> dict:
    """Live index levels (SPX / VIX / NDX) - a real-time print to de-stale the ~15-min CBOE chain
    and calibrate the SPY->SPX basis used by spot_blend and feed_health.

    Robinhood's index endpoint is primary; E*TRADE's market feed (independent, and it self-reports
    quoteStatus REALTIME/DELAYED) is the fallback, so a single broken endpoint no longer leaves the
    desk with zero live prints. `symbols` is comma-separated (default 'SPX,VIX'). Values track live
    during RTH; outside the session the print is the prior settle.
    """
    import asyncio
    q = {}
    try:
        q = await asyncio.to_thread(_rh_index_quote_sync, symbols) or {}
    except Exception as exc:  # noqa: BLE001 -- fall through to E*TRADE
        log.info("RH index quote failed (%s); trying E*TRADE", str(exc)[:140])
    if not q:
        try:
            q = await asyncio.to_thread(_etrade_index_quote_sync, symbols) or {}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not q:
        return {"error": ("No index quote from Robinhood OR E*TRADE. RH's endpoint may have moved "
                          "(set RH_INDEX_QUOTE_URL, {ids}=comma-joined UUIDs) and the E*TRADE token "
                          "may need re-authorizing via the etrade MCP."),
                "triedIds": _RH_INDEX_IDS}
    return {"indexes": q, "source": next(iter(q.values())).get("source"),
            "instrumentIds": {s: _RH_INDEX_IDS.get(s) for s in q if _RH_INDEX_IDS.get(s)},
            "note": "Live index print; feeds the spot_blend / feed_health basis when SPX is present."}


# ---- Watchlist radar: run the catalyst analytics across a named RH watchlist ----
def _rh_watchlist_symbols_sync(name: str) -> list:
    """Equity/ETF symbols in a named Robinhood watchlist (custom or followed), order-preserving."""
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    syms = []
    try:
        items = rh.account.get_watchlist_by_name(name) or {}
        results = items.get("results", items) if isinstance(items, dict) else items
        for it in (results or []):
            s = it.get("symbol")
            if not s:
                url = it.get("instrument")
                if url:
                    try:
                        s = (rh.helper.request_get(url) or {}).get("symbol")
                    except Exception:  # noqa: BLE001
                        s = None
            if s:
                syms.append(s.upper())
    except Exception:  # noqa: BLE001
        return []
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


@mcp.tool()
async def watchlist_radar(name: str, days: Annotated[int, Field(ge=1, le=120)] = 14) -> dict:
    """Catalyst radar across a named Robinhood watchlist: for every name, the next earnings date
    (BMO/AMC) and next ex-dividend date, flagged when they fall inside `days`, plus P/E and yield.

    Turns a saved list (e.g. 'Short Options', 'Weekly Dividend Stocks') into one event scan so you can
    see which names carry a binary/assignment event before the next covered-call or CSP cycle. Each name
    also carries a local technical read (trendScore + momentumScore, each -2..+2, plus exhaustion /
    rebound / death-cross flags) from daily bars. Names are matched by the exact list display name
    (case-sensitive).
    """
    import asyncio
    try:
        syms = await asyncio.to_thread(_rh_watchlist_symbols_sync, name)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not syms:
        return {"error": f"No equity symbols found in a watchlist named '{name}' "
                         f"(check the exact, case-sensitive list name)."}
    earn, div, fund, hist = await asyncio.gather(
        asyncio.gather(*[_next_earnings(s) for s in syms]),
        asyncio.to_thread(_div_info_sync, syms),
        asyncio.to_thread(_fundamentals_sync, syms),
        asyncio.to_thread(_hist_closes_sync, syms, "year"),
    )
    today = _today_et()
    rows, within = [], 0
    for s, e in zip(syms, earn):
        d_info = (div or {}).get(s) or {}
        proj = _project_ex(d_info) if d_info else {}
        nx_ex = proj.get("nextExDate")
        ex_days = (_dt.date.fromisoformat(nx_ex) - today).days if nx_ex else None
        e_days = e["daysAway"] if e else None
        f = (fund or {}).get(s) or {}
        soon = ((e_days is not None and e_days <= days) or (ex_days is not None and ex_days <= days))
        if soon:
            within += 1
        tScore = mScore = None
        exh = reb = dcx = None
        _cl = _ordered_closes(hist, s)
        if len(_cl) >= 35:
            _ci = _ind_compute(_cl)
            tScore, _ = _score_trend(_ci)
            mScore, _ = _score_momentum(_ci)
            _fl = _tech_flags(_ci)
            exh, reb, dcx = bool(_fl["exhaustion"]), bool(_fl["rebound"]), _fl["death_cross"]
        rows.append({"symbol": s,
                     "nextEarnings": (e or {}).get("date"), "earningsSession": (e or {}).get("session"),
                     "earningsDaysAway": e_days, "nextExDiv": nx_ex, "exDivDaysAway": ex_days,
                     "peRatio": f.get("peRatio"), "yieldPct": f.get("dividendYieldPct"),
                     "trendScore": tScore, "momentumScore": mScore,
                     "exhaustion": exh, "rebound": reb, "deathCross": dcx,
                     "eventWithinWindow": soon})

    def _proximity(r):
        cand = [x for x in (r["earningsDaysAway"], r["exDivDaysAway"]) if x is not None]
        return min(cand) if cand else 9999

    rows.sort(key=_proximity)
    return {"watchlist": name, "symbols": len(syms), "windowDays": days,
            "eventsWithinWindow": within, "radar": rows,
            "note": "Earnings from RH; ex-div projected (last ex + frequency); trendScore/momentumScore "
                    "(-2..+2) and exhaustion/rebound/deathCross flags from daily-bar technicals. "
                    "Verify before acting."}


# ---- Per-symbol earnings history (EPS actual / estimate / surprise) ----
def _rh_earnings_results_sync(symbol: str) -> dict:
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        rows = rh.stocks.get_earnings(symbol.upper()) or []
    except Exception as exc:  # noqa: BLE001
        raise EdgeError(f"RH earnings fetch failed for {symbol}: {str(exc)[:120]}")
    out = []
    for x in rows:
        eps = x.get("eps") or {}
        rep = x.get("report") or {}
        act = _to_float(eps.get("actual"))
        est = _to_float(eps.get("estimate"))
        surprise = round(act - est, 4) if (act is not None and est is not None) else None
        surprise_pct = (round(100.0 * (act - est) / abs(est), 1)
                        if (act is not None and est not in (None, 0.0)) else None)
        out.append({"year": x.get("year"), "quarter": x.get("quarter"),
                    "epsActual": act, "epsEstimate": est,
                    "surprise": surprise, "surprisePct": surprise_pct,
                    "reportDate": rep.get("date"), "timing": rep.get("timing"),
                    "verified": rep.get("verified")})
    out.sort(key=lambda r: ((r["year"] or 0), (r["quarter"] or 0)))
    return {"symbol": symbol.upper(), "quarters": out}


@mcp.tool()
async def earnings_results(symbol: str) -> dict:
    """Trailing earnings for one symbol: EPS actual vs estimate, the surprise ($ and %), report date and
    timing (BMO/AMC) - up to the last ~8 quarters from Robinhood.

    Use for EPS-surprise history on wheel / covered-call names, to gauge how a stock has handled prior
    prints before selling premium into the next one.
    """
    import asyncio
    try:
        res = await asyncio.to_thread(_rh_earnings_results_sync, symbol)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not res["quarters"]:
        return {"symbol": res["symbol"], "note": "No earnings history returned (ETF/fund or unlisted)."}
    return res


# ---- Per-symbol fundamentals (P/E, mkt cap, div yield, sector, 52wk range) ----
def _fundamentals_sync(symbols) -> dict:
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    syms = (symbols if isinstance(symbols, list)
            else [s.strip().upper() for s in str(symbols).split(",") if s.strip()])
    ff = rh.stocks.get_fundamentals(syms) or []
    out = {}
    for sym, f in zip(syms, ff):
        if not f:
            out[sym] = None
            continue
        out[sym] = {
            "peRatio": _to_float(f.get("pe_ratio")),
            "pbRatio": _to_float(f.get("pb_ratio")),
            "marketCap$": _to_float(f.get("market_cap")),
            "sharesOutstanding": _to_float(f.get("shares_outstanding")),
            "float": _to_float(f.get("float")),
            "dividendYieldPct": _to_float(f.get("dividend_yield")),
            "dividendPerShare$": _to_float(f.get("dividend_per_share")),
            "high52wk": _to_float(f.get("high_52_weeks")),
            "low52wk": _to_float(f.get("low_52_weeks")),
            "avgVolume": _to_float(f.get("average_volume")),
            "sector": f.get("sector"), "industry": f.get("industry"),
            "ceo": f.get("ceo"), "hqCity": f.get("headquarters_city"),
            "hqState": f.get("headquarters_state"),
            "description": ((f.get("description") or "")[:600] or None),
        }
    return out


@mcp.tool()
async def equity_fundamentals(symbols: str) -> dict:
    """Snapshot fundamentals for one or more symbols (comma-separated): P/E, P/B, market cap, shares,
    dividend yield, 52-week range, sector/industry, and a short profile - from Robinhood.

    For factoring the underlying business into a wheel / covered-call decision (valuation, yield, size),
    since the 0DTE tools are SPX-index-only. Max ~10 symbols per call.
    """
    import asyncio
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not syms:
        return {"error": "Provide at least one symbol."}
    try:
        f = await asyncio.to_thread(_fundamentals_sync, syms)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    missing = [s for s in syms if not f.get(s)]
    return {"fundamentals": {s: v for s, v in f.items() if v}, "notFound": (missing or None)}


# =====================================================================
# Equity technical-read engine (local, deterministic)            v0.10.0
# Public-domain indicator formulas: SMA-seeded EMA (adjust=False),
# Wilder RSI-14, MACD 12/26/9, TRIX-15/9, Bollinger 20/2 (population sigma).
# Trend + Momentum pillar scores (-2..+2) and exhaustion / bearish /
# rebound / death-cross flags. Descriptive only -- no buy/sell verdict.
# =====================================================================

def _ind_strip(vals):
    return [v for v in vals if v is not None]


def _ema_series(values, period):
    """EMA, None-padded warmup; seed = SMA of first `period` obs (adjust=False)."""
    n = len(values)
    out = [None] * n
    if n < period:
        return out
    k = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def _rsi_wilder(close, period=14):
    n = len(close)
    out = [None] * n
    if n < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, n):
        ch = close[i] - close[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def _rv(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rv(avg_gain, avg_loss)
    for i in range(period + 1, n):
        g, l = gains[i - 1], losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = _rv(avg_gain, avg_loss)
    return out


def _macd_calc(close, fast=12, slow=26, signal=9):
    ef = _ema_series(close, fast)
    es = _ema_series(close, slow)
    line = [(a - b) if (a is not None and b is not None) else None for a, b in zip(ef, es)]
    sig_valid = _ema_series(_ind_strip(line), signal)
    sig = [None] * len(close)
    first = next((i for i, v in enumerate(line) if v is not None), None)
    if first is not None:
        for off, v in enumerate(sig_valid):
            sig[first + off] = v
    hist = [(m - s) if (m is not None and s is not None) else None for m, s in zip(line, sig)]
    return line, sig, hist


def _trix_calc(close, period=15, signal=9):
    n = len(close)
    e1 = _ind_strip(_ema_series(close, period))
    e2 = _ind_strip(_ema_series(e1, period))
    e3 = _ind_strip(_ema_series(e2, period))
    trix_valid = []
    for i in range(1, len(e3)):
        prev = e3[i - 1]
        trix_valid.append((e3[i] - prev) / prev * 100.0 if prev != 0 else 0.0)
    sig_valid = _ind_strip(_ema_series(trix_valid, signal))
    t = [None] * n
    for off, v in enumerate(trix_valid):
        idx = n - len(trix_valid) + off
        if idx >= 0:
            t[idx] = v
    s = [None] * n
    for off, v in enumerate(sig_valid):
        idx = n - len(sig_valid) + off
        if idx >= 0:
            s[idx] = v
    return t, s


def _bollinger_calc(close, period=20, mult=2.0):
    if len(close) < period:
        return None, None, None, None
    window = close[-period:]
    mid = sum(window) / period
    var = sum((x - mid) ** 2 for x in window) / period   # population variance
    sd = var ** 0.5
    upper = mid + mult * sd
    lower = mid - mult * sd
    rng = upper - lower
    pct_b = (close[-1] - lower) / rng if rng != 0 else 0.5
    return mid, upper, lower, pct_b


def _ind_slope(series, lookback):
    idx = [i for i, v in enumerate(series) if v is not None]
    if len(idx) <= lookback:
        return None
    return series[idx[-1]] - series[idx[-1 - lookback]]


def _ind_r4(v):
    return round(v, 4) if isinstance(v, float) else v


def _ind_compute(close, slope_lookback=5):
    warn = None
    if len(close) < 210:
        warn = f"Only {len(close)} bars; EMA200/some indicators may be None (ideal >=220)."
    ema20 = _ema_series(close, 20)
    ema50 = _ema_series(close, 50)
    ema200 = _ema_series(close, 200)
    rsi = _rsi_wilder(close, 14)
    m_line, m_sig, m_hist = _macd_calc(close, 12, 26, 9)
    t_line, t_sig = _trix_calc(close, 15, 9)
    bb_mid, bb_up, bb_lo, pct_b = _bollinger_calc(close, 20, 2.0)

    def _last(s):
        v = _ind_strip(s)
        return v[-1] if v else None

    def _prev(s):
        v = _ind_strip(s)
        return v[-2] if len(v) >= 2 else None

    bsb = None
    for back in range(len(close)):
        i = len(close) - 1 - back
        if ema20[i] is not None and close[i] < ema20[i]:
            bsb = back
            break

    return {
        "n_bars": len(close), "warning": warn, "close": close[-1],
        "ema20": _last(ema20), "ema50": _last(ema50), "ema200": _last(ema200),
        "ema20_slope": _ind_slope(ema20, slope_lookback),
        "ema50_slope": _ind_slope(ema50, slope_lookback),
        "ema200_slope": _ind_slope(ema200, slope_lookback),
        "rsi14": _last(rsi), "rsi14_prev": _prev(rsi),
        "macd_line": _last(m_line), "macd_signal": _last(m_sig),
        "macd_hist": _last(m_hist), "macd_hist_prev": _prev(m_hist),
        "trix": _last(t_line), "trix_prev": _prev(t_line),
        "trix_signal": _last(t_sig), "trix_signal_prev": _prev(t_sig),
        "bars_since_below_ema20": bsb,
        "bb_mid": bb_mid, "bb_upper": bb_up, "bb_lower": bb_lo, "percent_b": pct_b,
    }


def _score_trend(ind):
    c = ind["close"]; e20 = ind["ema20"]; e50 = ind["ema50"]; e200 = ind["ema200"]
    s200 = ind["ema200_slope"]
    pts, bits = 0, []
    if e20 is not None:
        if c > e20: pts += 1; bits.append("price>EMA20")
        else: pts -= 1; bits.append("price<EMA20")
    if e20 is not None and e50 is not None:
        if e20 > e50: pts += 1; bits.append("EMA20>EMA50")
        else: pts -= 1; bits.append("EMA20<EMA50")
    if e50 is not None and e200 is not None:
        if e50 > e200: pts += 1; bits.append("EMA50>EMA200")
        else: pts -= 1; bits.append("EMA50<EMA200")
    if s200 is not None:
        if s200 > 0: pts += 1; bits.append("EMA200 up")
        else: pts -= 1; bits.append("EMA200 down")
    score = 2 if pts >= 3 else 1 if pts >= 1 else 0 if pts == 0 else -1 if pts >= -2 else -2
    return score, ", ".join(bits)


def _score_momentum(ind):
    rsi = ind["rsi14"]; hist = ind["macd_hist"]
    trix = ind["trix"]; trix_sig = ind["trix_signal"]
    pts, bits = 0, []
    if rsi is not None:
        if rsi >= 55: pts += 1; bits.append(f"RSI {rsi:.0f}>=55")
        elif rsi <= 45: pts -= 1; bits.append(f"RSI {rsi:.0f}<=45")
        else: bits.append(f"RSI {rsi:.0f} neutral")
    if hist is not None:
        if hist > 0: pts += 1; bits.append("MACD hist>0")
        else: pts -= 1; bits.append("MACD hist<0")
    if trix is not None and trix_sig is not None:
        if trix > trix_sig and trix > 0: pts += 1; bits.append("TRIX>signal>0")
        elif trix < trix_sig and trix < 0: pts -= 1; bits.append("TRIX<signal<0")
        else: bits.append("TRIX mixed")
    score = 2 if pts >= 2 else 1 if pts == 1 else 0 if pts == 0 else -1 if pts == -1 else -2
    return score, ", ".join(bits)


def _tech_flags(ind):
    c = ind["close"]; e20 = ind["ema20"]; e50 = ind["ema50"]; e200 = ind["ema200"]
    s200 = ind["ema200_slope"]
    rsi, rsi_p = ind["rsi14"], ind["rsi14_prev"]
    hist, hist_p = ind["macd_hist"], ind["macd_hist_prev"]
    trix, trix_sig = ind["trix"], ind["trix_signal"]
    pb = ind["percent_b"]
    stretch = (c / e20 - 1.0) if e20 else 0.0
    exhaustion, bearish, rebound = [], [], []
    if rsi is not None and rsi_p is not None and rsi >= 70 and rsi < rsi_p:
        exhaustion.append(f"RSI turning from overbought ({rsi_p:.0f}->{rsi:.0f})")
    if hist is not None and hist_p is not None and hist > 0 and hist < hist_p:
        exhaustion.append("MACD histogram shrinking in positive territory")
    if pb is not None and pb >= 1.0:
        exhaustion.append("price at/above upper Bollinger Band (%B>=1)")
    if stretch >= 0.10:
        exhaustion.append(f"price stretched {stretch*100:.0f}% above EMA20")
    if e50 and e200 and s200 is not None and c < e50 and e50 < e200 and s200 < 0:
        bearish.append("price<EMA50<EMA200 with EMA200 down")
    if hist is not None and hist_p is not None and hist < 0 and hist < hist_p:
        bearish.append("MACD histogram deepening in negative territory")
    if trix is not None and trix_sig is not None and trix < trix_sig and trix < 0:
        bearish.append("TRIX<signal below zero")
    if rsi is not None and rsi_p is not None and rsi < 45 and rsi < rsi_p:
        bearish.append(f"RSI weak and falling ({rsi:.0f})")
    if rsi is not None and rsi_p is not None and rsi_p < 35 and rsi > rsi_p:
        rebound.append(f"RSI turning from oversold ({rsi_p:.0f}->{rsi:.0f})")
    if hist is not None and hist_p is not None and hist > hist_p and hist_p < 0:
        rebound.append("MACD histogram crossing bullishly")
    bsb = ind.get("bars_since_below_ema20")
    if (e20 and c > e20 and ind["ema20_slope"] is not None and ind["ema20_slope"] > 0
            and bsb is not None and 1 <= bsb <= 5):
        rebound.append(f"price reclaims EMA20 (closed below {bsb} bar(s) ago)")
    trix_p, sig_p = ind["trix_prev"], ind["trix_signal_prev"]
    if (trix is not None and trix_sig is not None and trix_p is not None and sig_p is not None
            and trix > trix_sig and trix_p <= sig_p and trix <= 0):
        rebound.append("fresh bullish TRIX cross below zero")
    death_cross = bool(e50 and e200 and e50 < e200 and c < e50)
    return {"exhaustion": exhaustion, "bearish": bearish, "rebound": rebound,
            "death_cross": death_cross, "stretch_pct": round(stretch * 100, 1)}


def _tech_read(ind, t, m, flags):
    tr = {2: "strong uptrend", 1: "uptrend", 0: "sideways",
          -1: "downtrend", -2: "strong downtrend"}.get(t, "?")
    mo = {2: "strong+", 1: "positive", 0: "flat", -1: "negative", -2: "strong-"}.get(m, "?")
    parts = [f"Trend {t:+d} ({tr})", f"momentum {m:+d} ({mo})"]
    if flags["exhaustion"]:
        parts.append("exhaustion: " + "; ".join(flags["exhaustion"]))
    if flags["bearish"]:
        parts.append("bearish: " + "; ".join(flags["bearish"]))
    if flags["rebound"]:
        parts.append("rebound: " + "; ".join(flags["rebound"]))
    if flags["death_cross"]:
        parts.append("active death-cross (EMA50<EMA200, price<EMA50)")
    return " | ".join(parts)


def _ordered_closes(series_map, sym):
    d = (series_map or {}).get(sym) or {}
    return [c for _ts, c in sorted(d.items())]


@mcp.tool()
async def equity_technicals(symbol: str,
                            slope_lookback: Annotated[int, Field(ge=1, le=60)] = 5) -> dict:
    """Local, deterministic daily-bar technical read for one stock/ETF: EMA 20/50/200 structure,
    Wilder RSI-14, MACD 12/26/9, TRIX-15, Bollinger 20/2, plus a Trend score and a Momentum score
    (each -2..+2) and exhaustion / bearish / rebound / death-cross flags.

    Computed in-process from ~1yr of Robinhood daily closes (the same bars the rest of the desk uses)
    -- a reproducible local complement to the TradingView tools. Handy for timing a covered-call write
    into exhaustion, spotting a rolling-over underlying before a roll, or a CSP entry on a rebound.
    Descriptive only: it reports structure and momentum, it does NOT issue a buy/sell decision.
    """
    import asyncio
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"error": "symbol is required."}
    try:
        series = await asyncio.to_thread(_hist_closes_sync, [sym], "year")
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    items = sorted(((series or {}).get(sym) or {}).items())
    if len(items) < 35:
        return {"error": f"Not enough daily history for {sym} "
                         f"({len(items)} bars; need ~35+ for momentum, ~220 for full EMA200)."}
    close = [c for _ts, c in items]
    ind = _ind_compute(close, slope_lookback)
    t, t_detail = _score_trend(ind)
    m, m_detail = _score_momentum(ind)
    flags = _tech_flags(ind)
    return {
        "symbol": sym, "asOf": items[-1][0], "nBars": ind["n_bars"], "close": _ind_r4(ind["close"]),
        "trend": {"score": t, "detail": t_detail},
        "momentum": {"score": m, "detail": m_detail},
        "indicators": {
            "ema20": _ind_r4(ind["ema20"]), "ema50": _ind_r4(ind["ema50"]),
            "ema200": _ind_r4(ind["ema200"]), "ema20Slope": _ind_r4(ind["ema20_slope"]),
            "ema50Slope": _ind_r4(ind["ema50_slope"]), "ema200Slope": _ind_r4(ind["ema200_slope"]),
            "rsi14": _ind_r4(ind["rsi14"]), "macdLine": _ind_r4(ind["macd_line"]),
            "macdSignal": _ind_r4(ind["macd_signal"]), "macdHist": _ind_r4(ind["macd_hist"]),
            "trix": _ind_r4(ind["trix"]), "trixSignal": _ind_r4(ind["trix_signal"]),
            "bbMid": _ind_r4(ind["bb_mid"]), "bbUpper": _ind_r4(ind["bb_upper"]),
            "bbLower": _ind_r4(ind["bb_lower"]), "percentB": _ind_r4(ind["percent_b"]),
        },
        "flags": {"exhaustion": flags["exhaustion"], "bearish": flags["bearish"],
                  "rebound": flags["rebound"], "deathCross": flags["death_cross"],
                  "stretchVsEma20Pct": flags["stretch_pct"]},
        "read": _tech_read(ind, t, m, flags),
        "warning": ind["warning"],
        "note": "Daily-bar technicals from Robinhood historicals (SMA-seeded EMA, Wilder RSI-14, "
                "MACD 12/26/9, TRIX-15, Bollinger 20/2 population sigma). Descriptive, not a buy/sell signal.",
    }


# =====================================================================
# Cross-asset market-internals (ETF-ratio regime)               v0.10.0
# Ported from macro_pillar.py: weighted composite of RSP/SPY, HYG/LQD,
# IWM/SPY, SPY/TLT, XLY/XLP trend signals + 10Y-2Y curve + SPY-TLT corr.
# Complements regime_classifier (VIX / credit / curve macro-series read).
# =====================================================================
_MI_WEIGHTS = {
    "concentration": ("RSP/SPY", 0.25), "yield_curve": ("10Y-2Y", 0.20),
    "credit": ("HYG/LQD", 0.15), "size": ("IWM/SPY", 0.15),
    "equity_bond": ("SPY/TLT", 0.15), "sector": ("XLY/XLP", 0.10),
}
_MI_ETFS = ["SPY", "RSP", "IWM", "HYG", "LQD", "TLT", "XLY", "XLP"]


def _mi_sma(series, window):
    if len(series) < window:
        return None
    return sum(series[-window:]) / window


def _mi_ratio(num, den):
    # B15: when given {date: close} dicts, align on the INTERSECTION of dates (sorted) so we never
    # divide one series' Friday close by the other's Thursday close when their histories differ.
    # Falls back to the legacy positional tail-align for plain lists.
    if isinstance(num, dict) and isinstance(den, dict):
        common = sorted(set(num) & set(den))
        return [num[d] / den[d] for d in common if den[d] != 0]
    n = min(len(num), len(den))
    num, den = num[-n:], den[-n:]
    return [a / b for a, b in zip(num, den) if b != 0]


def _mi_pct_returns(series):
    out = []
    for i in range(1, len(series)):
        if series[i - 1] != 0:
            out.append(series[i] / series[i - 1] - 1.0)
    return out


def _mi_pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 5:
        return None
    xs, ys = xs[-n:], ys[-n:]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def _mi_trend_signal(series, fast, slow, slope_win):
    s_slow = _mi_sma(series, slow)
    if s_slow is None or len(series) < slow + slope_win:
        return None, "insufficient data"
    base = 1.0 if series[-1] > s_slow else -1.0
    slow_then = _mi_sma(series[:-slope_win], slow)
    if slow_then is None:
        return None, "insufficient data for slope"
    trend = 1.0 if s_slow > slow_then else -1.0
    sig = 0.5 * base + 0.5 * trend
    pos = "above" if base > 0 else "below"
    slp = "rising" if trend > 0 else "falling"
    return sig, f"ratio {pos} SMA{slow}, SMA{slow} {slp}"


def _mi_component(closes_by, sym_num, sym_den, key, fast, slow, slope_win):
    ratio_lbl, weight = _MI_WEIGHTS[key]
    comp = {"name": key, "ratio": ratio_lbl, "weight": weight,
            "signal": None, "detail": "", "available": True}
    num = closes_by.get(sym_num); den = closes_by.get(sym_den)
    if num and den:
        sig, detail = _mi_trend_signal(_mi_ratio(num, den), fast, slow, slope_win)
        comp["signal"], comp["detail"] = sig, detail
    if comp["signal"] is None:
        comp["available"] = False
    return comp


def _cross_asset_macro(closes_by, yield_spread=None, spread_source="none",
                       fast=50, slow=200, slope_win=20, corr_win=40):
    notes = []
    comps = {}
    comps["concentration"] = _mi_component(closes_by, "RSP", "SPY", "concentration", fast, slow, slope_win)

    cyc = {"name": "yield_curve", "ratio": "10Y-2Y", "weight": _MI_WEIGHTS["yield_curve"][1],
           "signal": None, "detail": "", "available": True}
    spread = yield_spread
    if spread is not None and not isinstance(spread, list):
        spread = [spread]
    if spread and len(spread) >= slope_win + 1:
        spread = [float(x) for x in spread]
        now, then = spread[-1], spread[-1 - slope_win]
        base = 1.0 if now > 0 else -1.0
        trend = 1.0 if now > then else -1.0
        cyc["signal"] = 0.5 * base + 0.5 * trend
        cyc["detail"] = f"spread {now:+.2f}, {'steepening' if trend > 0 else 'flattening'} [{spread_source}]"
    elif spread:
        now = float(spread[-1])
        cyc["signal"] = 0.5 if now > 0 else -0.5
        cyc["detail"] = f"spread {now:+.2f} (level only) [{spread_source}]"
        notes.append("yield_spread <21 obs: level only (+/-0.5), no slope.")
    else:
        cyc["available"] = False
        notes.append("No yield_spread: 20% curve weight redistributed across other components.")
    comps["yield_curve"] = cyc

    comps["credit"] = _mi_component(closes_by, "HYG", "LQD", "credit", fast, slow, slope_win)
    comps["size"] = _mi_component(closes_by, "IWM", "SPY", "size", fast, slow, slope_win)
    comps["equity_bond"] = _mi_component(closes_by, "SPY", "TLT", "equity_bond", fast, slow, slope_win)
    comps["sector"] = _mi_component(closes_by, "XLY", "XLP", "sector", fast, slow, slope_win)

    spy = closes_by.get("SPY"); tlt = closes_by.get("TLT")
    spy_tlt_corr = None
    if spy and tlt:
        if isinstance(spy, dict) and isinstance(tlt, dict):
            common = sorted(set(spy) & set(tlt))[-(corr_win + 1):]   # B15: date-aligned returns
            spy_ser = [spy[d] for d in common]
            tlt_ser = [tlt[d] for d in common]
        else:
            spy_ser = spy[-(corr_win + 1):]
            tlt_ser = tlt[-(corr_win + 1):]
        rs = _mi_pct_returns(spy_ser)
        rt = _mi_pct_returns(tlt_ser)
        spy_tlt_corr = _mi_pearson(rs, rt)

    avail = [c for c in comps.values() if c["available"] and c["signal"] is not None]
    if not avail:
        raise EdgeError("No cross-asset components with sufficient data.")
    wsum = sum(c["weight"] for c in avail)
    composite = sum(c["signal"] * c["weight"] for c in avail) / wsum
    composite = max(-1.0, min(1.0, composite))

    eb = comps["equity_bond"]
    inflationary = bool(spy_tlt_corr is not None and spy_tlt_corr > 0.25
                        and eb["available"] and eb["signal"] is not None and eb["signal"] <= 0)

    rsp_sig = comps["concentration"]["signal"] or 0
    iwm_sig = comps["size"]["signal"] or 0
    cr_sig = comps["credit"]["signal"] or 0
    if inflationary:
        regime = "Inflationary"
    elif composite <= -0.5 and cr_sig < 0:
        regime = "Contraction"
    elif composite >= 0.4 and iwm_sig > 0:
        regime = "Broadening"
    elif rsp_sig < 0 and iwm_sig < 0 and composite > -0.5:
        regime = "Concentration"
    else:
        regime = "Transitional"

    if composite >= 0.5:
        pillar, plabel = 2, "Strongly favorable macro"
    elif composite >= 0.2:
        pillar, plabel = 1, "Favorable macro"
    elif composite > -0.2:
        pillar, plabel = 0, "Neutral macro"
    elif composite > -0.5:
        pillar, plabel = -1, "Adverse macro"
    else:
        pillar, plabel = -2, "Strongly adverse macro"
    if regime in ("Contraction", "Inflationary") and pillar > -1:
        pillar = -1
        plabel = f"Adverse macro (cap due to {regime} regime)"
        notes.append(f"Pillar capped at -1 due to {regime} regime.")

    return {"composite": round(composite, 3), "regime": regime, "pillarScore": pillar,
            "pillarLabel": plabel, "inflationaryFlag": inflationary,
            "spyTltCorr": round(spy_tlt_corr, 3) if spy_tlt_corr is not None else None,
            "components": [{"ratio": c["ratio"], "weight": c["weight"],
                            "signal": round(c["signal"], 2) if c["signal"] is not None else None,
                            "detail": c["detail"], "available": c["available"]}
                           for c in comps.values()],
            "notes": notes}


def _mi_yield_spread_sync():
    """(spread, source, latest): prefer FRED T10Y2Y series (slope-capable), then DGS10-DGS2 level."""
    try:
        rows = _fetch_series("T10Y2Y", start=_recent_window(180))
        vals = [v for _d, v in rows]
        if len(vals) >= 21:
            return vals, "FRED T10Y2Y", vals[-1]
        if vals:
            return vals[-1], "FRED T10Y2Y (level)", vals[-1]
    except Exception:  # noqa: BLE001
        pass
    try:
        ten = _latest_obs("DGS10"); two = _latest_obs("DGS2")
        if ten and two:
            sp = round(float(ten[1]) - float(two[1]), 3)
            return sp, "FRED DGS10-DGS2 (level)", sp
    except Exception:  # noqa: BLE001
        pass
    return None, "unavailable", None


@mcp.tool()
async def market_internals(yield_spread: Optional[float] = None) -> dict:
    """Cross-asset market-internals regime from daily ETF-ratio trends: concentration (RSP/SPY),
    credit (HYG/LQD), size (IWM/SPY), equity-vs-bond (SPY/TLT), sector rotation (XLY/XLP), plus the
    10Y-2Y curve and the rolling SPY-TLT return correlation. Returns a weighted composite (-1..+1), a
    Macro-Sentiment pillar (-2..+2), a regime label (Broadening / Concentration / Contraction /
    Inflationary / Transitional), and an inflationary flag.

    A price-based internals lens that complements `regime_classifier` (which reads VIX term-structure
    and macro series). ETF closes come from ~1yr of Robinhood daily historicals. The 2s10s spread is
    auto-filled from FRED (T10Y2Y) when `yield_spread` is omitted; if neither is available its 20%
    weight is redistributed across the other components.
    """
    import asyncio
    try:
        closes_by = await asyncio.to_thread(_hist_closes_sync, _MI_ETFS, "year")
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if yield_spread is not None:
        spread, spread_source, spread_latest = yield_spread, "caller", yield_spread
    else:
        try:
            spread, spread_source, spread_latest = await asyncio.to_thread(_mi_yield_spread_sync)
        except Exception:  # noqa: BLE001
            spread, spread_source, spread_latest = None, "unavailable", None
    # B15: pass the raw {date: close} dicts (NOT positional lists) so the ratio and SPY-TLT
    # correlation align on shared dates. _ordered_closes dropped the dates, which let a series with
    # a missing/extra day pair mismatched closes and skew every internals signal.
    closes_map = {s: ((closes_by or {}).get(s) or {}) for s in _MI_ETFS}
    try:
        res = await asyncio.to_thread(_cross_asset_macro, closes_map, spread, spread_source)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    res["asOf"] = _today_et().isoformat()
    res["yieldSpread"] = spread_latest
    res["yieldSpreadSource"] = spread_source
    res["note"] = ("Cross-asset internals from daily ETF-ratio trends (RSP/SPY, HYG/LQD, IWM/SPY, "
                   "SPY/TLT, XLY/XLP) + 2s10s + SPY-TLT corr. Complements regime_classifier; ETF "
                   "closes from Robinhood historicals. Not investment advice.")
    return res


if __name__ == "__main__":
    main()
