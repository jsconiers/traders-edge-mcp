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

Data sources (no API key required):
  * CBOE delayed quotes:  https://cdn.cboe.com/api/global/delayed_quotes/
  * TreasuryDirect:       https://www.treasurydirect.gov/TA_WS/securities/upcoming
Optional: set FMP_API_KEY or FINNHUB_API_KEY for a fully live economic calendar.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import csv
import io
import sys
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
CHAIN_TTL = 90.0          # seconds to cache the (large) CBOE chain pull
QUOTE_TTL = 60.0
REQUEST_TIMEOUT = 90.0
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


_cache: dict = {}


async def _async_sleep(secs: float) -> None:
    import asyncio
    await asyncio.sleep(secs)


async def _get_json(url: str, ttl: float) -> Any:
    hit = _cache.get(url)
    if hit and (time.monotonic() - hit[1]) < ttl:
        return hit[0]
    last = None
    for _ in range(3):
        try:
            r = await _get_client().get(url)
            break
        except Exception as exc:  # noqa: BLE001 (transient disconnects; retry)
            last = exc
            await _async_sleep(1.2)
    else:
        raise EdgeError(f"Request failed: {url} ({last})")
    if r.status_code == 404:
        raise EdgeError(f"Not found (404): {url}")
    try:
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        raise EdgeError(f"Bad response from {url}: {str(exc)[:160]}") from exc
    _cache[url] = (data, time.monotonic())
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
    """Years from 'now' (ET) to 16:00 ET on the expiration date; floored so 0DTE gamma stays finite."""
    try:
        ed = _dt.date.fromisoformat(expiry)
    except ValueError:
        return 1.0 / 365.0
    expiry_dt = _dt.datetime.combine(ed, _dt.time(16, 0), tzinfo=ET)
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
    raw = await _get_json(CBOE_OPTIONS_URL.format(root=SPX_FILE_ROOT), CHAIN_TTL)
    d = raw.get("data", {}) or {}
    seq = d.get("seqno") or d.get("last_trade_time")
    pc = _cache.get("__parsed__")
    if pc and pc[0] == seq and (time.monotonic() - pc[2]) < CHAIN_TTL:
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


@mcp.tool()
async def traders_edge_status() -> dict:
    """Health check: confirms the CBOE options + vol feeds are reachable and reports current spot."""
    out: dict = {"sources": {"options": "CBOE delayed JSON", "vol": "CBOE indices",
                             "events": "TreasuryDirect + curated"}, "note": "Data ~15 min delayed."}
    try:
        ch = await _load_chain()
        out["spot"] = round(ch["spot"], 2)
        out["contracts"] = len(ch["options"])
        out["asof"] = ch.get("asof")
        out["reachable"] = True
    except EdgeError as exc:
        out["reachable"] = False
        out["error"] = str(exc)
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
    return {"root": rootU, "spot": round(ch["spot"], 2), "count": len(rows), "expirations": rows}


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
        ch = await _load_chain()
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
    try:
        ch = await _load_chain()
    except EdgeError as exc:
        return {"error": str(exc)}
    o = next((x for x in ch["options"] if (x["symbol"] or "").upper() == sym), None)
    if not o:
        return {"error": f"{sym} not found in the current SPX chain."}
    root, expiry, cp, strike = parsed
    spot = ch["spot"]
    T = _year_frac(expiry)
    detail = {"symbol": sym, "root": root, "type": cp, "strike": strike, "expiration": expiry,
              "dte": (_dt.date.fromisoformat(expiry) - _today_et()).days,
              "spot": round(spot, 2), "bid": o["bid"], "ask": o["ask"], "mid": round(o["mid"], 2),
              "last": o["last"], "iv": round(o["iv"], 4) if o["iv"] else None,
              "openInterest": int(o["oi"]), "volume": int(o["volume"])}
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
    try:
        ch = await _load_chain()
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
        ch = await _load_chain()
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if zero_dte else (expiration or _nearest_expiry(ch["options"], root))
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
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
    resolved = "today" if zero_dte else exp
    return {
        "root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
        "callWall": ({"strike": call_wall[0], "gamma_$mm": _mm(call_wall[1])} if call_wall else None),
        "putWall": ({"strike": put_wall[0], "gamma_$mm": _mm(put_wall[1])} if put_wall else None),
        "maxPain": _max_pain(opts),
        "topGammaStrikes": [{"strike": k, "netGamma_$mm": _mm(v)} for k, v in top_abs],
    }


@mcp.tool()
async def zero_dte_exposure(root: str = "SPXW") -> dict:
    """One-shot 0DTE dashboard: GEX, zero-gamma flip, call/put walls, max-pain pin, and expected move.

    Filters to today's SPXW expiration (falls back to the nearest expiry with a note if nothing expires
    today). The pin (max-pain) and the gamma walls are where price tends to get magnetized into the
    close on a positive-gamma day; the expected move is the ATM straddle (~1-sigma for the session).
    """
    try:
        ch = await _load_chain()
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
        ch = await _load_chain()
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    exp = None if (zero_dte or not expiration) else expiration
    opts = _filter(ch["options"], root=root, expiration=exp, zero_dte=zero_dte)
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
    scope = "today" if zero_dte else (exp or "all expirations")
    return {
        "root": root.upper(), "scope": scope, "spot": round(spot, 2), "asof": ch.get("asof"),
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

# Best-effort scheduled US macro for 2026 (8:30 ET unless noted). Verify exact dates for tick-precision;
# set FMP_API_KEY or FINNHUB_API_KEY for a fully live calendar.
_CURATED_2026 = [
    ("2026-06-25", "08:30", "GDP (Q1 final)", "medium"),
    ("2026-06-26", "08:30", "PCE Price Index (May)", "high"),
    ("2026-07-15", "08:30", "CPI (June)", "high"),
    ("2026-07-16", "08:30", "PPI (June)", "medium"),
    ("2026-07-16", "08:30", "Retail Sales (June)", "high"),
    ("2026-07-31", "08:30", "PCE Price Index (June)", "high"),
    ("2026-08-12", "08:30", "CPI (July)", "high"),
    ("2026-08-14", "08:30", "Retail Sales (July)", "high"),
    ("2026-09-11", "08:30", "CPI (August)", "high"),
    ("2026-09-16", "08:30", "Retail Sales (August)", "high"),
    ("2026-10-14", "08:30", "CPI (September)", "high"),
    ("2026-11-13", "08:30", "CPI (October)", "high"),
    ("2026-12-10", "08:30", "CPI (November)", "high"),
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
    return None


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
    if live is not None:
        events, src = live, "live API"
    else:
        events, src = _build_events(start, end), "curated + rules (no API key set)"
    events = events + await _treasury_auctions(start, end)
    imp = importance.lower()
    if imp in ("high", "medium"):
        wanted = {"high"} if imp == "high" else {"high", "medium"}
        events = [e for e in events if e["importance"] in wanted]
    events.sort(key=lambda e: (e["date"], e.get("time", "")))
    out = {"window": {"from": start.isoformat(), "to": end.isoformat()},
           "source": src, "count": len(events), "events": events}
    if live is None:
        out["note"] = "Set FMP_API_KEY or FINNHUB_API_KEY for tick-precise CPI/PCE/PPI dates."
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
    return {"event": e["name"], "date": e["date"], "timeET": e.get("time"),
            "importance": e["importance"], "source": e["source"],
            "countdownHours": round(hrs, 1), "countdown": f"{int(hrs // 24)}d {int(hrs % 24)}h"}


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
_fred_cache: dict = {}


def _fred_get_client() -> httpx.Client:
    global _fred_client
    if _fred_client is None:
        _fred_client = httpx.Client(timeout=FRED_TIMEOUT, follow_redirects=True,
                               headers={"User-Agent": FRED_UA})
    return _fred_client


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
    _fred_cache[key] = (rows, time.monotonic())
    return rows


def _latest_obs(series_id: str) -> Optional[tuple[str, float]]:
    # Bounded windows keep downloads small (full history of daily series is large/slow).
    for win in (450, 1800):
        try:
            rows = _fetch_series(series_id, start=_recent_window(win))
        except FredError:
            return None
        if rows:
            return rows[-1]
    return None


def _recent_window(days: int) -> str:
    return (_dt.date.today() - _dt.timedelta(days=days)).isoformat()


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


async def _yahoo_price(symbol: str) -> Optional[float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r = await _get_client().get(url, params={"interval": "1d", "range": "1d"})
        meta = r.json()["chart"]["result"][0]["meta"]
        px = meta.get("regularMarketPrice") or meta.get("previousClose")
        return float(px) if px is not None else None
    except Exception:  # noqa: BLE001
        return None


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
        d = float(p["delta"]); gg = float(p.get("gamma") or 0.0)
        th = float(p.get("theta") or 0.0); vg = float(p.get("vega") or 0.0)
        beta = _beta_of(p)
        dd = d * qty * mult * undpx
        return {"broker": p["broker"], "symbol": p["symbol"], "underlying": (p.get("underlying") or root),
                "type": "option", "qty": qty, "strike": strike, "expiry": expiry, "delta$": dd,
                "gamma$_1pct": gg * qty * mult * undpx * undpx * 0.01,
                "theta$_day": th * qty * mult, "vega$_1pct": vg * qty * mult,
                "betaDelta$": dd * beta, "mv": p.get("mv"), "greeksSource": "broker"}
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
    for pos in opos:
        try:
            q = _f(pos.get("quantity")) or 0.0
            if q == 0:
                continue
            qty = q if (pos.get("type") or "").lower() != "short" else -q
            chain = pos.get("chain_symbol", "")
            opt_url = pos.get("option", "")
            od = {}
            if opt_url:
                try:
                    od = rh.helper.request_get(opt_url) or {}
                except Exception:  # noqa: BLE001
                    od = {}
            oid = od.get("id") or (opt_url.rstrip("/").split("/")[-1] if opt_url else None)
            strike = _f(od.get("strike_price"))
            expiry = od.get("expiration_date")
            cp = {"call": "C", "put": "P"}.get(od.get("type"))
            md = {}
            if oid:
                try:
                    raw = rh.get_option_market_data_by_id(oid)
                    md = raw[0] if isinstance(raw, list) and raw else (raw or {})
                except Exception:  # noqa: BLE001
                    md = {}
            undpx = None
            if chain:
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


async def _collect_positions(include_alpaca: bool, include_file: bool, include_robinhood: bool = True, include_etrade: bool = True) -> tuple:
    positions, meta = [], {}
    if include_alpaca:
        try:
            rows = await _alpaca_get("/v2/positions")
            positions += _normalize_alpaca(rows)
            meta["alpaca"] = len(rows)
        except EdgeError as exc:
            meta["alpacaError"] = str(exc)
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
    if include_file:
        try:
            fpos, path = _read_positions_file()
            positions += fpos
            meta["file"], meta["filePath"] = len(fpos), path
        except EdgeError as exc:
            meta["fileError"] = str(exc)
    return positions, meta


async def _price_map(positions: list, spot: float) -> dict:
    need = set()
    for p in positions:
        if p["type"] == "equity" and not p.get("price"):
            need.add(p["underlying"])
        elif p["type"] == "option":
            root = (_parse_occ(p["symbol"]) or [""])[0] or (p.get("underlying") or "")
            if root not in ("SPX", "SPXW"):
                need.add(p.get("underlying") or root)
    need = [s for s in need if s]
    if not need:
        return {}
    import asyncio
    vals = await asyncio.gather(*[_yahoo_price(s) for s in need])
    return {s: v for s, v in zip(need, vals)}


async def _aggregate(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True) -> dict:
    positions, meta = await _collect_positions(include_alpaca, include_file, include_robinhood, include_etrade)
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
    return {"positions": risks, "spot": spot, "meta": meta}


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
    out["etrade"] = {"libInstalled": et_lib, "envExists": os.path.exists(et_env),
                     "tokenPickleExists": os.path.exists(et_tok)}
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
async def net_greeks(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True) -> dict:
    """Net portfolio Greeks aggregated across Alpaca and the positions file.

    Sums dollar delta, dollar gamma (per 1% move), theta (per day), and vega (per 1% vol). SPX/SPXW
    options get full Black-Scholes Greeks off CBOE; equities contribute delta only (beta-weighted);
    other instruments use whatever the positions file provides. Delta is also expressed in SPX points.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade)
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
        "positions": len(risks),
        "netDelta$": round(netd, 0),
        "netDelta_betaWeighted$": round(netbd, 0),
        "netDelta_SPXpoints": (round(netbd / spot, 1) if spot else None),
        "netGamma$_per_1pct": round(netg, 0),
        "netTheta$_per_day": round(nett, 0),
        "netVega$_per_1pct_vol": round(netv, 0),
        "greeksCoverage": cov,
        "sources": agg["meta"],
    }
    return out


@mcp.tool()
async def risk_summary(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True) -> dict:
    """Portfolio risk overview: beta-weighted SPX exposure, gross/long/short notional, and breakdowns.

    Groups exposure by broker and by underlying, and lists the largest directional contributors.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade)
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
async def concentration(include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True) -> dict:
    """Exposure concentration by underlying, flagging any name above the concentration threshold.

    Threshold defaults to 25% of gross exposure (override with CONCENTRATION_PCT).
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade)
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
    include_alpaca: bool = True, include_file: bool = True, include_robinhood: bool = True, include_etrade: bool = True,
) -> dict:
    """Estimate portfolio P&L under a set of SPX % moves using net beta-delta + net gamma convexity.

    `moves_pct` is a comma-separated list (e.g. '-2,-1,-0.5,0.5,1,2'); defaults to that set. SPX/SPXW
    positions use delta + gamma; everything else is beta-weighted linear. A quick risk read, not a
    full revaluation.
    """
    agg = await _aggregate(include_alpaca, include_file, include_robinhood, include_etrade)
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
        try:
            acct = await _alpaca_get("/v2/account")
            eq = float(acct.get("equity") or 0.0)
            last = float(acct.get("last_equity") or 0.0)
            rpl = eq - last
            src = "alpaca (equity - last_equity)"
        except EdgeError as exc:
            return {"error": f"No realized_pl provided and Alpaca unavailable: {exc}"}
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


def _rh_recent_option_orders(stop_date: _dt.date, max_pages: int = 6) -> list:
    """Filled+other option orders, newest-first, paginating only until we pass stop_date (ET)."""
    import robin_stocks.robinhood as rh
    _rh_login_sync()
    try:
        url = rh.urls.option_orders()
    except Exception:  # noqa: BLE001
        url = "https://api.robinhood.com/options/orders/"
    out, page = [], 0
    data = rh.helper.request_get(url, "regular")
    while data and isinstance(data, dict):
        results = data.get("results", []) or []
        out.extend(results)
        page += 1
        oldest_d = None
        if results:
            try:
                oldest_d = _dt.datetime.fromisoformat(
                    results[-1].get("created_at", "").replace("Z", "+00:00")).astimezone(ET).date()
            except Exception:  # noqa: BLE001
                oldest_d = None
        nxt = data.get("next")
        if not nxt or page >= max_pages or (oldest_d and oldest_d < stop_date):
            break
        data = rh.helper.request_get(nxt, "regular")
    return out


def _order_to_fill(o: dict):
    if o.get("state") != "filled":
        return None
    legs = o.get("legs", []) or []
    net = _to_float(o.get("net_amount")) or 0.0
    direction = o.get("net_amount_direction") or o.get("direction")
    net_cf = net if direction == "credit" else -net
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
        return None
    trade_date = trade_date or when.date().isoformat()
    rec = {"time": when, "trade_date": trade_date, "chain": o.get("chain_symbol", ""),
           "n_legs": len(legs), "net_cf": net_cf,
           "gross_premium": _to_float(o.get("processed_premium")),
           "qty": _to_float(o.get("processed_quantity")) or _to_float(o.get("quantity")) or 0.0}
    if len(legs) == 1:
        lg = legs[0]
        ex0 = (lg.get("executions") or [{}])
        rec.update({"option_id": (lg.get("option") or "").rstrip("/").split("/")[-1],
                    "side": lg.get("side"), "effect": lg.get("position_effect"),
                    "strike": _to_float(lg.get("strike_price")),
                    "cp": {"call": "C", "put": "P"}.get(lg.get("option_type")),
                    "expiry": lg.get("expiration_date"),
                    "price": _to_float(ex0[0].get("price")) if ex0 else None})
    else:
        rec.update({"option_id": "multi:" + (o.get("id") or ""), "effect": None,
                    "strike": None, "cp": None, "expiry": None})
    return rec


def _day_fills_sync(date_iso: str) -> list:
    target = _dt.date.fromisoformat(date_iso)
    fills = []
    for o in _rh_recent_option_orders(target):
        f = _order_to_fill(o)
        if f and f["trade_date"] == date_iso:
            fills.append(f)
    fills.sort(key=lambda r: r["time"])
    return fills


async def _day_fills(date_iso: str) -> list:
    import asyncio
    return await asyncio.to_thread(_day_fills_sync, date_iso)


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


def _round_trips(fills: list) -> list:
    from collections import defaultdict, deque
    lots = defaultdict(deque)
    trips = []
    for f in fills:
        oid, eff, qty, net = f.get("option_id"), f.get("effect"), (f.get("qty") or 0.0), f["net_cf"]
        if eff == "open" and qty > 0:
            lots[oid].append([qty, net, f["time"]])
        elif eff == "close" and qty > 0:
            remaining = qty
            close_per = net / qty if qty else 0.0
            while remaining > 1e-9 and lots[oid]:
                lot = lots[oid][0]
                lot_qty, lot_cost, lot_time = lot
                take = min(remaining, lot_qty)
                open_per = lot_cost / lot_qty if lot_qty else 0.0
                trips.append({"open": lot_time, "close": f["time"], "chain": f.get("chain"),
                              "strike": f.get("strike"), "cp": f.get("cp"), "qty": take,
                              "pnl": take * (open_per + close_per),
                              "holdSec": (f["time"] - lot_time).total_seconds(),
                              "expiry": f.get("expiry"), "option_id": oid})
                lot_qty -= take
                lot[0], lot[1] = lot_qty, lot_cost - open_per * take
                remaining -= take
                if lot_qty <= 1e-9:
                    lots[oid].popleft()
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
        fills = await _day_fills(d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if not fills:
        return {"date": d, "note": "No filled option orders for this date.", "realized$": 0.0,
                "orders": 0}
    trips = _round_trips(fills)
    cash = round(sum(f["net_cf"] for f in fills), 2)
    if not trips:
        return {"date": d, "orders": len(fills), "netCashFlow$": cash,
                "note": "No completed round trips - positions may still be open or expired by assignment."}
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
        out["reconNote"] = "Net cash flow differs from round-trip realized; some positions expired or remain open."
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
        fills = await _day_fills(d)
    except EdgeError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    trips = _round_trips(fills)
    if not trips:
        return {"date": d, "note": "No completed round trips for this date.",
                "orders": len(fills)}
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
    return out


@mcp.tool()
async def should_i_trade(date: Optional[str] = None, target: Optional[float] = None) -> dict:
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
    trips = _round_trips(fills)
    cur = _build_curve(trips, tgt)
    total = cur["total"]
    peak = cur["peak"]
    reasons, flags = [], []

    past_target = total >= tgt and tgt > 0
    giveback = peak >= tgt and (peak - total) >= giveback_frac * tgt
    last3 = trips[-3:]
    consec_losses = len(last3) >= 3 and all(t["pnl"] < 0 for t in last3)
    consec2 = len(trips) >= 2 and all(t["pnl"] < 0 for t in trips[-2:])
    deep_dd = total <= -0.5 * tgt
    # rapid re-entry: tight gaps between last few opens
    opens = [f["time"] for f in fills if f.get("effect") == "open"]
    rapid = False
    if len(opens) >= 3:
        gaps = [(opens[i] - opens[i - 1]).total_seconds() for i in range(-2, 0)]
        rapid = all(g < rapid_secs for g in gaps)
    lh, lm = (int(x) for x in late_et.split(":"))
    late = is_today and (now.hour, now.minute) >= (lh, lm) and now.hour < 16

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

    if past_target or consec_losses or giveback:
        verdict = "STOP"
    elif late or rapid or deep_dd or consec2:
        verdict = "CAUTION"
    else:
        verdict = "GO"
    if not reasons:
        reasons.append("No discipline flags: within target, no tilt signals, normal pacing.")
    return {"date": d, "verdict": verdict, "flags": flags, "reasons": reasons,
            "realized$": round(total, 2), "peak$": round(peak, 2), "target$": round(tgt, 2),
            "roundTrips": len(trips), "asof": now.strftime("%H:%M:%S ET") if is_today else "EOD review"}


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
        ch = await _load_chain()
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
    em = _em_levels(spot, opts, None if resolved == "today" else resolved)
    return {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
            "asof": ch.get("asof"), **em}


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
        ch = await _load_chain()
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
        else:
            row["probTouch%"] = 100.0
        for side in ("C", "P"):
            o = by[k].get(side)
            if not o:
                continue
            iv = o["iv"]
            d2 = (np.log(spot / k) + (RISK_FREE - 0.5 * iv * iv) * T) / (iv * sqrtT)
            p_itm = float(_norm_cdf(np.array([d2 if side == "C" else -d2]))[0])
            row[side] = {"probITM%": round(100.0 * p_itm, 1),
                         "delta": round(o.get("delta", 0.0), 3), "iv": round(iv, 4)}
        rows.append(row)
    return {"root": root.upper(), "expiration": resolved, "spot": round(spot, 2),
            "asof": ch.get("asof"), "strikes": rows}


@mcp.tool()
async def daily_game_plan(root: str = "SPXW") -> dict:
    """One call for today's 0DTE map: spot, expected-move bands, gamma regime + flip, call/put walls,
    high-OI pins, and max-pain - assembled into support/resistance you can trade against.

    Resistance = call wall / +sigma / high call OI; support = put wall / -sigma / high put OI; pivots =
    max-pain, gamma flip, spot. SPX pins toward max-pain and gamma walls into the close on long-gamma days.
    """
    try:
        ch = await _load_chain()
    except EdgeError as exc:
        return {"error": str(exc)}
    spot = ch["spot"]
    opts = _filter(ch["options"], root=root, zero_dte=True)
    resolved, note = "today", None
    if not opts:
        exp = _nearest_expiry(ch["options"], root)
        opts = _filter(ch["options"], root=root, expiration=exp)
        resolved, note = exp, f"Nothing expires today; using nearest expiry {exp}."
    comp = _gex_components(spot, opts)
    if not comp:
        return {"error": "No valid contracts (need OI and IV)."}
    flip = _gamma_flip(spot, comp)
    call, put, net = _per_strike(comp)
    cw = max(({k: v for k, v in call.items() if k >= spot}).items(), key=lambda kv: kv[1], default=None)
    pw = max(({k: v for k, v in put.items() if k <= spot}).items(), key=lambda kv: kv[1], default=None)
    mp = _max_pain(opts)
    em = _em_levels(spot, opts, None if resolved == "today" else resolved)
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
    return {"shortCalls": len(shorts), "totalPremiumCaptured$": round(tot_prem, 2),
            "rollThresholds": {"delta": roll_delta, "dte": roll_dte}, "positions": rows}


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
        orders = await asyncio.to_thread(_rh_recent_option_orders, stop, 30)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    fills_by_day = defaultdict(list)
    for o in orders:
        f = _order_to_fill(o)
        if f and f["trade_date"] >= stop_iso:
            fills_by_day[f["trade_date"]].append(f)
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
    return {
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
        orders = await asyncio.to_thread(_rh_recent_option_orders, jan1, 40)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    fills = []
    for o in orders:
        f = _order_to_fill(o)
        if f:
            fills.append(f)
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
    return {
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


@mcp.tool()
async def snapshot_log() -> dict:
    """Capture the current 0DTE state (spot, total GEX, gamma flip, call/put walls, max-pain, expected
    move, VIX/VIX1D, regime) to a local SQLite log for intraday trend tracking.

    Call periodically through the session; `snapshot_history` then shows how dealer gamma and the key
    levels drifted (GEX migration). Stored at ~/.trading/traders_edge.db.
    """
    import asyncio
    z, vx, rg = await asyncio.gather(zero_dte_exposure(), vix_complex(), regime_classifier())
    if isinstance(z, dict) and "error" in z:
        return {"error": f"chain unavailable: {z['error']}"}
    em = z.get("expectedMove") or {}
    idx = (vx.get("indices") or {}) if isinstance(vx, dict) else {}
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
    return {"logged": True, "db": TE_DB_PATH,
            "snapshot": {k: row[k] for k in ("ts", "spot", "total_gex", "gamma_flip", "call_wall",
                                             "put_wall", "max_pain", "expected_move", "vix", "vix1d",
                                             "regime", "regime_score")}}


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
    results = []
    for sym, k, e in jobs[:5]:
        try:
            results.append(await asyncio.to_thread(_roll_scan_sync, sym, k, e, min_dte, max_dte,
                                                   target_delta, 4))
        except Exception as exc:  # noqa: BLE001
            results.append({"underlying": sym, "error": str(exc)[:140]})
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
    orders = await asyncio.to_thread(_rh_recent_option_orders, _today_et() - _dt.timedelta(days=7), 8)
    by = defaultdict(list)
    for o in orders:
        f = _order_to_fill(o)
        if f:
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
    dr, sit, snap = await asyncio.gather(daily_review(), should_i_trade(), snapshot_log(),
                                         return_exceptions=True)
    dr, sit, snap = _safe_dict(dr), _safe_dict(sit), _safe_dict(snap)
    out = {"date": _today_et().isoformat()}
    if "pnl$" in dr:
        out["result"] = {"realized$": dr.get("pnl$"), "trips": dr.get("trades"),
                         "winRate%": dr.get("winRate%"), "targetHitAt": dr.get("targetHitAt"),
                         "beforeTarget": dr.get("beforeTarget"), "afterTarget": dr.get("afterTarget")}
    else:
        out["result"] = {"note": dr.get("note") or dr.get("error") or "no fills today"}
    tgt = _target()
    realized = dr.get("pnl$")
    after = (dr.get("afterTarget") or {})
    notes = []
    if dr.get("targetHitAt"):
        ap = after.get("pnl$")
        if ap is not None and ap < 0:
            notes.append(f"Hit target at {dr['targetHitAt']}, then gave back ${ap:.2f} after - the leak pattern.")
        elif ap is not None and ap > 0:
            notes.append(f"Hit target at {dr['targetHitAt']} and added ${ap:.2f} after.")
        else:
            notes.append(f"Hit target at {dr['targetHitAt']}.")
    elif realized is not None:
        notes.append(f"Target ${tgt:.0f} not reached (realized ${realized:.2f}).")
    out["discipline"] = {"verdict": sit.get("verdict"), "notes": notes}
    if isinstance(snap, dict) and snap.get("logged"):
        s = snap.get("snapshot") or {}
        out["closingLevels"] = {"spot": s.get("spot"), "gammaFlip": s.get("gamma_flip"),
                                "callWall": s.get("call_wall"), "putWall": s.get("put_wall"),
                                "vix": s.get("vix"), "regime": s.get("regime")}
        out["snapshotLogged"] = True
    else:
        out["snapshotLogged"] = False
    return out


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
    orders = await asyncio.to_thread(_rh_recent_option_orders, monday, 20)
    by = defaultdict(list)
    for o in orders:
        f = _order_to_fill(o)
        if f and monday.isoformat() <= f["trade_date"] <= friday.isoformat():
            by[f["trade_date"]].append(f)
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
    return out


@mcp.tool()
async def tilt_detector(date: Optional[str] = None) -> dict:
    """Scan a session trade sequence for tilt: revenge sizing (size up after losses), rushing (shrinking
    time between entries), intraday win-rate decay, and trading after a give-back from peak. Flags tilt
    while there is still time to stop. date (ET) defaults to today.
    """
    import asyncio
    d = date or _today_et().isoformat()
    orders = await asyncio.to_thread(_rh_recent_option_orders, _dt.date.fromisoformat(d), 10)
    fills = [f for o in orders if (f := _order_to_fill(o)) and f["trade_date"] == d]
    if not fills:
        return {"date": d, "note": "No fills for this session."}
    fills.sort(key=lambda r: r["time"])
    trips = _round_trips(fills)
    if len(trips) < 4:
        return {"date": d, "roundTrips": len(trips),
                "note": f"Only {len(trips)} round trips - too few to assess tilt."}
    flags, ev = [], {}
    after_loss, after_win = [], []
    for i in range(1, len(trips)):
        (after_loss if trips[i - 1]["pnl"] < 0 else after_win).append(abs(trips[i]["qty"]))
    al = _stats.mean(after_loss) if after_loss else 0.0
    aw = _stats.mean(after_win) if after_win else 0.0
    ev["avgQtyAfterLoss"], ev["avgQtyAfterWin"] = round(al, 2), round(aw, 2)
    if aw > 0 and al > 1.25 * aw:
        flags.append(f"Revenge sizing: avg {al:.1f} contracts after a loss vs {aw:.1f} after a win.")
    opens = sorted(tr["open"] for tr in trips)
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
    return {"date": d, "roundTrips": len(trips), "tiltLevel": level, "flags": flags,
            "evidence": ev, "advice": advice}


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
    orders = await asyncio.to_thread(_rh_recent_option_orders, stop, 40)
    calls_credit = puts_credit = btc_debit = 0.0
    sto_calls = sto_puts = 0
    expiries = set()
    for o in orders:
        f = _order_to_fill(o)
        if not f or f["chain"] != sym:
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
    return {"symbol": sym, "lookbackDays": lookback_days,
            "netOptionPremium$": round(net_prem, 2),
            "callPremium$": round(calls_credit, 2), "putPremium$": round(puts_credit, 2),
            "buyToCloseCost$": round(btc_debit, 2),
            "shortCallsSold": sto_calls, "shortPutsSold": sto_puts,
            "expiryCyclesTraded": len([e for e in expiries if e]),
            "shares": round(shares, 4), "avgCost$": round(avg, 2) if avg else None,
            "currentPrice$": hold.get("price"),
            "effectiveBasisAfterPremium$": round(eff_basis, 2) if eff_basis is not None else None,
            "basisReductionPerShare$": round(net_prem / shares, 2) if shares > 0 else None,
            "note": ("Net premium = call+put credits minus buy-to-close debits over the window. "
                     "Assignment/exercise legs may post separately; verify against statements.")}


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
async def spot_blend(spy_mult: float = 10.0, basis: float = 0.0) -> dict:
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
    age_min = None
    try:
        if asof:
            ad = _dt.datetime.fromisoformat(asof.replace("Z", "+00:00"))
            age_min = round((_dt.datetime.now(_dt.timezone.utc) - ad).total_seconds() / 60, 1)
    except Exception:
        pass
    spx_est = (spy * spy_mult + basis) if spy else None
    gap = round(spx_est - chain_spot, 1) if (spx_est and chain_spot) else None

    def side(level):
        if not (spx_est and chain_spot and level):
            return None
        was = "above" if chain_spot > level else "below"
        now = "above" if spx_est > level else "below"
        return {"level": level, "chainSide": was, "liveSide": now, "crossed": was != now}

    return {"chainSpot": chain_spot, "chainAgeMin": age_min, "spyLive": spy, "spyMult": spy_mult,
            "basis": basis, "spxLiveEst": round(spx_est, 1) if spx_est else None, "gapPts": gap,
            "vsGammaFlip": side(flip), "vsCallWall": side(cw), "vsPutWall": side(pw),
            "note": "spxLiveEst = SPY*mult + basis. Calibrate basis to a real SPX quote for accuracy."}


@mcp.tool()
async def pcs_sizer(short_delta: float = 0.30, width: float = 10.0,
                    expiry: Optional[str] = None) -> dict:
    """Size an SPX put credit spread (your ASD 0DTE PCS): from the live chain it picks the short put
    nearest short_delta and the long put width points below, then reports net credit, max loss,
    breakeven, return-on-risk, and an approximate probability of profit. expiry defaults to 0DTE.
    """
    chain = await _load_chain()
    spot = chain.get("spot")
    opts = chain.get("options") or []
    exp = expiry or _nearest_expiry(opts, "SPXW")
    puts = [o for o in _filter(opts, root="SPXW", expiration=exp)
            if o.get("cp") == "P" and o.get("strike")]
    if not puts:
        return {"error": f"No SPXW puts found for {exp}."}
    with_delta = [o for o in puts if o.get("delta") is not None]
    if not with_delta:
        return {"error": "No delta data on puts."}
    short = min(with_delta, key=lambda o: abs(abs(o["delta"]) - short_delta))
    ks = short["strike"]
    longs = [o for o in puts if o["strike"] <= ks - 0.01]
    if not longs:
        return {"error": "No long put strike below the short available."}
    longp = min(longs, key=lambda o: abs(o["strike"] - (ks - width)))
    actual_width = round(ks - longp["strike"], 2)
    sc, lc = (short.get("mid") or 0.0), (longp.get("mid") or 0.0)
    credit = round(sc - lc, 2)
    maxloss = round(actual_width - credit, 2)
    return {"expiry": exp, "spot": spot,
            "shortPut": {"strike": ks, "delta": round(short["delta"], 3), "mid": round(sc, 2)},
            "longPut": {"strike": longp["strike"], "delta": round(longp.get("delta") or 0.0, 3),
                        "mid": round(lc, 2)},
            "width": actual_width, "creditPerSh$": credit, "creditPerContract$": round(credit * 100, 2),
            "maxLossPerSh$": maxloss, "maxLossPerContract$": round(maxloss * 100, 2),
            "breakeven": round(ks - credit, 2),
            "returnOnRisk%": round(100 * credit / maxloss, 1) if maxloss > 0 else None,
            "approxPOP%": round(100 * (1 - abs(short["delta"])), 1),
            "note": ("approxPOP ~ P(short put expires OTM) = 1-abs(delta); true POP is a bit higher. "
                     "Mids are ~15-min delayed.")}


@mcp.tool()
async def event_risk_radar(days: Annotated[int, Field(ge=1, le=60)] = 7) -> dict:
    """What can gap your book in the next N days: high-impact economic events plus any holdings
    reporting earnings, merged into one timeline flagged by what you hold. Combines the economic and
    single-name earnings calendars with your positions.
    """
    import asyncio
    ec, ear, pos = await asyncio.gather(economic_calendar(), earnings_calendar(days=days),
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
    held = {p["symbol"] for p in pos if p.get("type") == "equity"} if isinstance(pos, list) else set()
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
                        state_rate: float = 0.0549) -> dict:
    """Estimated tax set-aside on your realized trading gains: pulls YTD realized short/long-term options
    P&L and applies your marginal federal + Georgia rates, with a quarterly figure. Short-term is taxed
    as ordinary income. Trading gains only - excludes W-2/Schedule C/clergy; not tax advice.
    """
    ts = await tax_summary(year=year)
    if "realizedTotal$" not in ts:
        base = {k: ts[k] for k in ("year", "error", "note") if k in ts}
        return {**base, "note": ts.get("note") or "No realized P&L to estimate."}
    st = ts.get("shortTerm$", 0.0) or 0.0
    lt = ts.get("longTerm$", 0.0) or 0.0
    fed_st = max(0.0, st) * fed_rate
    st_state = max(0.0, st) * state_rate
    ltcg = max(0.0, lt) * 0.15
    lt_state = max(0.0, lt) * state_rate
    total = fed_st + st_state + ltcg + lt_state
    return {"year": ts.get("year"), "realizedShortTerm$": round(st, 2), "realizedLongTerm$": round(lt, 2),
            "assumedFedRate": fed_rate, "assumedStateRate": state_rate,
            "estShortTermTax$": round(fed_st + st_state, 2), "estLongTermTax$": round(ltcg + lt_state, 2),
            "estTotalSetAside$": round(total, 2), "quarterlySetAside$": round(total / 4, 2),
            "note": ("Trading gains only; short-term taxed as ordinary income. Excludes W-2 / Schedule C "
                     "/ clergy housing. Verify with your CPA.")}


def main() -> None:
    log.info("Starting Traders Edge MCP server (stdio); risk-free=%.3f", RISK_FREE)
    mcp.run()


if __name__ == "__main__":
    main()
