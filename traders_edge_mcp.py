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


def main() -> None:
    log.info("Starting Traders Edge MCP server (stdio); risk-free=%.3f", RISK_FREE)
    mcp.run()


if __name__ == "__main__":
    main()
