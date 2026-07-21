"""Targeted regression probes for the B1-B15 fixes. Imports the PATCHED module and exercises the
changed logic with synthetic data (no network / no broker)."""
import datetime as _dt
import importlib.util
import os
import sys

_SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "traders_edge_mcp.py")
spec = importlib.util.spec_from_file_location("te", _SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILS = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---- B12: URL redaction ----
r = m._redact_url("https://api.example.com/v1/data?api_key=SUPERSECRET&sym=SPX&token=abc123")
check("B12 redact strips api_key+token", "SUPERSECRET" not in r and "abc123" not in r and "***" in r, r)
check("B12 redact keeps non-secret params", "sym=SPX" in r, r)

# ---- B8: session calendar ----
check("B8 2026-07-03 is a holiday (not trading)", not m._is_trading_day(_dt.date(2026, 7, 3)))
check("B8 2026-07-20 is a trading day", m._is_trading_day(_dt.date(2026, 7, 20)))
check("B8 2026-11-27 half day closes 13:00", m._session_close_et(_dt.date(2026, 11, 27)) == _dt.time(13, 0))
check("B8 normal day closes 16:00", m._session_close_et(_dt.date(2026, 7, 20)) == _dt.time(16, 0))
check("B8 market closed on July 3 holiday",
      not m._market_open_et(_dt.datetime(2026, 7, 3, 11, 0, tzinfo=m.ET)))
check("B8 market open normal weekday 11:00",
      m._market_open_et(_dt.datetime(2026, 7, 20, 11, 0, tzinfo=m.ET)))
check("B8 market closed 13:30 on half day",
      not m._market_open_et(_dt.datetime(2026, 11, 27, 13, 30, tzinfo=m.ET)))


# ---- B2: multi-leg decomposition + round trips ----
def leg(opt_id, side, effect, strike, otype, price, qty, ts, td="2026-07-20"):
    return {"option": f"https://api.robinhood.com/options/instruments/{opt_id}/",
            "side": side, "position_effect": effect, "strike_price": str(strike),
            "option_type": otype, "expiration_date": "2026-07-20",
            "executions": [{"timestamp": ts, "trade_date": td, "price": str(price),
                            "quantity": str(qty)}], "ratio_quantity": "1"}


def order(oid, legs, ts, net_amount, direction):
    return {"id": oid, "state": "filled", "chain_symbol": "SPXW", "legs": legs,
            "updated_at": ts, "created_at": ts, "net_amount": str(net_amount),
            "net_amount_direction": direction, "processed_premium": None,
            "processed_quantity": "1", "quantity": "1"}


open_o = order("O1", [
    leg("opt6800", "sell", "open", 6800, "put", 5.00, 1, "2026-07-20T14:30:00Z"),
    leg("opt6790", "buy", "open", 6790, "put", 2.00, 1, "2026-07-20T14:30:00Z"),
], "2026-07-20T14:30:00Z", 300, "credit")
close_o = order("O2", [
    leg("opt6800", "buy", "close", 6800, "put", 1.00, 1, "2026-07-20T15:30:00Z"),
    leg("opt6790", "sell", "close", 6790, "put", 0.50, 1, "2026-07-20T15:30:00Z"),
], "2026-07-20T15:30:00Z", 50, "debit")

fills = m._order_to_fills(open_o) + m._order_to_fills(close_o)
check("B2 spread decomposes to 4 leg fills", len(fills) == 4, str(len(fills)))
check("B2 every leg fill carries order_id", all(f.get("order_id") for f in fills))
check("B2 leg effects present (not None)", all(f.get("effect") in ("open", "close") for f in fills))
cash = round(sum(f["net_cf"] for f in fills), 2)
trips, stats = m._round_trips_full(fills)
tot = round(sum(t["pnl"] for t in trips), 2)
check("B2 spread yields 2 round trips", len(trips) == 2, str(len(trips)))
check("B2 round-trip P&L == net cash flow (+250)", tot == 250.0 and cash == 250.0, f"tot={tot} cash={cash}")
check("B2 no unmatched/fallback on clean spread",
      stats["unmatchedCloses"] == 0 and stats["orderLevelFallback"] == 0, str(stats))
check("B2 trips carry open/close order ids",
      all(t.get("open_order_id") == "O1" and t.get("close_order_id") == "O2" for t in trips))

# single-leg still works (regression)
so = order("S1", [leg("optX", "sell", "open", 6800, "put", 3.00, 1, "2026-07-20T14:00:00Z")],
           "2026-07-20T14:00:00Z", 300, "credit")
sc = order("S2", [leg("optX", "buy", "close", 6800, "put", 1.00, 1, "2026-07-20T15:00:00Z")],
           "2026-07-20T15:00:00Z", 100, "debit")
sf = m._order_to_fills(so) + m._order_to_fills(sc)
st, sstats = m._round_trips_full(sf)
check("B2 single-leg round trip pnl == +200",
      len(st) == 1 and round(st[0]["pnl"], 2) == 200.0, str(st))

# unmatched close (open predates window)
lone_close = order("C9", [leg("optZ", "buy", "close", 6800, "put", 1.00, 1, "2026-07-20T15:00:00Z")],
                   "2026-07-20T15:00:00Z", 100, "debit")
_t2, ustats = m._round_trips_full(m._order_to_fills(lone_close))
check("B2 unmatched close is counted", ustats["unmatchedCloses"] == 1, str(ustats))

# ---- B7: recent-orders tuple shape + truncation flag semantics (via warnings helper) ----
w = m._fill_data_warnings({"truncated": True, "legFallback": 0}, {"unmatchedCloses": 0, "orderLevelFallback": 0})
check("B7 truncation surfaces a warning", any("page cap" in x for x in w), str(w))
w2 = m._fill_data_warnings({"truncated": False}, {"unmatchedCloses": 2, "unmatchedCloseCF$": -30.0,
                                                  "orderLevelFallback": 0})
check("B2 unmatched-close surfaces a warning", any("no matching open" in x for x in w2), str(w2))

# ---- B1: index option delta$ uses chain spot when broker underlying price missing ----
p = {"broker": "robinhood", "type": "option", "qty": -2, "symbol": "SPXW_P6800",
     "underlying": "SPXW", "strike": 6800, "expiry": "2026-07-20", "cp": "P",
     "delta": -0.28, "gamma": 0.001, "theta": -1.0, "vega": 1.0, "mv": None}
res = m._position_risk(p, {}, 6850.0, {})   # empty prices -> must fall back to spot 6850
check("B1 SPXW delta$ non-zero via chain-spot fallback", abs(res["delta$"]) > 100000, str(res.get("delta$")))
check("B1 greeksSource stays 'broker' when spot resolves", res["greeksSource"] == "broker", res["greeksSource"])
# and when NO spot either, it must LABEL the zeroing rather than pretend
res0 = m._position_risk(p, {}, 0.0, {})
check("B1 zero-underlying case is labelled",
      res0["delta$"] == 0.0 and "NO UNDERLYING" in res0["greeksSource"], res0["greeksSource"])

# ---- B5: _latest_obs continues past an empty window instead of bailing ----
calls = {"n": 0}
_orig_fetch = m._fetch_series


def fake_fetch(series_id, start=None, **kw):
    calls["n"] += 1
    if calls["n"] == 1:
        raise m.FredError("no observations in window")
    return [("2026-07-01", 1.23)]


m._fetch_series = fake_fetch
try:
    got = m._latest_obs("DGS10")
finally:
    m._fetch_series = _orig_fetch
check("B5 latest_obs falls through to wider window", got == ("2026-07-01", 1.23), str(got))
check("B5 latest_obs actually retried (2 attempts)", calls["n"] == 2, str(calls["n"]))

# ---- B15: market-internals ratio aligns on shared DATES, not position ----
# den is missing the last date; positional zip would wrongly pair num's last close with den's.
num = {"2026-01-02": 10.0, "2026-01-05": 20.0, "2026-01-06": 99.0}
den = {"2026-01-02": 2.0, "2026-01-05": 5.0}
ratio = m._mi_ratio(num, den)
check("B15 mi_ratio date-aligns (drops unmatched 2026-01-06)", ratio == [5.0, 4.0], str(ratio))
# legacy list path preserved
check("B15 mi_ratio legacy list path", m._mi_ratio([10.0, 20.0], [2.0, 5.0]) == [5.0, 4.0])

# ---- B4: pcs put-delta recompute is wired (function object exists & module compiles) ----
check("B4 pcs_sizer present", callable(getattr(m, "pcs_sizer", None)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL REGRESSION PROBES PASSED")
