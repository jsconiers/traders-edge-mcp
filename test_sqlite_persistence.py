"""Validate the #3 SQLite session-persistence layer against a throwaway DB (no RH, no network)."""
import asyncio
import datetime as _d
import importlib.util
import os
import sys
import tempfile

DB = os.path.join(tempfile.gettempdir(), f"te_test_{os.getpid()}.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["TE_DB_PATH"] = DB   # must be set BEFORE import (read at module load)

_SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "traders_edge_mcp.py")
spec = importlib.util.spec_from_file_location("te", _SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.TE_DB_PATH == DB, m.TE_DB_PATH

FAILS = []


def check(name, cond, extra=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---- build canned fills for a PAST date via the real decomposition path ----
def leg(oid, side, eff, k, ot, px, ts):
    return {"option": f"https://x/options/instruments/{oid}/", "side": side, "position_effect": eff,
            "strike_price": str(k), "option_type": ot, "expiration_date": ts[:10],
            "executions": [{"timestamp": ts, "trade_date": ts[:10], "price": str(px),
                            "quantity": "1"}], "ratio_quantity": "1"}


def order(oid, legs, ts, net, direction):
    return {"id": oid, "state": "filled", "chain_symbol": "SPXW", "updated_at": ts, "created_at": ts,
            "net_amount": str(net), "net_amount_direction": direction, "processed_premium": None,
            "processed_quantity": "1", "quantity": "1", "legs": legs}


o_open = order("O1", [leg("a", "sell", "open", 6800, "put", 5.0, "2026-07-10T14:30:00Z"),
                      leg("b", "buy", "open", 6790, "put", 2.0, "2026-07-10T14:30:00Z")],
               "2026-07-10T14:30:00Z", 300, "credit")
o_close = order("O2", [leg("a", "buy", "close", 6800, "put", 1.0, "2026-07-10T15:30:00Z"),
                       leg("b", "sell", "close", 6790, "put", 0.5, "2026-07-10T15:30:00Z")],
                "2026-07-10T15:30:00Z", 50, "debit")
canned = [o_open, o_close]
fills = [f for o in canned for f in m._order_to_fills(o)]
check("built 4 canned leg fills", len(fills) == 4, str(len(fills)))

# ---- direct sync round-trip: ingest -> read -> reconstruct ----
m._ingest_fills_and_mark_sync(fills, ["2026-07-10"])
read = m._fills_read_range_sync("2026-07-10", "2026-07-10")
check("read back all 4 fills", len(read) == 4, str(len(read)))
check("time reconstructed as aware datetime",
      all(isinstance(f["time"], _d.datetime) and f["time"].tzinfo is not None for f in read))
check("net_cf preserved through DB",
      round(sum(f["net_cf"] for f in read), 2) == round(sum(f["net_cf"] for f in fills), 2))
check("day marked ingested", "2026-07-10" in m._ingested_days_sync())

# reconstructed fills must feed the matcher and give the same 2 trips == cash
trips, stats = m._round_trips_full(sorted(read, key=lambda r: r["time"]))
check("reconstructed fills -> 2 trips == cash",
      len(trips) == 2 and round(sum(t["pnl"] for t in trips), 2) == round(sum(f["net_cf"] for f in read), 2),
      f"trips={len(trips)}")

# idempotent: re-ingest same fills, count unchanged
m._ingest_fills_and_mark_sync(fills, ["2026-07-10"])
check("idempotent upsert (no dupes)", len(m._fills_read_range_sync("2026-07-10", "2026-07-10")) == 4)

# ---- sessions table ----
m._session_upsert_sync({"date": "2026-07-10", "realized": 250.0, "fee_inclusive": 248.5, "fees": 1.5,
                        "trips": 2, "cross_time": "10:15:00", "verdict": "STOP",
                        "updated": "2026-07-10T16:05:00-04:00"})
srows = m._sessions_read_range_sync("2026-07-01", "2026-07-31")
check("session persisted+read", len(srows) == 1 and srows[0]["verdict"] == "STOP" and srows[0]["realized"] == 250.0,
      str(srows))

# ---- _fills_window: cache hit for an already-ingested PAST window, no RH call ----
call_log = {"n": 0}


def boom(*a, **k):
    call_log["n"] += 1
    raise AssertionError("RH should NOT be paged for a fully-ingested past window")


m._rh_recent_option_orders = boom
by, meta = asyncio.run(m._fills_window(_d.date(2026, 7, 10), _d.date(2026, 7, 10)))
check("_fills_window served from cache (no RH)", meta["source"] == "cache" and call_log["n"] == 0, str(meta))
check("_fills_window cache returned the day", len(by.get("2026-07-10", [])) == 4, str({k: len(v) for k, v in by.items()}))

# ---- _fills_window: live path for a NOT-yet-ingested past window, then persists+marks ----
def canned_pull(start, max_pages=40, page_size=100):
    call_log["n"] += 1
    o1 = order("P1", [leg("z", "sell", "open", 6700, "put", 4.0, "2026-07-08T14:30:00Z")],
               "2026-07-08T14:30:00Z", 400, "credit")
    o2 = order("P2", [leg("z", "buy", "close", 6700, "put", 1.0, "2026-07-08T15:30:00Z")],
               "2026-07-08T15:30:00Z", 100, "debit")
    return [o1, o2], {"truncated": False, "pages": 1, "oldestSeen": "2026-07-08"}


m._rh_recent_option_orders = canned_pull
call_log["n"] = 0
by2, meta2 = asyncio.run(m._fills_window(_d.date(2026, 7, 8), _d.date(2026, 7, 8)))
check("_fills_window live pull when uncached", meta2["source"] == "live" and call_log["n"] == 1, str(meta2))
check("live window returned the day's fills", len(by2.get("2026-07-08", [])) == 2, str({k: len(v) for k, v in by2.items()}))
check("live pull marked the past day ingested", "2026-07-08" in m._ingested_days_sync())

# now it should be cache-served (RH must not be called again)
m._rh_recent_option_orders = boom
call_log["n"] = 0
by3, meta3 = asyncio.run(m._fills_window(_d.date(2026, 7, 8), _d.date(2026, 7, 8)))
check("second call now cache-served", meta3["source"] == "cache" and call_log["n"] == 0, str(meta3))

# cleanup
try:
    os.remove(DB)
except OSError:
    pass

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL SQLITE PERSISTENCE TESTS PASSED")
