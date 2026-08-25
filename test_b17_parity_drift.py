import asyncio, traders_edge_mcp as T

def mk(spot): return {"spot": spot, "options": [1], "freshness":
    {"feed":"robinhood","marketOpen":True,"verdict":"live","stale":False,"paritySpot":spot}}

async def main():
    T._market_open_et = lambda: True
    fails = []
    def ck(n, c):
        print(("ok   " if c else "FAIL ") + n)
        if not c: fails.append(n)

    # 1. The real 2026-08-25 incident: parity 7655.70 vs live 7685.01
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7685.01, "etrade_market"))
    r = await T._verify_parity_spot(mk(7655.70)); f = r["freshness"]
    ck("B17 29pt drift flagged STALE_MARKS", f["parityCheck"] == "STALE_MARKS")
    ck("B17 spot swapped to live print", r["spot"] == 7685.01)
    ck("B17 stale flag set", f["stale"] is True)
    ck("B17 original parity spot preserved", f["paritySpot"] == 7655.7)
    ck("B17 premium-derived flagged stale", f["premiumDerivedStale"] is True)
    ck("B17 drift reported", f["parityDriftPts"] == 29.31)
    # the whole point: spot must land ABOVE the 7668.02 flip, not below
    ck("B17 spot now correct side of 7668 flip", r["spot"] > 7668.02)

    # 2. Normal noise (4pt wide-spread jitter) must NOT trip
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7689.0, "etrade_market"))
    r = await T._verify_parity_spot(mk(7685.0)); f = r["freshness"]
    ck("B17 4pt noise stays ok", f["parityCheck"] == "ok")
    ck("B17 noise does not swap spot", r["spot"] == 7685.0)
    ck("B17 noise leaves stale False", f.get("stale") is False)

    # 3. Boundary: 15bp exactly = pass, just over = fail
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7685.0*1.0015, "x"))
    ck("B17 exactly 15bp passes", (await T._verify_parity_spot(mk(7685.0)))["freshness"]["parityCheck"] == "ok")
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7685.0*1.0016, "x"))
    ck("B17 16bp trips", (await T._verify_parity_spot(mk(7685.0)))["freshness"]["parityCheck"] == "STALE_MARKS")

    # 4. No print available must NOT silently confirm
    T._live_spx_print = lambda: asyncio.sleep(0, result=(None, None))
    r = await T._verify_parity_spot(mk(7655.70)); f = r["freshness"]
    ck("B17 no print -> unverified", f["parityCheck"] == "unverified")
    ck("B17 no print does not fabricate a swap", r["spot"] == 7655.70)

    # 5. Downside drift (live BELOW parity) must trip too
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7600.0, "x"))
    ck("B17 negative drift trips", (await T._verify_parity_spot(mk(7655.70)))["freshness"]["parityCheck"] == "STALE_MARKS")

    # 6. Market closed = no-op (last-close marks are correct then)
    T._market_open_et = lambda: False
    r = await T._verify_parity_spot(mk(7655.70))
    ck("B17 closed session is a no-op", "parityCheck" not in r["freshness"])

    print("\n" + ("B17 FAILED: " + ", ".join(fails) if fails else "ALL B17 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
