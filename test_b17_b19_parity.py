import asyncio, traders_edge_mcp as T

def mk(spot): return {"spot": spot, "options": [1], "freshness":
    {"feed":"robinhood","marketOpen":True,"verdict":"live","stale":False,"paritySpot":spot}}

async def main():
    T._market_open_et = lambda: True
    # B19: default tier 2 OFF so the B17 probes exercise the broker-print path in isolation.
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(None, {"reason": "stubbed off"}))
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

    # ---- B19: SPY day-over-day tier 2 ----
    T._market_open_et = lambda: True   # probe 6 above closed the session; reopen it
    PREV_SPX, PREV_SPY = 7652.86, 763.47
    def spy(last):   # build the tier-2 reference the way _spy_implied_spx does
        return round(PREV_SPX * (last / PREV_SPY), 2)

    # tier 2 carries the load when the broker print is gone
    T._live_spx_print = lambda: asyncio.sleep(0, result=(None, None))
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(spy(766.495), {"spyLive": 766.495}))
    r = await T._verify_parity_spot(mk(7655.70)); f = r["freshness"]
    ck("B19 no broker print -> tier 2 used", f["refTier"] == "spy_dod")
    ck("B19 tier 2 catches the 09:34 stale marks", f["parityCheck"] == "STALE_MARKS")
    ck("B19 tier 2 supplies corrected spot", abs(r["spot"] - 7685.01) < 5)
    ck("B19 tier 2 spot lands above 7668 flip", r["spot"] > 7668.02)

    # healthy session: tier 2 must NOT false-positive
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(spy(764.745), {"spyLive": 764.745}))
    r = await T._verify_parity_spot(mk(7665.10)); f = r["freshness"]
    ck("B19 tier 2 passes a healthy chain", f["parityCheck"] == "ok")
    ck("B19 tier 2 does not swap spot when ok", r["spot"] == 7665.10)

    # both references gone -> unverified, never a silent pass
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(None, {"reason": "unavailable: spyLive"}))
    r = await T._verify_parity_spot(mk(7655.70)); f = r["freshness"]
    ck("B19 both refs down -> unverified", f["parityCheck"] == "unverified")
    ck("B19 unverified names the reason", "spyLive" in f["parityCheckNote"])

    # broker print present but WRONG: disagreement with tier 2 must be recorded
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7500.0, "etrade_market"))
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(spy(764.745), {}))
    f = (await T._verify_parity_spot(mk(7665.10)))["freshness"]
    ck("B19 bad broker print flagged as disagreement", "refDisagreement" in f)
    ck("B19 tier 1 still authoritative", f["refTier"] == "broker_print")

    # agreeing references must NOT raise a disagreement flag
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7665.17, "etrade_market"))
    f = (await T._verify_parity_spot(mk(7665.10)))["freshness"]
    ck("B19 agreeing refs raise no flag", "refDisagreement" not in f)
    ck("B19 spyImpliedSpx surfaced alongside tier 1", "spyImpliedSpx" in f)

    print("\n" + ("B17 FAILED: " + ", ".join(fails) if fails else "ALL B17/B19 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
