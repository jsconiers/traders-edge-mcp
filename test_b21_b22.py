import asyncio, traders_edge_mcp as T

def mk(spot, src):
    return {"spot": spot, "options": [1], "source": src,
            "freshness": {"feed": src, "marketOpen": True, "verdict": "fresh",
                          "stale": False, "paritySpot": spot}}

async def main():
    T._market_open_et = lambda: True
    T._spy_implied_spx = lambda: asyncio.sleep(0, result=(None, {"reason": "off"}))
    fails = []
    def ck(n, c):
        print(("ok   " if c else "FAIL ") + n)
        if not c: fails.append(n)

    # --- B21: the real 2026-09-01 12:30 incident, on the CBOE path ---
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7642.76, "etrade_market"))
    r = await T._verify_parity_spot(mk(7653.48, "cboe_delayed")); f = r["freshness"]
    ck("B21 delayed CBOE spot corrected", r["spot"] == 7642.76)
    ck("B21 labelled DELAYED_CHAIN_CORRECTED", f["parityCheck"] == "DELAYED_CHAIN_CORRECTED")
    ck("B21 not mislabelled as RH stale marks", "RH option marks" not in f["warning"])
    ck("B21 original chain spot preserved", f["paritySpot"] == 7653.48)
    ck("B21 drift reported", f["parityDriftPts"] == -10.72)
    ck("B21 premium-derived flagged", f["premiumDerivedStale"] is True)
    # the whole point: spot must land BELOW the 7650 put wall, not above it
    ck("B21 lands on correct side of 7650 put wall", r["spot"] < 7650.0)

    # RH path keeps the WIDER parity bar: 10.72pt (14bp) is inside 15bp and must NOT trip there.
    # That asymmetry is the whole point of B21 -- same drift, different meaning per source.
    r2b = await T._verify_parity_spot(mk(7653.48, "robinhood")); f2b = r2b["freshness"]
    ck("B21 same drift does NOT trip on RH path", f2b["parityCheck"] == "ok")
    ck("B21 RH spot left alone", r2b["spot"] == 7653.48)
    # RH path must still trip + keep its wording on a genuine stale-marks gap (the B17 case)
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7685.01, "etrade_market"))
    r2 = await T._verify_parity_spot(mk(7655.70, "robinhood")); f2 = r2["freshness"]
    ck("B21 RH path still STALE_MARKS", f2["parityCheck"] == "STALE_MARKS")
    ck("B21 RH wording intact", "RH option marks" in f2["warning"])

    # in-tolerance CBOE must pass and keep its source label
    T._live_spx_print = lambda: asyncio.sleep(0, result=(7653.9, "etrade_market"))
    f3 = (await T._verify_parity_spot(mk(7653.48, "cboe_delayed")))["freshness"]
    ck("B21 in-tolerance CBOE passes", f3["parityCheck"] == "ok")
    ck("B21 source labelled cboe not rh", f3["spotSource"].startswith("cboe_delayed"))

    # --- B22: auth-failure detector ---
    ck("B22 detects the null-deref symptom",
       T._rh_auth_failed(AttributeError("'NoneType' object has no attribute 'get'")))
    ck("B22 detects 401", T._rh_auth_failed(Exception("401 Client Error: Unauthorized for url")))
    ck("B22 detects Unauthorized text", T._rh_auth_failed(Exception("Unauthorized")))
    ck("B22 ignores unrelated errors", not T._rh_auth_failed(ValueError("bad strike 7650")))
    ck("B22 ignores timeouts", not T._rh_auth_failed(TimeoutError("probe timed out")))
    ck("B22 hint names re-auth", "pickle" in T.RH_REAUTH_HINT and "restart" in T.RH_REAUTH_HINT)

    print("\n" + ("B21/B22 FAILED: " + ", ".join(fails) if fails else "ALL B21/B22 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
