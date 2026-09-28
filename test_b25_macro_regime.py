import asyncio, traders_edge_mcp as T

def raw(n):
    f = getattr(T, n); return getattr(f, "fn", getattr(f, "__wrapped__", f))
async def fast(d): return d

# capture the REAL function before any test stubs replace it
import inspect as _inspect
_REAL_RC_SRC = _inspect.getsource(raw("regime_classifier"))

def stub(gex, macro):
    T.zero_dte_exposure = lambda: fast({"spot": 7707.0, "regime": gex})
    T.regime_classifier = lambda: fast({"regime": macro, "compositeScore": 3,
                                        "posture": "macro posture text"})
    T.vix_complex = lambda: fast({"indices": {"VIX": {"value": 16.0}}})
    T.economic_calendar = lambda: fast({"events": []})
    T.earnings_calendar = lambda days=10: fast({"nextEarnings": []})
    T._recent_session_summary = lambda: fast(None)

async def main():
    fails = []
    def ck(n, c):
        print(("ok   " if c else "FAIL ") + n)
        if not c: fails.append(n)
    mb = raw("morning_brief")

    # 1. rename: macroRegime present, old ambiguous 'regime' key gone
    stub("long gamma (pin / mean-revert)", "RISK-ON")
    r = await mb()
    ck("B25 macroRegime key present", "macroRegime" in r)
    ck("B25 old 'regime' key removed", "regime" not in r)
    ck("B25 macroRegime carries a scope note", "NOT a dealer-gamma read" in r["macroRegime"]["scope"])
    ck("B25 gexRegime still in levels", r["levels"]["gexRegime"].startswith("long gamma"))

    # 2. THE BUG: calm macro + short gamma must raise an explicit conflict
    stub("short gamma (trend / amplify)", "RISK-ON")
    r = await mb()
    ck("B25 calm-macro/short-gamma raises conflict", "regimeConflict" in r)
    ck("B25 conflict says follow breaks, don't fade",
       "follow breaks" in r["regimeConflict"] and "do not fade" in r["regimeConflict"])

    # 3. the inverse conflict
    stub("long gamma (pin / mean-revert)", "RISK-OFF / STRESS")
    r = await mb()
    ck("B25 stressed-macro/long-gamma raises conflict", "regimeConflict" in r)
    ck("B25 inverse conflict says expect pinning", "pinning" in r["regimeConflict"])

    # 4. agreement must NOT raise a false conflict
    stub("long gamma (pin / mean-revert)", "RISK-ON")
    ck("B25 agreeing regimes raise no conflict", "regimeConflict" not in (await mb()))
    stub("short gamma (trend / amplify)", "CAUTION")
    ck("B25 short gamma + caution raises no conflict", "regimeConflict" not in (await mb()))

    # 5. posture strings no longer make gamma claims they never measured
    src = _REAL_RC_SRC
    posture_block = src[src.index("posture = {"):src.index("}[regime]")]
    for bad in ("long-gamma backdrop", "negative gamma", "gamma flips", "fading extremes"):
        ck(f"B25 posture no longer claims '{bad}'", bad not in posture_block)

    print("\n" + ("B25 FAILED: " + ", ".join(fails) if fails else "ALL B25 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
