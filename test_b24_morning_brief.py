import asyncio, time, traders_edge_mcp as T

def raw(n):
    f = getattr(T, n); return getattr(f, "fn", getattr(f, "__wrapped__", f))

async def fast(d):  return d
async def hang():   await asyncio.sleep(999)

def stub(**over):
    base = dict(
        zero_dte_exposure=lambda: fast({"spot": 7700.0, "regime": "long gamma"}),
        regime_classifier=lambda: fast({"regime": "RISK-ON"}),
        vix_complex=lambda: fast({"indices": {"VIX": {"value": 15.0}}}),
        economic_calendar=lambda: fast({"events": []}),
        earnings_calendar=lambda days=10: fast({"nextEarnings": []}),
    )
    base.update(over)
    for k, v in base.items():
        setattr(T, k, v)
    T._recent_session_summary = lambda: fast({"date": "2026-09-18", "pnl$": 1.0})

async def main():
    fails = []
    def ck(n, c):
        print(("ok   " if c else "FAIL ") + n)
        if not c: fails.append(n)
    T.MORNING_BRIEF_DEP_TIMEOUT = 1.0
    mb = raw("morning_brief")

    # 1. THE BUG: a hung regime_classifier must not hang the brief
    stub(regime_classifier=hang)
    t0 = time.monotonic(); r = await mb(); dt = time.monotonic() - t0
    ck("B24 hung dependency does not hang the brief", dt < 5)
    ck("B24 hung dependency is NAMED in degraded",
       any("regime_classifier" in d and "timed out" in d for d in r.get("degraded", [])))
    ck("B24 other sections survive (spot still present)", r["levels"]["spot"] == 7700.0)
    ck("B24 other sections survive (vol still present)", r["vol"]["vix"] == 15.0)
    ck("B24 empty-not-zero note present", "EMPTY, not zero" in r.get("degradedNote", ""))

    # 2. a raising dependency is named by exception type
    async def boom(): raise ValueError("x")
    stub(vix_complex=boom)
    r = await mb()
    ck("B24 erroring dependency named by type",
       any("vix_complex" in d and "ValueError" in d for d in r.get("degraded", [])))

    # 3. trailing session summary is bounded too
    stub(); T._recent_session_summary = hang
    t0 = time.monotonic(); r = await mb(); dt = time.monotonic() - t0
    ck("B24 hung session summary is bounded", dt < 5)
    ck("B24 hung session summary is named",
       any("_recent_session_summary" in d for d in r.get("degraded", [])))

    # 4. all healthy -> no degraded key at all (no false alarm)
    stub()
    r = await mb()
    ck("B24 healthy brief carries no degraded flag", "degraded" not in r)

    # 5. the FRED user-agent is no longer the tarpitted value
    ck("B24 FRED_UA is not the tarpitted 'Mozilla/5.0'", T.FRED_UA != "Mozilla/5.0")
    ck("B24 FRED_UA identifies the tool honestly", "traders-edge-mcp" in T.FRED_UA)

    print("\n" + ("B24 FAILED: " + ", ".join(fails) if fails else "ALL B24 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
