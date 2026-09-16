import asyncio, datetime as dt, traders_edge_mcp as T

TODAY = dt.date(2026, 9, 16)

def snap(last_trade_time, prev_day_close, close):
    return {"data": {"last_trade_time": last_trade_time,
                     "prev_day_close": prev_day_close, "close": close,
                     "current_price": close}}

async def main():
    T._today_et = lambda: TODAY
    fails = []
    def ck(n, c):
        print(("ok   " if c else "FAIL ") + n)
        if not c: fails.append(n)

    async def run(payload, cachebust):
        T._get_json = lambda u, ttl, _p=payload: asyncio.sleep(0, result=_p)
        T._cache.put(f"spxprev:{TODAY.isoformat()}", None)   # clear
        try: T._cache._d.pop(f"spxprev:{TODAY.isoformat()}", None)
        except Exception: pass
        return await T._cboe_spx_prev_close()

    # 1. THE REAL INCIDENT: pre-roll snapshot dated yesterday
    v = await run(snap("2026-09-15T16:14:59", 7619.98, 7585.73), 1)
    ck("B23 pre-roll uses close, not prev_day_close", v == 7585.73)
    ck("B23 does NOT return Monday's 7619.98", v != 7619.98)

    # 2. post-roll snapshot dated today -> prev_day_close is correct
    v = await run(snap("2026-09-16T10:02:11", 7585.73, 7604.70), 2)
    ck("B23 post-roll uses prev_day_close", v == 7585.73)

    # 3. weekend: Friday snapshot, today Monday -> Friday's close IS the prior session
    T._today_et = lambda: dt.date(2026, 9, 14)
    v = await run(snap("2026-09-11T16:14:59", 7656.98, 7619.98), 3)
    ck("B23 weekend gap uses Friday close", v == 7619.98)
    T._today_et = lambda: TODAY

    # 4. unreadable date -> refuse, never guess
    v = await run(snap("not-a-date", 7619.98, 7585.73), 4)
    ck("B23 unreadable date returns None", v is None)

    # 5. missing date entirely -> refuse
    v = await run({"data": {"prev_day_close": 7619.98, "close": 7585.73}}, 5)
    ck("B23 missing date returns None", v is None)

    # 6. end-to-end: the 36pt error is gone
    T._get_json = lambda u, ttl: asyncio.sleep(0, result=snap("2026-09-15T16:14:59", 7619.98, 7585.73))
    try: T._cache._d.pop(f"spxprev:{TODAY.isoformat()}", None)
    except Exception: pass
    T._spy_quote_sync = lambda: (758.81, 757.39)
    imp, det = await T._spy_implied_spx()
    ck("B23 tier2 now agrees with broker print", abs(imp - 7603.60) < 10)
    ck("B23 tier2 no longer returns ~7639", abs(imp - 7639.70) > 25)
    print(f"     tier2 implied = {imp} (broker print was 7603.60)")

    print("\n" + ("B23 FAILED: " + ", ".join(fails) if fails else "ALL B23 PROBES PASSED"))
    return 1 if fails else 0

raise SystemExit(asyncio.run(main()))
