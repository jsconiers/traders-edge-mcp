import asyncio, json, traders_edge_mcp as T
def raw(n):
    f = getattr(T, n); return getattr(f, "fn", getattr(f, "__wrapped__", f))
async def main():
    fails=[]
    def ck(n,c):
        print(("ok   " if c else "FAIL ")+n)
        if not c: fails.append(n)
    # 1. future date is refused outright
    try:
        await raw("eod_wrap")(date="2099-01-01"); ck("B20 future date refused", False)
    except Exception as e: ck("B20 future date refused", "future" in str(e))
    # 2. malformed date refused
    try:
        await raw("eod_wrap")(date="31-08-2026"); ck("B20 bad format refused", False)
    except Exception as e: ck("B20 bad format refused", "YYYY-MM-DD" in str(e))
    # 3. THE BUG: today (Sep 1, pre-market, no fills) must NOT persist a NULL row
    r = await raw("eod_wrap")()
    ck("B20 empty day does not persist", r["persisted"]["session"] is False)
    ck("B20 empty day explains why", "nothing to persist" in r["persisted"].get("reason",""))
    ck("B20 pre-market not labelled 'session close'",
       r.get("closingLevels",{}).get("source") != "closing snapshot")
    # 4. Monday now reachable by date, with real numbers
    m = await raw("eod_wrap")(date="2026-08-31")
    ck("B20 date param wraps Monday", m["date"] == "2026-08-31")
    ck("B20 Monday realized correct", m["result"].get("realized$") == 120.38)
    ck("B20 Monday trips correct", m["result"].get("trips") == 3)
    ck("B20 Monday persisted", m["persisted"].get("session") is True)
    print("\n"+("B20 FAILED: "+", ".join(fails) if fails else "ALL B20 PROBES PASSED"))
    print("\n--- Monday wrap ---"); print(json.dumps(m, indent=2, default=str))
    return 1 if fails else 0
raise SystemExit(asyncio.run(main()))
