import os, sys, time, builtins
os.environ["RH_ENV_FILE"] = "/nonexistent/.env"          # never load real credentials
os.environ["RH_USERNAME"], os.environ["RH_PASSWORD"] = "test-user", "test-pass"
import traders_edge_mcp as T
import robin_stocks.robinhood as rh
import robin_stocks.robinhood.authentication as A
import robin_stocks.robinhood.helper as H

def _boom_input(*a, **k): raise AssertionError("input() must NEVER be called inside the server")
builtins.input = _boom_input

calls = {"login": 0}
mtime = {"v": 1000.0}
T._rh_pickle_mtime = lambda: mtime["v"]

def fake_login_challenge(*a, **k):          # walks into the challenge exactly as robin_stocks line 81 does
    calls["login"] += 1
    A._validate_sherrif_id("device-token", "workflow-id")
def fake_login_ok(*a, **k):
    calls["login"] += 1

def reset():
    T._rh_logged_in = False; T._rh_block["pickle_mtime"] = None; calls["login"] = 0

fails = []
def ck(n, c):
    print(("ok   " if c else "FAIL ") + n)
    if not c: fails.append(n)

# 1. THE BUG: a verification challenge must fail FAST, not poll the phone
reset(); rh.login = fake_login_challenge
t0 = time.monotonic()
try: T._rh_login_sync(); raised = None
except T.EdgeError as e: raised = str(e)
dt = time.monotonic() - t0
ck("B26 challenge raises instead of blocking", raised is not None)
ck("B26 challenge fails in under 1s", dt < 1.0)
ck("B26 error says SESSION_EXPIRED", raised and "SESSION_EXPIRED" in raised)
ck("B26 error carries re-auth command", raised and "rh_reauth.py" in raised)
ck("B26 not marked logged-in after challenge", T._rh_logged_in is False)

# 2. no second credential login while the pickle is unchanged (no push-notification spam)
before = calls["login"]
for _ in range(5):
    try: T._rh_login_sync()
    except T.EdgeError: pass
ck("B26 5 retries send ZERO new credential logins", calls["login"] == before)

# 3. once the user re-auths (pickle changes), it retries and succeeds
mtime["v"] = 2000.0; rh.login = fake_login_ok
T._rh_login_sync()
ck("B26 changed pickle triggers exactly one retry", calls["login"] == before + 1)
ck("B26 logged in after out-of-band re-auth", T._rh_logged_in is True)

# 4. stdout protection + hardening
ck("B26 robin_stocks output routed to stderr", H.get_output() is sys.stderr)
ck("B26 interactive handler replaced", getattr(A._validate_sherrif_id, "_te_hardened", False))
T._rh_harden(); T._rh_harden()
ck("B26 hardening is idempotent", getattr(A._validate_sherrif_id, "_te_hardened", False))

# 5. tier-2 helpers degrade instead of breaking the parity gate
reset(); rh.login = fake_login_challenge
ck("B26 _spy_quote_sync -> (None, None) on SESSION_EXPIRED", T._spy_quote_sync() == (None, None))
ck("B26 _spy_live_sync  -> None on SESSION_EXPIRED", T._spy_live_sync() is None)

# 6. B22 detector recognises the new error; hint no longer destroys the device token
ck("B26 _rh_auth_failed recognises SESSION_EXPIRED",
   T._rh_auth_failed(T.EdgeError("Robinhood SESSION_EXPIRED (verification required).")))
ck("B26 hint no longer says 'rm -f' the pickle", "rm -f" not in T.RH_REAUTH_HINT)
ck("B26 hint no longer says 'restart' is needed", "No restart needed" in T.RH_REAUTH_HINT)

print("\n" + ("B26 FAILED: " + ", ".join(fails) if fails else "ALL B26 PROBES PASSED"))
raise SystemExit(1 if fails else 0)
