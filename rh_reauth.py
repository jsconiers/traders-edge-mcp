#!/usr/bin/env python3
"""Interactive Robinhood re-auth for traders-edge (B26).

Run this in a Terminal -- NEVER inside an MCP server. It may ask you to approve a
device prompt in the Robinhood app, or to type an SMS/email code. The server
deliberately refuses to do either, because in a stdio MCP server stdin/stdout ARE
the protocol channel.

It writes the session pickle to exactly the path traders-edge reads, using the same
credentials file, and it does NOT delete the existing pickle first: that pickle
carries the trusted device token, and a brand-new device guarantees a challenge.

    cd ~/Claude/MCP/traders-edge-mcp && .venv/bin/python rh_reauth.py

No Claude Desktop restart is needed afterwards -- the server notices the pickle
changed and uses it on its next call.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv            # noqa: E402
import robin_stocks.robinhood as rh      # noqa: E402

import traders_edge_mcp as T             # noqa: E402  (canonical paths/defaults only)


def main() -> int:
    envf = os.environ.get("RH_ENV_FILE", T.RH_ENV_DEFAULT)
    if os.path.exists(envf):
        load_dotenv(envf)
    user, pw = os.environ.get("RH_USERNAME"), os.environ.get("RH_PASSWORD")
    if not (user and pw):
        print(f"RH_USERNAME / RH_PASSWORD not found (looked in {envf}).", file=sys.stderr)
        return 2
    pickle_dir = os.environ.get("RH_PICKLE_PATH", T.RH_PICKLE_DIR_DEFAULT)
    target = T._rh_pickle_file()
    before = T._rh_pickle_mtime()
    print(f"Re-authenticating Robinhood. Session file: {target}")
    print("If prompted, approve the device login in the Robinhood app, or enter the code.\n")
    rh.login(user, pw, store_session=True, pickle_path=pickle_dir,
             pickle_name=os.environ.get("RH_PICKLE_NAME", ""), expiresIn=86400 * 7)
    after = T._rh_pickle_mtime()
    if after is None or after == before:
        print("\nWARNING: login returned but the session file was not rewritten. "
              "The server will keep reporting SESSION_EXPIRED.", file=sys.stderr)
        return 1
    print("\nDone. traders-edge will use the new session on its next call -- no restart needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
