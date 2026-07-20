#!/usr/bin/env bash
# (#10) One command to run the offline plumbing suite before restarting Claude Desktop.
# No network, no broker. Lives alongside test_traders_edge.py.
set -e
# Prefer the repo virtualenv, fall back to system python3.
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
F="${1:-traders_edge_mcp.py}"
echo "== --selftest ==";              "$PY" "$F" --selftest
echo "== freshness lint (#1) ==";     "$PY" test_freshness_lint.py "$F"
echo "== sqlite persistence (#3) =="; "$PY" test_sqlite_persistence.py "$F"
# Optional B1-B15 plumbing regression: only runs if that script is also present.
if [ -f regress_b1_b15.py ]; then
  echo "== plumbing regression ==";   "$PY" regress_b1_b15.py
else
  echo "== plumbing regression == (skipped: regress_b1_b15.py not in repo)"
fi
echo "ALL OFFLINE SUITES PASSED"
