# Contributing

Issues and pull requests are welcome.

## Development

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m py_compile traders_edge_mcp.py
.venv/bin/python test_traders_edge.py
```

## Guidelines
- Keep tools dependency-light and key-less by default; gate any keyed data source behind an
  optional environment variable with a graceful fallback.
- stdio MCP servers must log to **stderr only** (never stdout) so the protocol stream stays clean.
- Document the convention for any new positioning metric (sign, units, assumptions).
- Add/extend offline tests in `test_traders_edge.py` for new math.
