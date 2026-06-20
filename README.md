# Traders Edge MCP

A consolidated **0DTE-focused options cockpit** for SPX / SPXW, exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server. It pulls **free, key-less**
market data and turns it into the dealer-positioning, volatility, and event signals an index-options
scalper actually watches — chain & Greeks, gamma exposure (GEX), the zero-gamma flip, call/put walls,
max-pain, 0DTE expected move, dealer DEX / vanna / charm, the full VIX term structure, and an
economic-event clock.

Greeks are **recomputed analytically** (vectorized Black–Scholes via numpy) from open interest and
implied vol, with proper Eastern-time time-to-expiry so 0DTE gamma stays realistic into the bell.

> **Data is ~15 minutes delayed** (CBOE delayed quotes). That is fine for *positioning and regime*.
> Overlay a live broker quote (e.g. Robinhood/E\*TRADE/Alpaca MCP) for execution pricing.

## Tools (12)

### Chain & Greeks
| Tool | What it does |
|------|--------------|
| `options_chain` | SPX/SPXW chain near the money with IV + recomputed delta/gamma. Defaults to the nearest SPXW expiry; `zero_dte=True` for today. |
| `option_quote` | Full detail for one OCC symbol (e.g. `SPXW260619C05500000`): quote, IV, delta/gamma/vanna/charm. |
| `expirations` | Available SPX/SPXW expirations and days-to-expiry. |

### Dealer positioning
| Tool | What it does |
|------|--------------|
| `gamma_exposure` | Total dealer **GEX** ($ per 1% move) + the **zero-gamma flip** level and long/short-gamma regime. |
| `gamma_walls` | **Call wall / put wall** (largest gamma strikes), net-gamma strikes, and **max-pain** for an expiration. |
| `zero_dte_exposure` | One-shot **0DTE dashboard**: GEX, flip, walls, max-pain pin, expected move (ATM straddle), gamma concentration. |
| `dealer_exposure` | Dealer **DEX** (dollar delta), **vanna** (per 1% vol), and **charm** (per day) exposure. |

### Vol complex
| Tool | What it does |
|------|--------------|
| `vix_complex` | VIX1D / VIX9D / VIX / VIX3M / VVIX / SKEW with a regime read. |
| `vix_term_structure` | Front-to-back VIX curve + contango/backwardation regime. |

### Event clock
| Tool | What it does |
|------|--------------|
| `economic_calendar` | Upcoming high-impact US macro events + **live Treasury auctions** over N days. |
| `next_event` | The single next macro event with an ET countdown. |

Plus `traders_edge_status` (health check / current spot).

## Data sources (no API key required)

- **CBOE delayed quotes** — the keyless backbone:
  - Option chain: `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json`
    (contains both AM-settled monthly **SPX** and PM-settled **SPXW** weeklies/0DTE — ~32k contracts
    with open interest, IV, and Greeks).
  - Vol indices: `https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{SYM}.json`
- **TreasuryDirect** — live upcoming auctions: `https://www.treasurydirect.gov/TA_WS/securities/upcoming`

**Optional:** set `FMP_API_KEY` or `FINNHUB_API_KEY` for a fully live economic calendar (tick-precise
CPI / PCE / PPI release dates). Without a key, the calendar is built from rule-based releases
(jobless claims, NFP, ISM), the 2026 FOMC schedule, a curated macro table, and live Treasury auctions —
every event is **source-tagged** so you know its provenance.

## Methodology & conventions

- **Greeks:** vectorized Black–Scholes, `q=0` (index options), `r=TE_RISK_FREE` (default 4.3%; gamma is
  ~insensitive to it). Normal CDF via an Abramowitz–Stegun approximation (max error ~7e-8); no scipy.
- **Time to expiry:** years from now (ET) to **16:00 ET** on the expiration date, floored at ~30 minutes
  so 0DTE gamma stays finite at the close.
- **GEX convention:** dealers assumed **long calls / short puts** → call gamma adds, put gamma subtracts.
  Dollar gamma per 1% move per option = `gamma × OI × 100 × spot² × 0.01`. Positive total GEX ⇒ dealers
  long gamma (vol-dampening / mean-reverting); negative ⇒ short gamma (moves amplified).
- **Zero-gamma flip:** net signed dollar gamma is recomputed across 81 spot levels (±10%); the flip is
  the zero-crossing nearest spot.
- **Max-pain:** the strike minimizing total option-holder intrinsic payout for that expiration.
- **Expected move (0DTE):** the ATM straddle mid (~1-sigma for the session).
- **DEX / vanna / charm:** same dealer (long-calls / short-puts) sign convention as GEX.

> These are standard market-positioning heuristics computed from delayed open interest, **not** a
> guarantee of dealer books or future price. Use as one input alongside your own read.

## Install

```bash
cd traders-edge-mcp
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Configure (Claude Desktop)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "traders-edge": {
      "command": "/Users/<you>/Claude/mcp/traders-edge-mcp/.venv/bin/python",
      "args": ["/Users/<you>/Claude/mcp/traders-edge-mcp/traders_edge_mcp.py"]
    }
  }
}
```

(See `claude_desktop_config.example.json`.) Restart Claude Desktop after editing.

## Test

```bash
.venv/bin/python test_traders_edge.py     # offline math/parsing tests
```

## Disclaimer

For research and educational use only. Not investment advice. Market data is delayed; positioning
metrics are modeled heuristics. You are responsible for your own trading decisions.

## License

MIT — see [LICENSE](LICENSE).
