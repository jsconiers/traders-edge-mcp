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

## Tools (42)

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

### 0DTE decision support
| Tool | What it does |
|------|--------------|
| `expected_move` | ATM-straddle implied range (~1-sigma) for the session, plus +/-1 & +/-2 sigma levels and the IV-based move. |
| `strike_probabilities` | Per-strike risk-neutral prob-ITM and prob-of-touch (Black-Scholes from each strike's IV). |
| `daily_game_plan` | One call for today's 0DTE map: expected-move bands + gamma flip/walls + max-pain + high-OI pins, assembled into support/resistance. |

### Tier 2 — Macro context (FRED, key-less)
| Tool | What it does |
|------|--------------|
| `fed_funds` | Current Fed Funds rate + recent monthly path. |
| `yield_curve` | Treasury curve (3M–30Y), 2s10s / 3m10s spreads, inversion flags. |
| `inflation` | CPI / core CPI / PCE / core PCE (YoY) + 5Y/10Y breakevens. |
| `labor_market` | Unemployment, payroll change, participation, wages, claims. |
| `growth` | Real GDP, industrial production, retail sales. |
| `financial_conditions` | NFCI, HY & IG credit spreads, dollar index, VIX. |
| `recession_indicators` | Sahm Rule, curve spreads, composite read. |
| `series` / `latest` | Any FRED series ID over a window, or latest values for a list. |
| `series_search` | Catalog keyword search (needs free `FRED_API_KEY`). |
| `fred_status` | FRED health check. |

Macro data is pulled key-less from the FRED fredgraph CSV endpoint.

### Cross-broker risk / Greeks aggregator
| Tool | What it does |
|------|--------------|
| `net_greeks` | Net dollar delta / gamma / theta / vega across Alpaca + your positions file; delta also in SPX points. |
| `risk_summary` | Beta-weighted SPX exposure, gross/long/short notional, by-broker & by-underlying breakdowns, top contributors. |
| `concentration` | Exposure % by underlying; flags names above the threshold (default 25%, `CONCENTRATION_PCT`). |
| `scenario_shock` | Portfolio P&L across a set of SPX % moves (delta + gamma convexity). |
| `daily_target` | Today's realized P&L vs your daily target (`DAILY_TARGET`, default $524), with a post-target discipline check. |
| `robinhood_positions` | Live Robinhood holdings (stocks + option legs with broker-provided Greeks). |
| `etrade_positions` | Live E\*TRADE holdings (stocks + options; SPX/SPXW priced via CBOE). |
| `alpaca_positions` / `load_positions` | Raw position views from each source. |
| `risk_status` | Which position sources are configured / reachable. |

Positions are pulled **automatically** from your **Alpaca**, **Robinhood**, and **E\*TRADE** accounts,
and can be supplemented with a broker-agnostic **positions file** for anything held elsewhere:

- **Alpaca** — live `/v2/positions` (creds via `ALPACA_ENV_FILE`, default the alpaca-mcp `.env`).
- **Robinhood** — stock holdings plus option legs (with broker-provided delta/gamma/theta/vega/IV) via
  the cached `robin_stocks` session shared with the robinhood-local server. Creds from `RH_USERNAME`/
  `RH_PASSWORD` (or `RH_ENV_FILE`, default the robinhood-local `.env`); the session pickle lives in
  `~/.robinhood/` and refreshes every 7 days (a one-time device-approval prompt may appear in the
  Robinhood app on first use after expiry).
- **E\*TRADE** — stock + option positions via the cached `pyetrade` OAuth session shared with the etrade
  MCP (`~/.etrade/tokens.pickle`; idle tokens auto-renew). Creds from `ETRADE_CONSUMER_KEY`/`SECRET`
  (or `ET_ENV_FILE`). E\*TRADE access tokens expire nightly — if expired, re-authorize via the etrade
  MCP (`setup_etrade_auth.py`). SPX/SPXW E\*TRADE options are priced from CBOE; equity-option Greeks
  from E\*TRADE aren't fetched yet.
- **Positions file** — default `~/.trading/positions.json` (override `POSITIONS_FILE`).

Each source can be toggled per call via `include_alpaca` / `include_robinhood` / `include_etrade` /
`include_file`. SPX/SPXW options are auto-priced from CBOE; broker-supplied option Greeks are used
directly; equities are **beta-weighted** for SPX-equivalent exposure via a built-in beta map (editable
with `BETA_OVERRIDES="ICE:1.05,NVDA:1.7"` or `BETA_MAP_FILE=<json>`; unmapped symbols default to 1.0).
Example positions file:

```json
{"positions": [
  {"broker": "robinhood", "symbol": "ICE", "qty": 500, "type": "equity", "beta": 1.05},
  {"broker": "robinhood", "symbol": "SPXW260620P07400000", "qty": -2, "type": "option"}
]}
```

### Discipline / behavioral (Robinhood fills)
| Tool | What it does |
|------|--------------|
| `daily_pnl_curve` | Realized-P&L curve from your option fills (net of fees), with the target-cross marked and the give-back-after-target quantified. |
| `daily_review` | End-of-day scorecard: win rate, expectancy, profit factor, P&L by hour, and the before-vs-after-target split. |
| `should_i_trade` | Real-time GO / CAUTION / STOP gate from past-target status, give-back from peak, consecutive losses, churning, and time-of-session. |

Realized P&L is reconstructed from Robinhood option fills (`net_amount`, fees included) with round trips
matched open->close FIFO. These tools target the logged pattern of giving back gains after hitting target;
a recon note flags any day where positions expired or remain open (net cash flow != round-trip realized).

### Position management & macro regime
| Tool | What it does |
|------|--------------|
| `covered_call_manager` | Scans your Robinhood short calls: DTE, assignment prob (delta), premium captured vs extrinsic left, annualized yield, share-coverage check, earnings-before-expiry flag, and roll signals. |
| `earnings_calendar` | Next single-name earnings for your holdings (or a symbol list): date, BMO/AMC session, days away, within-window flag; ETFs/funds listed separately. |
| `regime_classifier` | Folds VIX + VIX term structure + NFCI + HY credit spreads + 2s10s curve + Sahm rule into one risk-on/neutral/risk-off score with a 0DTE posture. |

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
