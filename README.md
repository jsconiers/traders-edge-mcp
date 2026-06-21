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

## Tools (62)

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

### Performance, tax & snapshot history
| Tool | What it does |
|------|--------------|
| `discipline_backtest` | Replays your fills through the stop-at-target rule: actual vs stop-at-target P&L, the after-target leak (losing days), win rate, expectancy, profit factor, an equity curve, and by-day-of-week / by-hour breakdowns. |
| `tax_summary` | Year-to-date realized options P&L (short vs long term, by month, gross gains/losses) plus identical-contract wash-sale candidates. CPA hand-off; not tax advice. |
| `snapshot_log` | Logs the current 0DTE state (spot, GEX, gamma flip, call/put walls, max-pain, expected move, VIX/VIX1D, regime) to local SQLite. |
| `snapshot_history` | Reads back the day's snapshots and summarizes intraday drift — GEX migration and where the key levels moved. |
| `roll_candidates` | Roll-up-and-out targets for a covered call: candidate strikes/expiries with mark, delta, net credit vs closing the current call, and annualized yield. |

### Configuration
| Tool | What it does |
|------|--------------|
| `trading_config` | View or change your goals/discipline settings (daily target, give-back %, roll thresholds…) in `config.json` — live, no restart. |

Goals and discipline thresholds live in **`config.json`** next to the server (or point `TE_CONFIG_FILE`
elsewhere). Precedence is **env var > `config.json` > built-in default**, and edits are picked up live
(no restart). Change them by editing the file or via the tool — e.g. `trading_config(action="set",
key="daily_target", value="550")`. Editable keys: `daily_target`, `weekly_target`, `giveback_frac`,
`rapid_reentry_secs`, `late_session_et`, `max_trades_per_day`, `roll_delta`, `roll_dte`. See
`config.example.json`.

### Daily workflow (v0.8.0)
| Tool | What it does |
|------|--------------|
| `morning_brief` | Pre-open command center: regime + posture, key 0DTE levels (spot, expected move, gamma flip, call/put walls, max-pain), the vol complex, high-impact econ events, holdings reporting earnings within ~7 days, your last session result, and the discipline reset. |
| `eod_wrap` | End-of-day wrap: realized vs target, discipline adherence (stopped at target vs gave back), where the key levels closed, and a snapshot logged to history. |
| `weekly_review` | This week realized P&L vs your weekly target: Mon-Fri daily breakdown, best/worst day, win rate, progress to goal. |
| `tilt_detector` | Scans a session trade sequence for tilt: revenge sizing, rushing (shrinking entry gaps), intraday win-rate decay, and trading after a give-back from peak. |

### Wheel & income (v0.8.0)
| Tool | What it does |
|------|--------------|
| `wheel_tracker` | Lifetime wheel scorecard for a symbol: net option premium (calls + puts), contracts sold to open, buy-to-close cost, cycles, share position, and effective cost basis after premium. |
| `covered_call_writer` | Fresh covered calls to write on a holding: OTM strikes near a target delta across expiries, ranked by annualized yield, with contracts covered and an earnings/ex-dividend-before-expiry flag. |
| `csp_finder` | Cash-secured puts to sell: OTM strikes near a target delta ranked by annualized yield on the cash secured, with cash required per contract and an earnings flag. |
| `dividend_calendar` | Projected next ex-dividend dates for your holdings (last ex-date + frequency), with payout cadence, dividend/share, and yield -- drives early-assignment risk on short calls. |

### Risk analytics (v0.8.0)
| Tool | What it does |
|------|--------------|
| `correlation_matrix` | Daily-return correlation across holdings: pairwise matrix, each name average correlation, most/least correlated pairs, and portfolio-wide average -- a true-diversification check. |
| `account_growth` | Risk/return profile of your current holdings over a period (total return, CAGR, annualized vol, max drawdown, rough Sharpe), valuing today positions back through price history. Synthetic, not actual past account equity. |

### 0DTE execution & tax (v0.8.0)
| Tool | What it does |
|------|--------------|
| `spot_blend` | De-stales the gamma map: compares the delayed CBOE chain spot to a live SPY-implied SPX and flags whether spot has likely crossed the gamma flip or a wall since the snapshot. |
| `pcs_sizer` | Sizes an SPX put credit spread (ASD 0DTE PCS): short put nearest a target delta, long put a given width below, with net credit, max loss, breakeven, return-on-risk, and an approximate POP. |
| `event_risk_radar` | What can gap your book in the next N days: high-impact econ events plus holdings reporting earnings, merged into one timeline flagged by what you hold. |
| `estimated_tax` | Estimated tax set-aside on realized trading gains: YTD short/long-term options P&L x marginal federal + Georgia rates, with a quarterly figure. Trading gains only; not tax advice. |

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
