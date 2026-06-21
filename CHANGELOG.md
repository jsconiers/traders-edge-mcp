# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [0.8.0] - 2026-06-20

### Added — 14 tools (48 -> 62)

**Daily workflow**
- **`morning_brief`** — pre-open command center: regime + posture, key 0DTE levels (spot, expected
  move, gamma flip, call/put walls, max-pain), vol complex, high-impact econ events, holdings earnings
  within ~7 days, last session result, and the discipline reset. Pure composition of existing tools.
- **`eod_wrap`** — end-of-day wrap: realized vs target, discipline adherence (stop-at-target vs
  give-back), closing levels, and a snapshot logged to SQLite history.
- **`weekly_review`** — week realized P&L vs the `weekly_target` config: Mon-Fri breakdown, best/worst
  day, win rate, progress to goal.
- **`tilt_detector`** — scans a session trade sequence for tilt signatures: revenge sizing, rushing
  (shrinking entry gaps), intraday win-rate decay, and trading after a give-back from peak.

**Wheel & income**
- **`wheel_tracker`** — lifetime wheel scorecard for a symbol: net premium (calls + puts), contracts
  sold to open, buy-to-close cost, cycles, shares/avg cost, and effective basis after premium.
- **`covered_call_writer`** — fresh covered calls to write on a holding: OTM strikes near a target
  delta ranked by annualized yield, contracts covered, and earnings/ex-dividend-before-expiry flags.
- **`csp_finder`** — cash-secured puts to sell: OTM strikes near a target delta ranked by annualized
  yield on cash secured, with cash required per contract and an earnings flag.
- **`dividend_calendar`** — projected next ex-dividend dates for holdings (last ex-date + frequency
  estimated from yield/dps/price), with cadence, dividend/share, and yield.

**Risk analytics**
- **`correlation_matrix`** — daily-return correlation across holdings (pairwise matrix, per-name
  average, most/least correlated pairs, portfolio average) via Robinhood stock historicals.
- **`account_growth`** — risk/return profile of current holdings (total return, CAGR, annualized vol,
  max drawdown, rough Sharpe). NOTE: Robinhood removed the portfolio-equity-history endpoint, so this
  values *current* holdings back through price history — the current allocation profile, not actual
  past account equity (clearly labeled in output).

**0DTE execution & tax**
- **`spot_blend`** — de-stales the gamma map: compares the ~15-min-delayed CBOE chain spot to a live
  SPY-implied SPX (SPX is not quoted on Robinhood) and flags gamma-flip/wall crossings since the
  snapshot. SPYx10 carries a ~20-40pt dividend basis to SPX; a `basis` param calibrates it.
- **`pcs_sizer`** — sizes an SPX put credit spread (ASD 0DTE PCS): short put nearest a target delta,
  long put a width below, with net credit, max loss, breakeven, return-on-risk, and approximate POP.
- **`event_risk_radar`** — high-impact econ events + holdings earnings merged into one forward
  timeline flagged by what you hold.
- **`estimated_tax`** — estimated set-aside on realized trading gains (YTD short/long-term options P&L
  x marginal federal + Georgia rates) with a quarterly figure. Trading gains only; not tax advice.

## [0.7.0] - 2026-06-20

### Added
- **User config file (`config.json`) + `trading_config` tool** — goals/discipline settings
  (`daily_target`, `giveback_frac`, `rapid_reentry_secs`, `late_session_et`, `roll_delta`, `roll_dte`,
  …) are read from a JSON file next to the server with live reload (no restart). Precedence: env var >
  config.json > built-in default. `trading_config` shows / sets / resets keys.
- **`discipline_backtest`** — replays your historical fills through the stop-at-target rule and reports
  actual vs stop-at-target P&L, the after-target leak on losing days, win rate, expectancy, profit
  factor, best/worst day, an equity curve, and by-day-of-week / by-hour breakdowns.
- **`tax_summary`** — year-to-date realized options P&L (short vs long term, by month, gross
  gains/losses) plus identical-contract wash-sale candidates. CPA hand-off; not tax advice.
- **`snapshot_log` / `snapshot_history`** — log the current 0DTE state (spot, GEX, gamma flip,
  call/put walls, max-pain, expected move, VIX/VIX1D, regime) to local SQLite and read back the day's
  intraday drift (GEX migration). DB at `~/.trading/traders_edge.db` (override `TE_DB_PATH`).
- **`roll_candidates`** — roll-up-and-out suggestions for a covered call: candidate strikes/expiries
  with mark, delta, net credit vs closing the current call, and annualized yield. 48 tools total.

### Changed
- Discipline tools (`daily_target`, `daily_pnl_curve`, `daily_review`, `should_i_trade`) now resolve
  the daily target and the give-back / rapid-reentry / late-session thresholds from the live config.
- `_round_trips` now carries `expiry` and `option_id` per trip (contract identity for wash-sale logic).

## [0.6.0] - 2026-06-20

### Added
- **`covered_call_manager`** - scans Robinhood short-call positions: DTE, assignment probability
  (delta), premium captured vs extrinsic remaining, annualized yield, share-coverage check,
  earnings-before-expiry risk flag, and roll/management signals (params: `roll_delta`, `roll_dte`).
- **`earnings_calendar`** - next single-name earnings dates for your holdings (or a symbol list),
  sorted by proximity, with BMO/AMC session, days away, and within-window flag; ETFs/funds listed
  separately. Sourced from Robinhood earnings data (no extra API key).
- **`regime_classifier`** - one composite risk-on/constructive/neutral/caution/risk-off read folding
  VIX level + VIX term structure + NFCI + HY OAS + 2s10s curve + Sahm rule, with a 0DTE posture.
  42 tools total.

### Changed
- Robinhood option-position normalizer now also carries `avgPrice`, `mult`, and `mark` (enables the
  covered-call premium/extrinsic math).

## [0.5.0] - 2026-06-19

### Added
- **Discipline / anti-overtrading layer (3 tools).** `daily_pnl_curve` (realized-P&L curve from
  Robinhood fills with target-cross + give-back), `daily_review` (win rate, expectancy, profit factor,
  P&L by hour, before-vs-after-target split), and `should_i_trade` (real-time GO/CAUTION/STOP gate).
  Realized P&L is netted from `net_amount` (fees included) with FIFO open->close round-trip matching.
- **0DTE decision support (3 tools).** `expected_move` (ATM-straddle range + sigma levels),
  `strike_probabilities` (per-strike prob-ITM and prob-of-touch), and `daily_game_plan` (expected move
  + gamma flip/walls + max-pain + high-OI pins -> support/resistance). 39 tools total.

### Notes
- Discipline tools assume manual closes (the 0DTE scalping style); a recon note flags any expired or
  still-open positions where net cash flow diverges from round-trip realized P&L.
- Tunables: `TE_GIVEBACK_FRAC` (0.40), `TE_RAPID_REENTRY_SECS` (90), `TE_LATE_SESSION_ET` (15:45).

## [0.4.0] - 2026-06-19

### Added
- **E\*TRADE as a live aggregator source.** Auto-pulls E\*TRADE stock + option positions via the cached
  `pyetrade` OAuth session shared with the etrade MCP. New `etrade_positions` tool (33 tools total);
  toggle with `include_etrade`.
- **Editable beta map for SPX-weighting.** Auto-pulled equities are now beta-weighted using a built-in
  map (ICE, NVDA, SCHD, VOO, ...), overridable via `BETA_OVERRIDES` or `BETA_MAP_FILE`. `net_greeks` /
  `risk_summary` now report a meaningful `netDelta_betaWeighted$` distinct from raw delta.

### Notes
- Requires `pyetrade`. E\*TRADE access tokens expire nightly; re-authorize via the etrade MCP if the
  cached token is stale. E\*TRADE equity-option Greeks aren't fetched yet (SPX/SPXW priced via CBOE).

## [0.3.0] - 2026-06-19

### Added
- **Robinhood as a live aggregator source.** The risk/Greeks aggregator now auto-pulls Robinhood stock
  holdings (`build_holdings`) and option legs — with broker-provided delta/gamma/theta/vega/IV — via the
  cached `robin_stocks` session, alongside Alpaca. New `robinhood_positions` tool (32 tools total). Each
  source is toggleable per call (`include_alpaca` / `include_robinhood` / `include_file`).
- `_position_risk` now uses broker-supplied option Greeks when present (`greeksSource: "broker"`).

### Notes
- Requires `robin_stocks`; the session pickle (`~/.robinhood/`) is shared with the robinhood-local
  server and refreshes every 7 days (device-approval prompt on first use after expiry).

## [0.2.0] - 2026-06-19

Expanded from a Tier-1 options cockpit into a full trading-context server (31 tools).

### Added
- **Tier 2 — FRED macro (key-less):** `fed_funds`, `yield_curve`, `inflation`, `labor_market`,
  `growth`, `financial_conditions`, `recession_indicators`, `series`, `latest`, `series_search`,
  `fred_status`. Sourced from the FRED fredgraph CSV endpoint (no key); optional `FRED_API_KEY`
  enables catalog search.
- **Cross-broker risk / Greeks aggregator:** `net_greeks`, `risk_summary`, `concentration`,
  `scenario_shock`, `daily_target`, `alpaca_positions`, `load_positions`, `risk_status`. Aggregates
  live Alpaca positions with a broker-agnostic positions file; SPX/SPXW options priced off CBOE,
  equities beta-weighted. Includes a daily-target / post-target-overtrading discipline check.

### Notes
- FRED's CDN requires a plain User-Agent; the FRED client uses one (CBOE still uses a browser UA).

## [0.1.0] - 2026-06-19

Initial release. 12 tools across four modules, sourced from free, key-less data
(CBOE delayed quotes + TreasuryDirect), with an optional FMP/Finnhub key for a live
economic calendar.

### Added
- **Chain & Greeks:** `options_chain`, `option_quote`, `expirations`.
- **Dealer positioning:** `gamma_exposure` (GEX + zero-gamma flip), `gamma_walls`
  (call/put walls + max-pain), `zero_dte_exposure` (0DTE dashboard with pin & expected move),
  `dealer_exposure` (DEX / vanna / charm).
- **Vol complex:** `vix_complex`, `vix_term_structure`.
- **Event clock:** `economic_calendar` (rule-based + curated + live Treasury auctions),
  `next_event`.
- `traders_edge_status` health check.
- Vectorized Black–Scholes Greeks engine (numpy), ET-aware time-to-expiry, and offline test suite.
