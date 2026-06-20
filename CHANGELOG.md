# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

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
