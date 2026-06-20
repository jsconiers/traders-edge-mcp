# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

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
