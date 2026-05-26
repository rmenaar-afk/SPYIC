# Vibe Trading Skills

A Claude Code / Cowork plugin bundling **57 quantitative-trading skills**, focused on US equities & ETFs, global / European markets, and crypto / DeFi. Covers market-data sources, technical & fundamental analysis, factor research, options, risk, and report generation.

> Source: pruned from [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) (`agent/src/skills/`). Original 74-skill set has been narrowed by removing China A-share / HK-specific skills and skills with Chinese-only frontmatter.

## What's inside (57 skills)

| Category | Count | Skills |
|---|---|---|
| Strategy | 16 | candlestick, cross-market-strategy, elliott-wave, event-driven, execution-model, harmonic, ichimoku, minute-analysis, ml-strategy, multi-factor, pair-trading, seasonal, smc, strategy-generate, technical-basic, volatility |
| Analysis | 13 | behavioral-finance, commodity-analysis, correlation-analysis, dividend-analysis, earnings-revision, factor-research, global-macro, macro-analysis, market-microstructure, performance-attribution, quant-statistics, risk-analysis, valuation-model |
| Tool | 9 | backtest-diagnose, doc-reader, geopolitical-risk, pine-script, report-generate, social-media-intelligence, trade-journal, vnpy-export, web-reader |
| Crypto | 7 | crypto-derivatives, defi-yield, liquidation-heatmap, onchain-analysis, perp-funding-basis, stablecoin-flow, token-unlock-treasury |
| Asset class | 5 | asset-allocation, hedging-strategy, options-advanced, options-payoff, options-strategy |
| Data source | 4 | ccxt, data-routing, okx-market, yfinance |
| Flow | 3 | edgar-sec-filings, fundamental-filter, us-etf-flow |

## What was pruned (17 skills, available upstream)

These were dropped from the plugin but remain in `Mr. Buffet/Vibe-Trading/agent/src/skills/` if you ever want them back. They were either tied to Chinese A-share / HK markets, or written with Chinese-only frontmatter:

`adr-hshare`, `akshare`, `ashare-pre-st-filter`, `chanlun`, `convertible-bond`, `corporate-events`, `credit-analysis`, `earnings-forecast`, `etf-analysis`, `financial-statement`, `fund-analysis`, `hk-connect-flow`, `regulatory-knowledge`, `sector-rotation`, `sentiment-analysis`, `shadow-account`, `tushare`

To restore one: copy the folder back from `../Vibe-Trading/agent/src/skills/<name>/` into `skills/<name>/`.

## Folder layout

```
vibe-trading-skills/
├── .claude-plugin/
│   ├── plugin.json          # plugin manifest
│   └── marketplace.json     # single-entry marketplace manifest
├── skills/                  # 57 skill folders, each with SKILL.md
│   ├── yfinance/
│   ├── factor-research/
│   ├── options-strategy/
│   └── ...
└── README.md
```

## Install

### As a plugin (Claude Code)

From the Claude Code CLI in any project:

```bash
/plugin marketplace add /Users/robinmenaar/Documents/Claude/Projects/Mr. Buffet/vibe-trading-skills
/plugin install vibe-trading-skills@mr-buffet-marketplace
```

### As a plugin (Cowork desktop)

1. Open **Settings → Plugins → Add marketplace**
2. Point at the absolute path of this folder, or push it to a private GitHub repo and use that URL
3. Install **vibe-trading-skills** from the listed plugins

### Direct read access (no install)

Even without installing, anything pointed at this folder can read the skills directly. The `SKILL.md` files use the standard frontmatter (`name`, `description`, `category`) so they're portable to any agent framework that supports the convention.

## Notes

- A few kept skills (`pine-script`, `trade-journal`) reference Chinese trading platforms / brokers (TongHuaShun, Futu) alongside Western ones — they're predominantly English and useful globally, so they stayed.
- Several skills assume the upstream Vibe-Trading project layout (e.g. references to `backtest/loaders/...`). They still work as guidance, but the file paths inside example code are upstream-relative.

## Attribution

Original work © HKUDS and the Vibe-Trading contributors. This plugin is a redistribution of a subset of the `agent/src/skills/` directory, reorganized into the Claude Code plugin format. Refer to the upstream repository for license terms.
