---
name: yfinance
description: yfinance global market data interface — retrieve OHLCV, financials, fundamentals, insider transactions, institutional holdings, and run live equity screeners for US stocks, HK stocks, ETFs, and indices via Yahoo Finance. Includes EquityQuery screener, Sector/Industry data, Search, and WebSocket live streaming. Free, no API key required. Source: https://github.com/ranaroussi/yfinance
category: data-source
---
# yfinance

## Overview

yfinance is an open-source Python wrapper for Yahoo Finance, providing global market data (US stocks, HK stocks, ETFs, indices) including historical and real-time quotes. **Completely free, no registration or API key required.**

Source repo (local copy in workspace): `Mr. Buffet/yfinance/`
Install: `pip install yfinance`

The project has a built-in yfinance DataLoader (`backtest/loaders/yfinance_loader.py`). When backtesting, set `source: "yfinance"` or `source: "auto"` to invoke it automatically.

## Main Components

| Component | Purpose |
|-----------|---------|
| `yf.Ticker` | Single stock — price, info, financials, options |
| `yf.Tickers` | Multiple tickers in one call |
| `yf.download` | Bulk OHLCV download |
| `yf.screen` + `EquityQuery` | **Live equity screener** — filter by PE, PB, sector, region, etc. |
| `yf.Sector` / `yf.Industry` | Sector and industry metadata + top companies |
| `yf.Search` | Quote and news search |
| `yf.Market` | Market status and summary |
| `yf.WebSocket` | Live streaming price data |

## Quick Start

```bash
pip install yfinance pandas
```

```python
import yfinance as yf

# Apple daily bars for the past year
df = yf.download("AAPL", start="2025-01-01", end="2026-01-01", progress=False)
print(df.head())

# Tencent (HK-listed)
df = yf.download("0700.HK", start="2025-01-01", end="2026-01-01", progress=False)
print(df.head())
```

## Ticker Format Conversion

The project uses a unified ticker format. The DataLoader automatically converts to yfinance format:

| Project Format | yfinance Format | Market |
|---------------|----------------|--------|
| `AAPL.US` | `AAPL` | US stock |
| `MSFT.US` | `MSFT` | US stock |
| `700.HK` | `0700.HK` | HK stock |
| `9988.HK` | `9988.HK` | HK stock |
| `SPY.US` | `SPY` | US ETF |

**Rules:**
- US stocks: strip the `.US` suffix → use the raw ticker
- HK stocks: keep `.HK`, pad the number to 4 digits (`700` → `0700`)

## Supported Data Types

### 1. Historical OHLCV

```python
import yfinance as yf
import pandas as pd

# Single stock
df = yf.download("AAPL", start="2025-01-01", end="2026-01-01", progress=False)

# Batch download
df = yf.download(["AAPL", "MSFT", "GOOGL"], start="2025-01-01", end="2026-01-01", progress=False)

# Specific interval
df = yf.download("AAPL", start="2026-03-01", end="2026-03-30",
                 interval="1h", progress=False)  # 1m/5m/15m/30m/1h/1d/1wk/1mo
```

**Supported intervals:**
- Minute-level: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`
- Hourly: `1h`
- Daily and above: `1d`, `5d`, `1wk`, `1mo`, `3mo`

**Minute data limits:**
- `1m`: up to 7 days of history
- `2m/5m/15m/30m/60m/90m`: up to 60 days
- `1h`: up to 730 days
- `1d` and above: unlimited

### 2. Company Info

```python
ticker = yf.Ticker("AAPL")

info = ticker.info
print(f"Company: {info.get('longName')}")
print(f"Industry: {info.get('industry')}")
print(f"Market cap: {info.get('marketCap')}")
print(f"PE: {info.get('trailingPE')}")
print(f"EPS: {info.get('trailingEps')}")
print(f"Dividend yield: {info.get('dividendYield')}")
```

### 3. Financial Statements

```python
ticker = yf.Ticker("AAPL")

# Income statement (annual)
income = ticker.financials
# Income statement (quarterly)
income_q = ticker.quarterly_financials

# Balance sheet
balance = ticker.balance_sheet

# Cash flow statement
cashflow = ticker.cashflow

# Earnings data
earnings = ticker.earnings
```

### 4. Dividends and Splits

```python
ticker = yf.Ticker("AAPL")

# Dividend history
dividends = ticker.dividends

# Stock split history
splits = ticker.splits

# All corporate actions
actions = ticker.actions
```

### 5. Institutional Holdings

```python
ticker = yf.Ticker("AAPL")

# Institutional holders
holders = ticker.institutional_holders

# Major holders summary
major = ticker.major_holders

# Insider transactions
insider = ticker.insider_transactions
```

### 6. Indices and ETFs

```python
# Major indices
sp500 = yf.download("^GSPC", start="2025-01-01", end="2026-01-01", progress=False)  # S&P 500
nasdaq = yf.download("^IXIC", start="2025-01-01", end="2026-01-01", progress=False)  # NASDAQ
hsi = yf.download("^HSI", start="2025-01-01", end="2026-01-01", progress=False)      # Hang Seng Index

# ETFs
spy = yf.download("SPY", start="2025-01-01", end="2026-01-01", progress=False)
qqq = yf.download("QQQ", start="2025-01-01", end="2026-01-01", progress=False)
```

### 7. FX Rates

```python
# Currency pairs
usdcny = yf.download("CNY=X", start="2025-01-01", end="2026-01-01", progress=False)
usdhkd = yf.download("HKD=X", start="2025-01-01", end="2026-01-01", progress=False)
eurusd = yf.download("EURUSD=X", start="2025-01-01", end="2026-01-01", progress=False)
```

## Popular Ticker Reference

### US Stocks

| Ticker | Company |
|--------|---------|
| AAPL | Apple |
| MSFT | Microsoft |
| GOOGL | Alphabet (Google) |
| AMZN | Amazon |
| NVDA | NVIDIA |
| META | Meta Platforms |
| TSLA | Tesla |
| BRK-B | Berkshire Hathaway |

### HK Stocks

| Project Format | yfinance Format | Company |
|---------------|----------------|---------|
| 700.HK | 0700.HK | Tencent |
| 9988.HK | 9988.HK | Alibaba |
| 9618.HK | 9618.HK | JD.com |
| 3690.HK | 3690.HK | Meituan |
| 1810.HK | 1810.HK | Xiaomi |
| 2318.HK | 2318.HK | Ping An |

### Major Indices

| Ticker | Index |
|--------|-------|
| ^GSPC | S&P 500 |
| ^DJI | Dow Jones Industrial Average |
| ^IXIC | NASDAQ Composite |
| ^HSI | Hang Seng Index |
| ^N225 | Nikkei 225 |
| ^FTSE | FTSE 100 |

### Sector ETFs

| Ticker | Sector |
|--------|--------|
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Healthcare |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLI | Industrials |
| XLU | Utilities |

## Backtest Usage

### config.json Example

```json
{
  "source": "yfinance",
  "codes": ["AAPL.US", "MSFT.US"],
  "start_date": "2020-01-01",
  "end_date": "2026-03-30",
  "initial_cash": 1000000,
  "commission": 0.001,
  "extra_fields": null
}
```

### Cross-Market Auto Mode

```json
{
  "source": "auto",
  "codes": ["000001.SZ", "AAPL.US", "700.HK", "BTC-USDT"],
  "start_date": "2024-01-01",
  "end_date": "2026-03-30",
  "initial_cash": 1000000,
  "commission": 0.001,
  "extra_fields": null
}
```

`source: "auto"` routes automatically by ticker format: A-shares → tushare, HK/US stocks → yfinance, crypto → OKX.

---

## 8. Equity Screener (EquityQuery)

This is the most powerful feature for trade-idea generation. Build custom filters using `EquityQuery` and run them against the full Yahoo Finance universe.

### Predefined Screeners (ready to use)

```python
import yfinance as yf

# Run a predefined screen — no query building needed
result = yf.screen("undervalued_large_caps")
for stock in result['quotes']:
    print(stock['symbol'], stock.get('trailingPE'), stock.get('regularMarketPrice'))
```

**Available predefined queries:**

| Query Name | What it finds |
|------------|--------------|
| `undervalued_large_caps` | PE 0–20, PEG < 1, market cap $10B–$100B, NYSE/NMS |
| `undervalued_growth_stocks` | PE 0–20, PEG < 1, EPS growth > 25%, NYSE/NMS |
| `aggressive_small_caps` | Small caps sorted by volume, EPS growth < 15% |
| `day_gainers` | US stocks up > 3% today, market cap > $2B |
| `day_losers` | US stocks down > 2.5% today, market cap > $2B |
| `most_actives` | Highest volume US stocks, market cap > $2B |
| `most_shorted_stocks` | Highest short interest % of float |
| `growth_technology_stocks` | Tech sector, revenue growth > 25%, EPS growth > 25% |
| `small_cap_gainers` | Market cap < $2B, NYSE/NMS |
| `top_etfs_us` | US ETFs rated 4–5 stars by performance |
| `top_performing_etfs` | US ETFs rated 4–5 stars, sorted by lowest expense ratio |

### Custom Screener with EquityQuery

```python
import yfinance as yf
from yfinance import EquityQuery

# Example: US large-cap financials with low PE
q = EquityQuery('and', [
    EquityQuery('eq', ['region', 'us']),
    EquityQuery('eq', ['sector', 'Financial Services']),
    EquityQuery('btwn', ['peratio.lasttwelvemonths', 1, 15]),
    EquityQuery('gte', ['intradaymarketcap', 10_000_000_000]),
])
result = yf.screen(q, sortField='peratio.lasttwelvemonths', sortAsc=True, size=25)
for s in result['quotes']:
    print(s['symbol'], s.get('trailingPE'), s.get('regularMarketPrice'))
```

### Key EquityQuery Fields

| Field | Type | Example |
|-------|------|---------|
| `region` | eq | `'us'` |
| `sector` | eq | `'Technology'`, `'Financial Services'`, `'Healthcare'`, `'Energy'`, `'Industrials'` |
| `exchange` | is-in | `'NMS'` (Nasdaq), `'NYQ'` (NYSE) |
| `peratio.lasttwelvemonths` | btwn/gt/lt | PE ratio (trailing) |
| `pegratio_5y` | lt | PEG ratio (5-year) |
| `epsgrowth.lasttwelvemonths` | gte | EPS growth % TTM |
| `quarterlyrevenuegrowth.quarterly` | gte | Revenue growth % QoQ |
| `intradaymarketcap` | btwn/gte | Market cap in USD |
| `intradayprice` | gte/lte | Stock price |
| `percentchange` | gt/lt | Day change % |
| `dayvolume` | gt | Today's volume |
| `avgdailyvol3m` | gt | 3-month avg daily volume |
| `short_percentage_of_shares_outstanding.value` | gt | Short interest % |

### Operators

| Operator | Meaning |
|----------|---------|
| `eq` | Equals |
| `is-in` | One of a list |
| `btwn` | Between two values |
| `gt` / `gte` | Greater than / greater than or equal |
| `lt` / `lte` | Less than / less than or equal |
| `and` / `or` | Combine multiple EquityQuery conditions |

### Screener Result Shape

```python
result = yf.screen("undervalued_large_caps")
# result keys: 'start', 'count', 'total', 'quotes', 'predefinedScr', 'versionId'
quotes = result['quotes']
# Each quote dict contains: symbol, shortName, regularMarketPrice, trailingPE,
# forwardPE, priceToBook, marketCap, sector, industry, dividendYield, etc.
```

---

## 9. Sector & Industry Data

```python
import yfinance as yf

# Get sector overview
tech = yf.Sector('technology')
print(tech.overview)          # sector stats
print(tech.top_companies)     # DataFrame of largest companies
print(tech.top_etfs)          # ETFs tracking this sector
print(tech.industries)        # industries within this sector

# Get industry detail
semis = yf.Industry('semiconductors')
print(semis.overview)
print(semis.top_companies)
```

**Valid sector keys:** `technology`, `financial-services`, `healthcare`, `energy`, `industrials`, `consumer-cyclical`, `consumer-defensive`, `real-estate`, `utilities`, `communication-services`, `basic-materials`

---

## 10. Search (Quotes & News)

```python
import yfinance as yf

s = yf.Search("Apple", max_results=5, news_count=5)
print(s.quotes)   # list of matching tickers with basic data
print(s.news)     # list of recent news articles
```

---

## 11. Live Streaming (WebSocket)

```python
import yfinance as yf

# Stream real-time price updates
ws = yf.WebSocket(tickers=["AAPL", "MSFT", "SPY"], timeout=30)

def on_message(data):
    print(data['id'], data.get('price'), data.get('changePercent'))

ws.start(callback=on_message)

# Async version also available
ws_async = yf.AsyncWebSocket(tickers=["AAPL"], timeout=30)
```

---

## 12. Market Status

```python
import yfinance as yf

market = yf.Market("us_market")
print(market.status)       # 'OPEN' / 'CLOSED' / 'PRE' / 'POST'
print(market.summary)      # index-level summary data
```

---

## Trade-Idea Workflow (Mr. Buffet)

The recommended pattern for scanning for trade ideas:

```python
import yfinance as yf
from yfinance import EquityQuery

# Step 1: Screen for candidates
q = EquityQuery('and', [
    EquityQuery('eq', ['region', 'us']),
    EquityQuery('btwn', ['peratio.lasttwelvemonths', 1, 18]),
    EquityQuery('gte', ['intradaymarketcap', 5_000_000_000]),
    EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
])
candidates = yf.screen(q, sortField='peratio.lasttwelvemonths', sortAsc=True, size=50)

# Step 2: Pull fundamentals for top candidates
for stock in candidates['quotes'][:10]:
    sym = stock['symbol']
    t = yf.Ticker(sym)
    info = t.info
    print(f"{sym}: PE={info.get('trailingPE')}, PB={info.get('priceToBook')}, "
          f"ROE={info.get('returnOnEquity')}, FwdPE={info.get('forwardPE')}")

# Step 3: Pull financials for deep-dive
t = yf.Ticker("BAC")
print(t.financials)      # income statement
print(t.balance_sheet)   # balance sheet
print(t.cashflow)        # cash flow
```

---

## Notes

- **Free, no API key**: yfinance scrapes Yahoo Finance public data — no registration needed
- **Rate limits**: high-frequency requests may trigger temporary Yahoo bans — prefer batch downloads and screener over per-ticker loops
- **Screener limits**: Yahoo caps query results at 250 per call
- **Sandbox restriction**: The Cowork Linux sandbox blocks outbound Yahoo Finance connections (proxy 403). Run yfinance scripts locally on your Mac, not in the sandbox shell.
- **Minute data range**: limited by Yahoo Finance (1m = 7 days, 5m/15m = 60 days, 1h = 730 days)
- **HK tickers**: Yahoo Finance uses 4-digit numbers + `.HK`; pad with leading zeros where needed
- **Adjustment**: `auto_adjust=True` (default) returns forward-adjusted prices; the project loader uses `auto_adjust=False`
- **Timezone**: returned data includes timezone info; the DataLoader strips it automatically
- **extra_fields not supported in backtest loader**: PE/PB require separate `yf.Ticker().info` calls
- **Comparison with Tushare**: Tushare covers deep A-share data; yfinance covers global markets with screener capability
