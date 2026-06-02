"""
Zakamulin & Giner (SSRN 4282126) — Optimal Trend-Following With Transaction Costs
Three strategies: SMSM Optimal (A), 10-Month SMA (B), 12-Month Momentum (C)
Tested on 10 diverse mid-cap stocks | Monthly rebalance | 0.10% round-trip cost on signal change
"""

import os, warnings
from pathlib import Path
from datetime import datetime, date, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === CONFIG ===========================================================
ENV_PATH      = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\.env")
RESULTS_DIR   = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\results\midcap")
START_DATE    = date(2018, 1, 1)
END_DATE      = date.today()
TC_ROUNDTRIP  = 0.0010   # 0.10% per round-trip trade
MIN_MONTHS    = 36       # drop any ticker below this threshold

# Primary candidates — 10 mid-caps (~$2B–$15B), diverse sectors, listed since ≥2018
# Fallbacks substituted automatically if a primary has < MIN_MONTHS bars
ASSETS_PRIMARY = [
    "POOL",   # Consumer Discretionary  — pool products distributor
    "LSTR",   # Industrials             — freight brokerage
    "SAIA",   # Industrials             — LTL trucking
    "NDSN",   # Industrials             — dispensing equipment
    "RJF",    # Financials              — Raymond James Financial
    "WHR",    # Consumer Discretionary  — Whirlpool appliances
    "FICO",   # Technology              — Fair Isaac (FICO scores)
    "EXP",    # Materials               — Eagle Materials (cement/wallboard)
    "EXPO",   # Industrials             — Exponent engineering consulting
    "BCO",    # Industrials             — Brink's security services
]

# Fallbacks (same rough market-cap / sector diversity)
ASSETS_FALLBACK = [
    "SFM",    # Consumer Staples  — Sprouts Farmers Market
    "EPRT",   # Real Estate       — Essential Properties Realty
    "IBP",    # Industrials       — Installed Building Products
    "ICUI",   # Health Care       — ICU Medical
    "CSWI",   # Industrials       — CSW Industrials
]

# Zakamulin SMSM optimal weights (months 1–12, index 0 = most recent)
SMSM_WEIGHTS  = [0.20]*4 + [0.05]*4 + [-0.10]*4   # sums to 0.80

# === SETUP ============================================================
load_dotenv(ENV_PATH)
warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

# === DATA FETCH =======================================================
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest
from alpaca.data.timeframe  import TimeFrame
from alpaca.data.enums      import DataFeed, Adjustment

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def fetch_monthly_closes(symbol: str) -> pd.Series:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(START_DATE.year, START_DATE.month, START_DATE.day, tzinfo=timezone.utc),
        end=datetime(END_DATE.year, END_DATE.month, END_DATE.day, tzinfo=timezone.utc),
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    bars = client.get_stock_bars(req).df
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
    bars = bars.set_index("timestamp")["close"]
    monthly = bars.resample("ME").last().dropna()
    monthly.name = symbol
    return monthly


print("Fetching daily bars from Alpaca (IEX)…")

raw: dict[str, pd.Series] = {}
fallback_pool = list(ASSETS_FALLBACK)

for sym in ASSETS_PRIMARY:
    try:
        series = fetch_monthly_closes(sym)
        n = len(series)
        print(f"  {sym:6s}: {n} monthly bars", end="")
        if n < MIN_MONTHS:
            print(f"  *** INSUFFICIENT (< {MIN_MONTHS}) — swapping ***")
            while fallback_pool:
                fb = fallback_pool.pop(0)
                try:
                    fb_series = fetch_monthly_closes(fb)
                    fb_n = len(fb_series)
                    print(f"  {fb:6s}: {fb_n} monthly bars (fallback for {sym})", end="")
                    if fb_n >= MIN_MONTHS:
                        print("  OK")
                        raw[fb] = fb_series
                        break
                    else:
                        print(f"  *** also insufficient, skipping ***")
                except Exception as e:
                    print(f"  {fb}: fetch error ({e}), skipping")
            else:
                print(f"  No more fallbacks; skipping {sym}.")
        else:
            print("  OK")
            raw[sym] = series
    except Exception as e:
        print(f"  {sym}: fetch error ({e}), trying fallback…")
        while fallback_pool:
            fb = fallback_pool.pop(0)
            try:
                fb_series = fetch_monthly_closes(fb)
                fb_n = len(fb_series)
                print(f"  {fb:6s}: {fb_n} monthly bars (fallback for {sym})", end="")
                if fb_n >= MIN_MONTHS:
                    print("  OK")
                    raw[fb] = fb_series
                    break
                else:
                    print(f"  *** insufficient, skipping ***")
            except Exception as e2:
                print(f"  {fb}: fetch error ({e2}), skipping")

ASSETS = sorted(raw.keys())
print(f"\nFinal asset universe ({len(ASSETS)}): {ASSETS}")


# === SIGNAL FUNCTIONS =================================================

def signal_smsm(prices: pd.Series) -> pd.Series:
    """Strategy A: Zakamulin SMSM optimal weighted momentum."""
    log_ret = np.log(prices / prices.shift(1))
    signals = pd.Series(index=prices.index, dtype=float)
    for i in range(12, len(prices)):
        w_sum = sum(SMSM_WEIGHTS[j] * log_ret.iloc[i - j] for j in range(12))
        signals.iloc[i] = 1.0 if w_sum > 0 else 0.0
    return signals


def signal_sma10(prices: pd.Series) -> pd.Series:
    """Strategy B: 10-month SMA crossover."""
    sma = prices.rolling(10).mean()
    sig = (prices > sma).astype(float)
    sig[:10] = np.nan
    return sig


def signal_mom12(prices: pd.Series) -> pd.Series:
    """Strategy C: 12-month price momentum."""
    sig = (prices > prices.shift(12)).astype(float)
    sig[:12] = np.nan
    return sig


# === BACKTEST ENGINE ==================================================

def run_backtest(prices: pd.Series, signals: pd.Series, name: str) -> dict:
    """Apply signals to prices with transaction costs; return metrics + equity."""
    df = pd.DataFrame({"price": prices, "signal": signals}).dropna()

    monthly_ret = df["price"].pct_change().fillna(0)
    position    = df["signal"].shift(1).fillna(0)

    trade   = position.diff().abs()
    trade.iloc[0] = 0.0
    tc_cost = trade * (TC_ROUNDTRIP / 2)

    strat_ret = position * monthly_ret - tc_cost
    equity    = (1 + strat_ret).cumprod()

    n_months = len(strat_ret)
    n_years  = n_months / 12
    cagr     = equity.iloc[-1] ** (1 / n_years) - 1
    ann_vol  = strat_ret.std() * np.sqrt(12)
    sharpe   = (strat_ret.mean() * 12) / ann_vol if ann_vol > 0 else 0
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd   = drawdown.min()

    n_trades = int((position.diff().abs() > 0).sum())
    if n_trades > 0:
        long_months = monthly_ret[position > 0]
        win_rate    = (long_months > 0).mean() if len(long_months) > 0 else np.nan
    else:
        win_rate = np.nan

    return {
        "name":     name,
        "equity":   equity,
        "cagr":     cagr,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
    }


def buy_and_hold(prices: pd.Series) -> pd.Series:
    ret = prices.pct_change().fillna(0)
    return (1 + ret).cumprod()


# === RUN ALL ==========================================================

all_metrics = []

for sym in ASSETS:
    print(f"\n--- {sym} ---")
    prices = raw[sym]

    results = {
        "A_SMSM":  run_backtest(prices, signal_smsm(prices),  "A_SMSM"),
        "B_SMA10": run_backtest(prices, signal_sma10(prices), "B_SMA10"),
        "C_Mom12": run_backtest(prices, signal_mom12(prices), "C_Mom12"),
    }

    bh = buy_and_hold(prices)

    for key, r in results.items():
        all_metrics.append({
            "Asset":       sym,
            "Strategy":    r["name"],
            "CAGR":        round(r["cagr"] * 100, 2),
            "Sharpe":      round(r["sharpe"], 3),
            "MaxDrawdown": round(r["max_dd"] * 100, 2),
            "NumTrades":   r["n_trades"],
            "WinRate":     round(r["win_rate"] * 100, 2) if not np.isnan(r["win_rate"]) else np.nan,
        })

    # equity-curve plot (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_yscale("log")

    colors = {"A_SMSM": "#1f77b4", "B_SMA10": "#ff7f0e", "C_Mom12": "#2ca02c"}
    for key, r in results.items():
        eq = r["equity"]
        ax.plot(eq.index, eq.values, label=r["name"], color=colors[key], linewidth=1.5)

    common_start = results["A_SMSM"]["equity"].index[0]
    bh_aligned   = bh[bh.index >= common_start]
    bh_aligned   = bh_aligned / bh_aligned.iloc[0]
    ax.plot(bh_aligned.index, bh_aligned.values, label="Buy & Hold",
            color="#d62728", linewidth=1.5, linestyle="--")

    ax.set_title(f"{sym} — Zakamulin Trend-Following vs Buy & Hold (log scale)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (starts at 1.0)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_png = RESULTS_DIR / f"{sym}_equity_curves.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_png.name}")


# === SAVE & PRINT SUMMARY =============================================

summary_df = pd.DataFrame(all_metrics)
csv_path   = RESULTS_DIR / "summary.csv"
summary_df.to_csv(csv_path, index=False)

print("\n" + "=" * 80)
print("FULL SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))

# --- Sharpe ranking ---
print("\n" + "=" * 80)
print("SHARPE RATIO RANKING (best to worst)")
print("=" * 80)
ranking = (
    summary_df[["Asset", "Strategy", "Sharpe", "CAGR", "MaxDrawdown"]]
    .sort_values("Sharpe", ascending=False)
    .reset_index(drop=True)
)
ranking.index += 1
print(ranking.to_string())

# Best strategy per ticker
print("\n" + "=" * 80)
print("BEST STRATEGY PER TICKER (by Sharpe)")
print("=" * 80)
best_per_ticker = (
    summary_df.loc[summary_df.groupby("Asset")["Sharpe"].idxmax(),
                   ["Asset", "Strategy", "Sharpe", "CAGR", "MaxDrawdown"]]
    .sort_values("Sharpe", ascending=False)
    .reset_index(drop=True)
)
best_per_ticker.index += 1
print(best_per_ticker.to_string())

print("\n" + "=" * 80)
print(f"CSV  -> {csv_path}")
print(f"PNGs -> {RESULTS_DIR}")
