"""
Zakamulin Portfolio Backtest
Portfolio 1 — Best 5 Mid-Cap:  LSTR/A_SMSM, FICO/C_Mom12, EXP/A_SMSM, SAIA/C_Mom12, BCO/A_SMSM
Portfolio 2 — Broad 13:        All 10 mid-caps + SPY/QQQ/IWM, each slot with A_SMSM
                                (P1 assets keep their assigned strategy; remaining 8 use A_SMSM)
Equal-weight slots, monthly rebalance. Cash when signal=0. No leverage. 0.10% round-trip TC.
Compared against equal-weight buy-and-hold of same tickers (no signals, no TC).
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
ENV_PATH     = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\.env")
RESULTS_DIR  = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\results\portfolio")
START_DATE   = date(2018, 1, 1)
END_DATE     = date.today()
TC_ROUNDTRIP = 0.0010  # 0.10% per round-trip trade

# Zakamulin SMSM optimal weights (months 1-12, index 0 = most recent)
SMSM_WEIGHTS = [0.20] * 4 + [0.05] * 4 + [-0.10] * 4  # sums to 0.80

# Portfolio 1: 5 best mid-caps by Sharpe, each with their assigned strategy
P1_ASSETS = {
    "LSTR": "A_SMSM",
    "FICO": "C_Mom12",
    "EXP":  "A_SMSM",
    "SAIA": "C_Mom12",
    "BCO":  "A_SMSM",
}

# Portfolio 2: all 10 mid-caps + 3 ETFs = 13 total
# P1 slots keep their strategy; remaining 8 all use A_SMSM
P2_ASSETS = {
    **P1_ASSETS,
    "EXPO": "A_SMSM",
    "NDSN": "A_SMSM",
    "POOL": "A_SMSM",
    "RJF":  "A_SMSM",
    "WHR":  "A_SMSM",
    "SPY":  "A_SMSM",
    "QQQ":  "A_SMSM",
    "IWM":  "A_SMSM",
}

ALL_SYMBOLS = sorted(set(list(P1_ASSETS) + list(P2_ASSETS)))

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


print("Fetching monthly closes from Alpaca (IEX)...")
raw: dict[str, pd.Series] = {}
for sym in ALL_SYMBOLS:
    try:
        series = fetch_monthly_closes(sym)
        n = len(series)
        print(f"  {sym:6s}: {n} monthly bars")
        raw[sym] = series
    except Exception as e:
        print(f"  {sym}: ERROR — {e}")

missing = [s for s in ALL_SYMBOLS if s not in raw]
if missing:
    print(f"\nWARNING: missing data for {missing}; those slots will be excluded from portfolios.")

# Remove missing from portfolio maps
P1_ASSETS = {k: v for k, v in P1_ASSETS.items() if k in raw}
P2_ASSETS = {k: v for k, v in P2_ASSETS.items() if k in raw}

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


SIGNAL_FNS = {
    "A_SMSM": signal_smsm,
    "B_SMA10": signal_sma10,
    "C_Mom12": signal_mom12,
}

# === PORTFOLIO ENGINE =================================================

def build_portfolio(asset_strategy_map: dict, prices_raw: dict) -> tuple:
    """
    Builds equal-weight portfolio returns and B&H benchmark.

    Each slot has weight 1/N. Signal=0 → cash (0 return for that slot).
    TC applied as a fraction of the slot weight when position changes.

    Returns:
        port_ret   : pd.Series — monthly portfolio return (strategy)
        pos_df     : pd.DataFrame — 0/1 active signals per slot per month
        bh_ret     : pd.Series — equal-weight B&H monthly return
    """
    n_slots = len(asset_strategy_map)

    # Compute signals for all assets on their full history (for warm-up)
    sigs = {}
    rets = {}
    for sym, strat in asset_strategy_map.items():
        p = prices_raw[sym]
        sigs[sym] = SIGNAL_FNS[strat](p)
        rets[sym] = p.pct_change()

    # Align: dates where ALL assets have valid signals AND valid returns
    sig_df  = pd.DataFrame(sigs)
    ret_df  = pd.DataFrame(rets)

    common_idx = sig_df.dropna().index.intersection(ret_df.dropna().index)
    sig_df = sig_df.loc[common_idx]
    ret_df = ret_df.loc[common_idx]

    # After aligning, pct_change leaves first row as NaN → drop it
    # (ret at date t = (price[t]/price[t-1])-1; we need the prior price)
    ret_df = ret_df.dropna(how="any")
    sig_df = sig_df.loc[ret_df.index]

    # Execute signal at next-month close: position[t] = signal[t-1]
    pos_df = sig_df.shift(1).fillna(0)       # 0 or 1 per slot
    weight_df = pos_df / n_slots              # actual portfolio fraction

    # TC: half of round-trip applied when weight changes (entry or exit)
    trade_df = weight_df.diff().abs()
    trade_df.iloc[0] = 0.0                    # no TC on first bar
    tc_df = trade_df * (TC_ROUNDTRIP / 2)

    # Strategy portfolio return
    slot_ret_df = weight_df * ret_df - tc_df
    port_ret = slot_ret_df.sum(axis=1)

    # Equal-weight B&H (always invested 1/N in each asset, no TC)
    bh_ret = ret_df.mean(axis=1)             # arithmetic avg = equal-weight

    return port_ret, pos_df, bh_ret


def compute_metrics(port_ret: pd.Series, pos_df: pd.DataFrame, name: str) -> dict:
    equity = (1 + port_ret).cumprod()

    n_months = len(port_ret)
    n_years  = n_months / 12
    cagr     = equity.iloc[-1] ** (1 / n_years) - 1
    ann_vol  = port_ret.std() * np.sqrt(12)
    sharpe   = (port_ret.mean() * 12) / ann_vol if ann_vol > 0 else 0.0

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd   = drawdown.min()

    n_slots         = pos_df.shape[1]
    active_per_month = pos_df.sum(axis=1)
    pct_invested    = (active_per_month / n_slots).mean()
    avg_positions   = active_per_month.mean()

    return {
        "name":          name,
        "equity":        equity,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "pct_invested":  pct_invested,
        "avg_positions": avg_positions,
        "n_months":      n_months,
    }


def bh_metrics(bh_ret: pd.Series) -> dict:
    eq      = (1 + bh_ret).cumprod()
    n       = len(bh_ret)
    years   = n / 12
    cagr    = eq.iloc[-1] ** (1 / years) - 1
    vol     = bh_ret.std() * np.sqrt(12)
    sharpe  = (bh_ret.mean() * 12) / vol if vol > 0 else 0.0
    max_dd  = ((eq - eq.cummax()) / eq.cummax()).min()
    return {"equity": eq, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}


# === BUILD PORTFOLIOS =================================================

print(f"\nPortfolio 1 — Best 5 Mid-Cap ({len(P1_ASSETS)} assets): {list(P1_ASSETS)}")
p1_ret, p1_pos, p1_bh_ret = build_portfolio(P1_ASSETS, raw)
p1_m  = compute_metrics(p1_ret, p1_pos, "Best 5 Mid-Cap")
p1_bh = bh_metrics(p1_bh_ret)

print(f"Portfolio 2 — Broad 13 ({len(P2_ASSETS)} assets): {list(P2_ASSETS)}")
p2_ret, p2_pos, p2_bh_ret = build_portfolio(P2_ASSETS, raw)
p2_m  = compute_metrics(p2_ret, p2_pos, "Broad 13")
p2_bh = bh_metrics(p2_bh_ret)

# === PRINT RESULTS ====================================================

print("\n" + "=" * 80)
for m, bh, n_assets in [(p1_m, p1_bh, len(P1_ASSETS)), (p2_m, p2_bh, len(P2_ASSETS))]:
    print(f"\n{m['name']} ({n_assets} assets, {m['n_months']} months)")
    print(f"  {'Metric':<22} {'Strategy':>12}  {'B&H':>12}")
    print(f"  {'CAGR':<22} {m['cagr']*100:>11.2f}%  {bh['cagr']*100:>11.2f}%")
    print(f"  {'Sharpe':<22} {m['sharpe']:>12.3f}  {bh['sharpe']:>12.3f}")
    print(f"  {'Max Drawdown':<22} {m['max_dd']*100:>11.2f}%  {bh['max_dd']*100:>11.2f}%")
    print(f"  {'% Months Invested':<22} {m['pct_invested']*100:>11.1f}%  {'100.0':>12}%")
    print(f"  {'Avg Positions / Month':<22} {m['avg_positions']:>12.2f}  {float(n_assets):>12.1f}")
print("=" * 80)

# === PLOTS ============================================================

COLORS = {
    "strategy": "#1f77b4",
    "bh":       "#d62728",
}


def plot_portfolio(m: dict, bh: dict, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_yscale("log")

    eq    = m["equity"]
    bh_eq = bh["equity"].loc[eq.index]

    # Rebase both to 1.0 at shared start
    eq_norm    = eq    / eq.iloc[0]
    bh_eq_norm = bh_eq / bh_eq.iloc[0]

    ax.plot(eq_norm.index, eq_norm.values,
            label=f"Strategy  (CAGR={m['cagr']*100:.1f}%  Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']*100:.1f}%)",
            color=COLORS["strategy"], linewidth=2)
    ax.plot(bh_eq_norm.index, bh_eq_norm.values,
            label=f"EW B&H    (CAGR={bh['cagr']*100:.1f}%  Sharpe={bh['sharpe']:.2f}  MaxDD={bh['max_dd']*100:.1f}%)",
            color=COLORS["bh"], linewidth=1.5, linestyle="--")

    ax.set_title(f"{label} — Strategy vs Equal-Weight B&H (log scale)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (rebased to 1.0)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.name}")


plot_portfolio(p1_m, p1_bh, "Portfolio 1 — Best 5 Mid-Cap",
               RESULTS_DIR / "portfolio1_best5_equity.png")
plot_portfolio(p2_m, p2_bh, "Portfolio 2 — Broad 13",
               RESULTS_DIR / "portfolio2_broad13_equity.png")

# Combined overlay: both strategies on one chart for easy comparison
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_yscale("log")

for m, bh, color, label in [
    (p1_m, p1_bh, "#1f77b4", "P1 Best 5"),
    (p2_m, p2_bh, "#2ca02c", "P2 Broad 13"),
]:
    eq     = m["equity"]
    bh_eq  = bh["equity"].loc[eq.index]
    eq_n   = eq    / eq.iloc[0]
    bh_n   = bh_eq / bh_eq.iloc[0]
    ax.plot(eq_n.index, eq_n.values,
            label=f"{label} Strategy (Sharpe={m['sharpe']:.2f})",
            color=color, linewidth=2)
    ax.plot(bh_n.index, bh_n.values,
            label=f"{label} B&H",
            color=color, linewidth=1.2, linestyle="--", alpha=0.6)

ax.set_title("Portfolio Comparison — Strategy vs B&H (log scale)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Equity (rebased to 1.0)")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
combined_png = RESULTS_DIR / "portfolio_comparison.png"
fig.savefig(combined_png, dpi=150)
plt.close(fig)
print(f"Saved {combined_png.name}")

# === SUMMARY CSV ======================================================

summary_rows = []

for m, bh, n_assets, asset_map in [
    (p1_m, p1_bh, len(P1_ASSETS), P1_ASSETS),
    (p2_m, p2_bh, len(P2_ASSETS), P2_ASSETS),
]:
    tickers = "|".join(sorted(asset_map))
    strategies = "|".join(f"{k}/{v}" for k, v in sorted(asset_map.items()))

    summary_rows.append({
        "Portfolio":           m["name"],
        "Type":                "Strategy",
        "N_Assets":            n_assets,
        "Tickers":             tickers,
        "Strategies":          strategies,
        "CAGR_pct":            round(m["cagr"] * 100, 2),
        "Sharpe":              round(m["sharpe"], 3),
        "MaxDrawdown_pct":     round(m["max_dd"] * 100, 2),
        "PctMonthsInvested":   round(m["pct_invested"] * 100, 1),
        "AvgPositionsPerMonth":round(m["avg_positions"], 2),
        "N_Months":            m["n_months"],
    })
    summary_rows.append({
        "Portfolio":           m["name"],
        "Type":                "EW B&H Benchmark",
        "N_Assets":            n_assets,
        "Tickers":             tickers,
        "Strategies":          "always_long",
        "CAGR_pct":            round(bh["cagr"] * 100, 2),
        "Sharpe":              round(bh["sharpe"], 3),
        "MaxDrawdown_pct":     round(bh["max_dd"] * 100, 2),
        "PctMonthsInvested":   100.0,
        "AvgPositionsPerMonth":float(n_assets),
        "N_Months":            m["n_months"],
    })

summary_df = pd.DataFrame(summary_rows)
csv_path   = RESULTS_DIR / "summary.csv"
summary_df.to_csv(csv_path, index=False)

print("\n" + "=" * 80)
print("SUMMARY CSV")
print("=" * 80)
print(summary_df[["Portfolio","Type","CAGR_pct","Sharpe","MaxDrawdown_pct",
                   "PctMonthsInvested","AvgPositionsPerMonth"]].to_string(index=False))
print("=" * 80)
print(f"\nCSV  -> {csv_path}")
print(f"PNGs -> {RESULTS_DIR}")
