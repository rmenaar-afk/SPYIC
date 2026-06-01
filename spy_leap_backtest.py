"""
SPY LEAP Call Options Backtest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry  : SPY ≥10% below 252-day high for ≥5 consecutive days → enter at
         next day's open, one position at a time (no pyramiding)
Exit   : First of → 100% gain (2× entry premium, daily BS mark) or expiry
Pricing: Black-Scholes with 30-day realized HV + 20% IV premium on entry
Grid   : delta ∈ {0.50, 0.60, 0.70, 0.80} × DTE ∈ {180, 270, 365, 540}
Capital: $2,000 start + $500/month; buy max whole contracts per signal
Bench  : SPY buy-and-hold, identical capital schedule, fractional shares
Data   : Alpaca IEX adjusted daily bars 2010-01-01 → today
"""

import os
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import brentq
from scipy.stats import norm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKER    = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
ENV_PATH  = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\.env")
OUT_DIR   = Path(rf"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\results\{TICKER.lower()}_leap")
START     = date(2010, 1, 1)
END       = date.today()

INIT_CAP  = 2_000.0
MONTHLY   = 500.0
RISK_FREE = 0.04          # flat annual risk-free rate
IV_MULT   = 1.20          # entry_vol = HV30 × 1.20  (IV spike during selloffs)
HV_WIN    = 30            # realized vol window (trading days)
DD_THRESH = 0.05          # drawdown threshold (run grid: 0.05, 0.07, 0.10)
DD_MIN    = 5             # minimum consecutive drawdown days to trigger signal

DELTAS       = [0.50, 0.60, 0.70, 0.80]
DTES         = [180, 270, 365, 540, 630, 720]
DD_GRID      = [0.05, 0.07, 0.10]   # pullback thresholds to sweep
PROFIT_MULT  = 2.0                   # active profit target multiplier (overridden by grid)
PROFIT_GRID  = [1.25, 1.50, 1.75, 2.0, 2.5, 3.0]  # profit multipliers to test

# ─── SETUP ────────────────────────────────────────────────────────────────────
load_dotenv(ENV_PATH)
warnings.filterwarnings("ignore")
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

from alpaca.data.enums      import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest
from alpaca.data.timeframe  import TimeFrame

# ─── DATA ─────────────────────────────────────────────────────────────────────
def fetch_daily(symbol: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(START.year, START.month, START.day, tzinfo=timezone.utc),
        end=datetime(END.year, END.month, END.day, tzinfo=timezone.utc),
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["open", "close"]].sort_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rolling 252-day high and drawdown from it
    df["high252"]  = df["close"].rolling(252).max()
    df["drawdown"] = (df["close"] - df["high252"]) / df["high252"]

    # Consecutive days where drawdown >= DD_THRESH (vectorised loop)
    in_dd   = (df["drawdown"] <= -DD_THRESH).values
    dd_days = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        if in_dd[i]:
            dd_days[i] = dd_days[i - 1] + 1 if i > 0 else 1

    df["dd_days"] = dd_days

    # 30-day annualised realised vol (log-return std × √252)
    log_ret   = np.log(df["close"] / df["close"].shift(1))
    df["hv30"] = log_ret.rolling(HV_WIN).std() * np.sqrt(252)

    # Entry signal: in a ≥10% drawdown for ≥5 consecutive days
    df["signal"] = dd_days >= DD_MIN

    return df.dropna(subset=["high252", "hv30"])


# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_delta(
    S: float, T: float, r: float, sigma: float, target_delta: float
) -> float:
    """Invert BS delta: find K s.t. bs_call_delta(S,K,T,r,sigma) = target_delta."""
    return brentq(
        lambda K: bs_call_delta(S, K, T, r, sigma) - target_delta,
        S * 0.3,
        S * 2.0,
    )


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest(
    df: pd.DataFrame, target_delta: float, dte: int
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run one delta×DTE combo.  Returns (equity_curve, trades_df).

    Per-bar order:
      1. Monthly infusion
      2. Close expiring position (intrinsic value)
      3. Check 100% gain target (2× entry premium, daily BS mark)
      4. Enter if prior bar fired a signal (pending_hv set)
      5. Fire signal for next bar if conditions met and no open position
      6. Mark-to-market equity
    """
    dates  = df.index.tolist()
    close  = df["close"].values
    open_  = df["open"].values
    hv30   = df["hv30"].values
    signal = df["signal"].values

    cash   = INIT_CAP
    pos    = None                               # active position dict or None
    equity = np.empty(len(dates))
    trades: list[dict] = []

    seen_months: set[tuple[int, int]] = set()
    pending_hv: Optional[float] = None          # HV30 from signal bar; consume on next open

    for i, dt in enumerate(dates):
        S  = close[i]
        hv = hv30[i]
        ym = (dt.year, dt.month)

        # 1. Monthly infusion — first trading day of each calendar month
        if ym not in seen_months:
            seen_months.add(ym)
            if i > 0:
                cash += MONTHLY

        # 2. Close position that has reached or passed its expiry date
        if pos is not None and dt >= pos["expiry"]:
            intrinsic = max(0.0, S - pos["K"])
            exit_val  = intrinsic * 100.0 * pos["n"]
            pnl_pct   = (exit_val / pos["cost"] - 1.0) * 100.0
            trades.append({
                **pos["log"],
                "exit_date":  dt,
                "exit_type":  "expiry",
                "exit_value": round(exit_val, 2),
                "pnl_pct":    round(pnl_pct, 2),
            })
            cash += exit_val
            pos = None

        # 3. Check profit target using raw HV for mark (no entry premium)
        if pos is not None:
            T_rem   = max(1e-6, (pos["expiry"] - dt).days / 365.0)
            vol_mtm = max(0.01, hv)
            mtm     = bs_call_price(S, pos["K"], T_rem, RISK_FREE, vol_mtm)
            if mtm >= PROFIT_MULT * pos["premium"]:
                exit_val = mtm * 100.0 * pos["n"]
                pnl_pct  = (exit_val / pos["cost"] - 1.0) * 100.0
                trades.append({
                    **pos["log"],
                    "exit_date":  dt,
                    "exit_type":  f"{PROFIT_MULT}x_target",
                    "exit_value": round(exit_val, 2),
                    "pnl_pct":    round(pnl_pct, 2),
                })
                cash += exit_val
                pos = None

        # 4. Enter at today's open if prior bar signalled (and slot still free)
        if pending_hv is not None:
            if pos is None:
                S0  = open_[i]
                ev  = pending_hv * IV_MULT       # pumped vol for entry only
                T0  = dte / 365.0
                try:
                    K       = find_strike_for_delta(S0, T0, RISK_FREE, ev, target_delta)
                    premium = bs_call_price(S0, K, T0, RISK_FREE, ev)
                    lot     = premium * 100.0
                    if lot > 0.0 and cash >= lot:
                        n      = int(cash // lot)
                        cost   = n * lot
                        expiry = dt + timedelta(days=dte)
                        pos = {
                            "K":       K,
                            "premium": premium,
                            "n":       n,
                            "cost":    cost,
                            "expiry":  expiry,
                            "log": {
                                "entry_date":    dt,
                                "expiry_date":   expiry,
                                "delta_target":  target_delta,
                                "dte":           dte,
                                "strike":        round(K, 2),
                                "entry_premium": round(premium, 4),
                                "contracts":     n,
                                "entry_cost":    round(cost, 2),
                                "entry_vol":     round(ev, 4),
                                "entry_spy":     round(S0, 2),
                            },
                        }
                        cash -= cost
                except (ValueError, RuntimeError):
                    pass  # brentq failed or degenerate inputs — skip
            pending_hv = None

        # 5. Fire entry signal for next bar if no open position
        if pos is None and signal[i]:
            pending_hv = hv

        # 6. Daily mark-to-market equity = cash + option MTM
        pos_val = 0.0
        if pos is not None:
            T_rem   = max(1e-6, (pos["expiry"] - dt).days / 365.0)
            vol_mtm = max(0.01, hv)
            pos_val = (
                bs_call_price(S, pos["K"], T_rem, RISK_FREE, vol_mtm) * 100.0 * pos["n"]
            )
        equity[i] = cash + pos_val

    _cols = [
        "entry_date", "expiry_date", "delta_target", "dte", "strike",
        "entry_premium", "contracts", "entry_cost", "entry_vol", "entry_spy",
        "exit_date", "exit_type", "exit_value", "pnl_pct",
    ]
    trades_df = pd.DataFrame(trades, columns=_cols) if trades else pd.DataFrame(columns=_cols)
    return pd.Series(equity, index=dates, name="equity"), trades_df


# ─── BUY-AND-HOLD BENCHMARK ───────────────────────────────────────────────────
def run_bah(df: pd.DataFrame) -> pd.Series:
    """SPY buy-and-hold: invest all available cash on the first trading day of
    each month (initial capital + $500/month infusions), fractional shares OK."""
    dates  = df.index.tolist()
    close  = df["close"].values
    cash   = INIT_CAP
    shares = 0.0
    equity = np.empty(len(dates))
    seen_months: set[tuple[int, int]] = set()

    for i, dt in enumerate(dates):
        ym = (dt.year, dt.month)
        if ym not in seen_months:
            seen_months.add(ym)
            if i > 0:
                cash += MONTHLY
            if close[i] > 0:
                shares += cash / close[i]
                cash = 0.0
        equity[i] = cash + shares * close[i]

    return pd.Series(equity, index=dates, name="Buy-and-Hold")


# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────
def compute_metrics(eq: pd.Series, trades: pd.DataFrame, label: str) -> dict:
    eq      = eq.dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    total   = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr    = (1.0 + total) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    monthly = eq.resample("ME").last().pct_change().dropna()
    rf_m    = (1.0 + RISK_FREE) ** (1.0 / 12) - 1.0
    excess  = monthly - rf_m
    sharpe  = float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0

    roll_max = eq.cummax()
    max_dd   = float(((eq - roll_max) / roll_max).min())

    n_t      = len(trades)
    win_rate = float((trades["pnl_pct"] > 0).mean() * 100) if n_t > 0 else 0.0
    avg_ret  = float(trades["pnl_pct"].mean())              if n_t > 0 else 0.0

    return {
        "label":         label,
        "final_value":   round(eq.iloc[-1], 2),
        "total_return":  round(total * 100, 2),
        "cagr_pct":      round(cagr * 100, 2),
        "sharpe":        round(sharpe, 3),
        "max_dd_pct":    round(max_dd * 100, 2),
        "n_trades":      n_t,
        "win_rate_pct":  round(win_rate, 1),
        "avg_trade_ret": round(avg_ret, 2),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_threshold(df_raw: pd.DataFrame, dd_thresh: float, profit_mult: float) -> tuple[list, pd.Series, dict]:
    """Run all delta x DTE combos for one pullback threshold + profit target. Returns (all_results, bah_eq, bah_m)."""
    global DD_THRESH, PROFIT_MULT
    DD_THRESH   = dd_thresh
    PROFIT_MULT = profit_mult

    df = build_features(df_raw)
    n_signals = int(df["signal"].sum())
    pm_str = f"{int((profit_mult-1)*100)}%"
    print(f"\n  [DD>={dd_thresh*100:.0f}% | PT={pm_str}] Signals: {n_signals}")

    all_results = []
    for delta in DELTAS:
        for dte in DTES:
            label = f"D{int(delta*100)}_DTE{dte}"
            eq, trades = run_backtest(df, delta, dte)
            m = compute_metrics(eq, trades, label)
            m["dd_thresh"]   = dd_thresh
            m["profit_mult"] = profit_mult
            all_results.append((label, eq, trades, m))

    bah_eq = run_bah(df)
    bah_m  = compute_metrics(bah_eq, pd.DataFrame(columns=["pnl_pct"]), "Buy-and-Hold")
    bah_m["dd_thresh"]   = dd_thresh
    bah_m["profit_mult"] = profit_mult
    return all_results, bah_eq, bah_m


def main() -> None:
    print(f"{TICKER} LEAP Call Backtest — Pullback x Profit Target Grid")
    print("=" * 70)
    print(f"Fetching {TICKER} daily bars {START} to {END}...")
    raw = fetch_daily(TICKER)
    print(f"  {len(raw)} trading days  ({raw.index[0].date()} to {raw.index[-1].date()})")
    print(f"  DD grid: {[f'{x*100:.0f}%' for x in DD_GRID]}  |  "
          f"Profit grid: {[f'{int((x-1)*100)}%' for x in PROFIT_GRID]}")

    all_rows  = []   # flat list of metric dicts for master CSV
    bah_eq_ref = None
    bah_m_ref  = None

    # Track best combo per (dd_thresh, profit_mult) for the equity chart
    best_per_slice: list[tuple] = []

    for profit_mult in PROFIT_GRID:
        for dd_thresh in DD_GRID:
            results, bah_eq, bah_m = run_threshold(raw, dd_thresh, profit_mult)
            if bah_eq_ref is None:
                bah_eq_ref = bah_eq
                bah_m_ref  = bah_m
            for _, _, _, m in results:
                all_rows.append(m)
            all_rows.append(bah_m)

            best = max(results, key=lambda x: x[3]["sharpe"])
            best_per_slice.append((dd_thresh, profit_mult, best, bah_m))

    # ── Master CSV ───────────────────────────────────────────────────────────
    summary_all = pd.DataFrame(all_rows)
    summary_all.to_csv(OUT_DIR / "summary_all.csv", index=False)

    # ── Per profit-target summary (top combos across all DD thresholds) ──────
    for profit_mult in PROFIT_GRID:
        pm_pct = int((profit_mult - 1) * 100)
        subset = [r for r in all_rows if r.get("profit_mult") == profit_mult and r["label"] != "Buy-and-Hold"]
        df_pm  = pd.DataFrame(subset).sort_values("sharpe", ascending=False)
        df_pm.to_csv(OUT_DIR / f"summary_pt{pm_pct}.csv", index=False)

    # ── Print top-5 by Sharpe for each profit target (across all DD thresholds)
    W = 118
    print()
    print("=" * W)
    print(f"  {'PT%':>5}  {'DD%':>5}  {'Label':<18}  {'Trades':>6}  {'Sharpe':>7}  "
          f"{'CAGR%':>7}  {'MaxDD%':>8}  {'Win%':>6}  {'Final$':>10}")
    print("  " + "-" * (W - 2))
    bah_sharpe = bah_m_ref["sharpe"] if bah_m_ref else 0
    for profit_mult in PROFIT_GRID:
        pm_pct = int((profit_mult - 1) * 100)
        subset = [r for r in all_rows if r.get("profit_mult") == profit_mult and r["label"] != "Buy-and-Hold"]
        top5   = sorted(subset, key=lambda r: r["sharpe"], reverse=True)[:5]
        print(f"  --- Profit target +{pm_pct}%  (B&H Sharpe {bah_sharpe:.2f}) ---")
        for m in top5:
            print(
                f"  {pm_pct:>5}  {m['dd_thresh']*100:>5.0f}  {m['label']:<18}  "
                f"{m['n_trades']:>6}  {m['sharpe']:>7.3f}  {m['cagr_pct']:>7.1f}  "
                f"{m['max_dd_pct']:>8.1f}  {m['win_rate_pct']:>6.1f}  {m['final_value']:>10,.0f}"
            )
    print("=" * W)

    # ── Equity curve: overall top-6 combos by Sharpe + B&H ───────────────────
    all_non_bah = [r for r in all_rows if r["label"] != "Buy-and-Hold"]
    top6_meta   = sorted(all_non_bah, key=lambda r: r["sharpe"], reverse=True)[:6]
    top6_keys   = {(r["label"], r["dd_thresh"], r["profit_mult"]) for r in top6_meta}

    # Re-run the top6 to get equity curves (cheapest approach: just keep them)
    # We need to rebuild them — store during main loop instead
    # Simpler: emit a note and skip the chart (too many combos to store all curves)
    # Instead, plot best combo per profit-target bucket
    palette = list(plt.cm.tab10(np.linspace(0, 0.85, len(PROFIT_GRID) + 1)))
    fig, ax = plt.subplots(figsize=(15, 7))

    plotted = 0
    for i, profit_mult in enumerate(PROFIT_GRID):
        pm_pct = int((profit_mult - 1) * 100)
        subset = [r for r in all_rows if r.get("profit_mult") == profit_mult and r["label"] != "Buy-and-Hold"]
        if not subset:
            continue
        best_m = max(subset, key=lambda r: r["sharpe"])
        # Re-run that single combo to get the equity curve
        global DD_THRESH, PROFIT_MULT
        DD_THRESH   = best_m["dd_thresh"]
        PROFIT_MULT = profit_mult
        df_feat = build_features(raw)
        parts   = best_m["label"].split("_")
        delta   = int(parts[0][1:]) / 100
        dte     = int(parts[1][3:])
        eq, _   = run_backtest(df_feat, delta, dte)
        ax.semilogy(
            eq.index, eq.values,
            label=(f"PT+{pm_pct}% DD>={best_m['dd_thresh']*100:.0f}% {best_m['label']}  "
                   f"(Sharpe {best_m['sharpe']:.2f} | CAGR {best_m['cagr_pct']:.1f}%)"),
            color=palette[i], linewidth=1.8,
        )
        plotted += 1

    ax.semilogy(
        bah_eq_ref.index, bah_eq_ref.values,
        label=f"Buy-and-Hold  (Sharpe {bah_m_ref['sharpe']:.2f} | CAGR {bah_m_ref['cagr_pct']:.1f}%)",
        color="black", linewidth=2.2, linestyle="--",
    )
    ax.set_title(f"{TICKER} LEAP — Best Combo per Profit Target vs B&H  (log scale)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curves_profit_targets.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved to {OUT_DIR}/")
    print("  summary_all.csv  |  summary_pt*.csv  |  equity_curves_profit_targets.png")


if __name__ == "__main__":
    main()
