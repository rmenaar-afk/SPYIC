"""
0DTE Iron Condor Backtest -- SPY
=================================
Strategy
--------
  Put Credit Spread  : sell ~0.15-delta put, buy 10-wide put below
  Call Credit Spread : sell ~0.10-delta call, buy 5-wide call above
  Entry              : first bar at or after 10:00 AM ET
  Exit               : 75% profit captured  OR  3:00 PM ET (1 hr before close)
  Days               : every trading day with SPY minute data (2022-01-03 -> today)

Pricing
-------
  Black-Scholes with daily VIX as IV proxy.
  Time-to-expiry = calendar minutes remaining to 4:00 PM / (252 * 390 minutes).
  Strike widths are in SPY dollars (10-wide put, 5-wide call).

Output
------
  results/dte0_ic/summary.txt
  results/dte0_ic/trades.csv
  results/dte0_ic/equity_curve.png
"""

import os, sys, warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────────
ENV_PATH  = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\.env")
OUT_DIR   = Path(r"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\results\dte0_ic")
START     = date(2022, 1, 3)
END       = date.today()

TICKER    = "SPY"
PUT_DELTA_TARGET  = 0.15    # sell-put target delta
CALL_DELTA_TARGET = 0.10    # sell-call target delta
PUT_WIDTH         = 10.0    # put spread width in $ (10-wide)
CALL_WIDTH        = 5.0     # call spread width in $ (5-wide)
PROFIT_TARGET     = 0.75    # close when P&L >= 75% of max credit
ENTRY_TIME        = "10:00" # ET -- first bar at or after this time
EXIT_TIME         = "15:00" # ET -- force close at this time
RISK_FREE         = 0.04
IV_MULT           = 1.0     # VIX as-is (VIX already annualised %)
INIT_CAP          = 25_000.0
CONTRACTS         = 1       # contracts per leg (1 contract = 100 shares)

ET = ZoneInfo("America/New_York")
warnings.filterwarnings("ignore")
OUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ENV_PATH)

# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_spy_minutes(start: date, end: date) -> pd.DataFrame:
    """Fetch SPY 1-min bars from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    client = StockHistoricalDataClient(
        api_key    = os.environ["ALPACA_API_KEY"],
        secret_key = os.environ["ALPACA_SECRET_KEY"],
    )
    req = StockBarsRequest(
        symbol_or_symbols = TICKER,
        timeframe         = TimeFrame(1, TimeFrameUnit.Minute),
        start             = datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
        end               = datetime(end.year,   end.month,   end.day,   tzinfo=timezone.utc),
        adjustment        = Adjustment.ALL,
        feed              = DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(TICKER, level="symbol")
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    return bars[["open", "high", "low", "close", "volume"]]


def fetch_vix() -> pd.Series:
    """Fetch ^VIX daily close from Yahoo Finance."""
    import yfinance as yf
    v = yf.download("^VIX", start=str(START), end=str(END + timedelta(days=1)),
                    progress=False, auto_adjust=True)
    v.columns = [c[0] if isinstance(c, tuple) else c for c in v.columns]
    s = v["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s / 100.0   # convert 20.0 -> 0.20


# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────
def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0  # negative for puts

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def strike_for_put_delta(S, T, r, sigma, target_delta):
    """Find K such that |put_delta(S,K,T,r,sigma)| == target_delta. target_delta > 0."""
    lo, hi = S * 0.50, S * 1.10
    try:
        return brentq(lambda K: abs(bs_put_delta(S, K, T, r, sigma)) - target_delta, lo, hi)
    except Exception:
        return S * (1 - target_delta * sigma * np.sqrt(T) * 1.5)

def strike_for_call_delta(S, T, r, sigma, target_delta):
    """Find K such that call_delta(S,K,T,r,sigma) == target_delta. target_delta > 0."""
    lo, hi = S * 0.90, S * 1.50
    try:
        return brentq(lambda K: bs_call_delta(S, K, T, r, sigma) - target_delta, lo, hi)
    except Exception:
        return S * (1 + target_delta * sigma * np.sqrt(T) * 1.5)

def minutes_to_T(mins_remaining: float) -> float:
    """Convert minutes remaining in trading day to annualised time fraction."""
    return max(mins_remaining / (252.0 * 390.0), 1e-8)


# ── SINGLE-DAY SIMULATION ─────────────────────────────────────────────────────
def simulate_day(day_bars: pd.DataFrame, iv: float) -> dict | None:
    """
    Run one trading day. Returns a trade dict or None if no valid entry.
    """
    # Filter to regular session only: 9:30 - 15:59 ET
    session = day_bars.between_time("09:30", "15:59")
    if session.empty:
        return None

    # Market close = 4:00 PM; bars close at 16:00 but last tradeable is 15:59
    market_close_et = session.index[-1].replace(hour=16, minute=0, second=0, microsecond=0)

    # Find entry bar (first bar >= ENTRY_TIME)
    entry_bars = session[session.index.strftime("%H:%M") >= ENTRY_TIME]
    if entry_bars.empty:
        return None
    entry_bar = entry_bars.iloc[0]
    entry_ts  = entry_bars.index[0]

    S_entry = entry_bar["open"]
    mins_left_entry = (market_close_et - entry_ts).total_seconds() / 60.0
    T_entry = minutes_to_T(mins_left_entry)

    sigma = max(iv * IV_MULT, 0.05)  # floor at 5% annualised

    # ── Find short strikes ────────────────────────────────────────────────────
    K_put_short  = round(strike_for_put_delta(S_entry, T_entry, RISK_FREE, sigma, PUT_DELTA_TARGET),  2)
    K_call_short = round(strike_for_call_delta(S_entry, T_entry, RISK_FREE, sigma, CALL_DELTA_TARGET), 2)

    # Long strikes (fixed width below/above)
    K_put_long  = round(K_put_short  - PUT_WIDTH,  2)
    K_call_long = round(K_call_short + CALL_WIDTH, 2)

    # Sanity check: put strikes must be below spot; call strikes above spot
    if K_put_short >= S_entry or K_call_short <= S_entry:
        return None

    # ── Entry credit ─────────────────────────────────────────────────────────
    put_short_px  = bs_put_price (S_entry, K_put_short,  T_entry, RISK_FREE, sigma)
    put_long_px   = bs_put_price (S_entry, K_put_long,   T_entry, RISK_FREE, sigma)
    call_short_px = bs_call_price(S_entry, K_call_short, T_entry, RISK_FREE, sigma)
    call_long_px  = bs_call_price(S_entry, K_call_long,  T_entry, RISK_FREE, sigma)

    put_credit  = (put_short_px  - put_long_px)  * 100.0 * CONTRACTS
    call_credit = (call_short_px - call_long_px) * 100.0 * CONTRACTS
    total_credit = put_credit + call_credit
    if total_credit <= 0:
        return None

    max_profit = total_credit
    profit_target_dollar = PROFIT_TARGET * max_profit  # 75% of credit

    # Max loss = width * 100 * contracts - credit (use wider spread)
    max_loss_per_spread = max(PUT_WIDTH, CALL_WIDTH) * 100.0 * CONTRACTS
    max_loss = max_loss_per_spread - total_credit

    # ── Intraday simulation ───────────────────────────────────────────────────
    after_entry = session[session.index >= entry_ts]
    exit_ts     = None
    exit_reason = "eod"
    pnl         = None
    exit_price  = None

    for ts, bar in after_entry.iterrows():
        # Force close at EXIT_TIME
        if ts.strftime("%H:%M") >= EXIT_TIME:
            S_exit = bar["open"]
            mins_left = (market_close_et - ts).total_seconds() / 60.0
            T_exit = minutes_to_T(mins_left)
            cost = _spread_cost(S_exit, K_put_short, K_put_long, K_call_short, K_call_long,
                                T_exit, RISK_FREE, sigma)
            pnl = total_credit - cost
            exit_ts     = ts
            exit_reason = "time"
            exit_price  = S_exit
            break

        # Check profit target using bar's high AND low to find best-case exit within bar
        for S_check in [bar["low"], bar["high"], bar["close"]]:
            mins_left = (market_close_et - ts).total_seconds() / 60.0
            T_check = minutes_to_T(mins_left)
            cost = _spread_cost(S_check, K_put_short, K_put_long, K_call_short, K_call_long,
                                T_check, RISK_FREE, sigma)
            pnl_check = total_credit - cost
            if pnl_check >= profit_target_dollar:
                pnl = pnl_check
                exit_ts     = ts
                exit_reason = "profit_target"
                exit_price  = S_check
                break
        if exit_ts is not None:
            break

    # If still open (shouldn't happen but just in case), close at last bar
    if exit_ts is None:
        last = after_entry.iloc[-1]
        ts   = after_entry.index[-1]
        mins_left = max((market_close_et - ts).total_seconds() / 60.0, 1)
        T_exit = minutes_to_T(mins_left)
        S_exit = last["close"]
        cost = _spread_cost(S_exit, K_put_short, K_put_long, K_call_short, K_call_long,
                            T_exit, RISK_FREE, sigma)
        pnl = total_credit - cost
        exit_ts     = ts
        exit_reason = "eod"
        exit_price  = S_exit

    # High-move flag: SPY moved more than 1% from prior close to entry price
    prior_close = day_bars["close"].iloc[0] if len(day_bars) > 0 else S_entry
    # use first bar open as proxy for prior session reference; better: compare to prev day close
    # We'll compute this in main() with actual prev-day close; for now store S_open
    S_open930 = session.iloc[0]["open"]

    return {
        "date":          entry_ts.date(),
        "entry_ts":      entry_ts,
        "exit_ts":       exit_ts,
        "exit_reason":   exit_reason,
        "S_entry":       round(S_entry, 2),
        "S_open930":     round(S_open930, 2),
        "S_exit":        round(exit_price, 2),
        "K_put_short":   round(K_put_short,  2),
        "K_put_long":    round(K_put_long,   2),
        "K_call_short":  round(K_call_short, 2),
        "K_call_long":   round(K_call_long,  2),
        "put_credit":    round(put_credit,   2),
        "call_credit":   round(call_credit,  2),
        "total_credit":  round(total_credit, 2),
        "pnl":           round(pnl, 2),
        "pnl_pct":       round(pnl / max_profit * 100, 1),
        "max_profit":    round(max_profit, 2),
        "max_loss":      round(max_loss, 2),
        "iv":            round(sigma, 4),
    }


def _spread_cost(S, K_put_short, K_put_long, K_call_short, K_call_long, T, r, sigma):
    """Current cost to close both spreads."""
    put_short  = bs_put_price (S, K_put_short,  T, r, sigma)
    put_long   = bs_put_price (S, K_put_long,   T, r, sigma)
    call_short = bs_call_price(S, K_call_short, T, r, sigma)
    call_long  = bs_call_price(S, K_call_long,  T, r, sigma)
    return ((put_short - put_long) + (call_short - call_long)) * 100.0 * CONTRACTS


VIX_THRESHOLDS = [20, 19, 18, 17, 16, 15]  # only trade when VIX < this level
HIGH_MOVE_PCT  = 0.01                        # flag days where SPY gaps >1% from prev close


def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    """Compute summary metrics for a subset of trades."""
    if df.empty:
        return {"label": label, "n": 0}
    n          = len(df)
    total_pnl  = df["pnl"].sum()
    win_rate   = (df["pnl"] > 0).mean() * 100
    avg_win    = df.loc[df["pnl"] > 0, "pnl"].mean() if (df["pnl"] > 0).any() else 0
    avg_loss   = df.loc[df["pnl"] < 0, "pnl"].mean() if (df["pnl"] < 0).any() else 0
    pf_den     = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    pf         = abs(df.loc[df["pnl"] > 0, "pnl"].sum()) / pf_den if pf_den > 0 else float("inf")
    eq         = INIT_CAP + df["pnl"].cumsum()
    max_dd     = (eq - eq.cummax()).min()
    years      = max((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25, 0.01)
    cagr       = ((eq.iloc[-1] / INIT_CAP) ** (1 / years) - 1) * 100
    daily_ret  = df["pnl"] / INIT_CAP
    sharpe     = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    avg_credit = df["total_credit"].mean()
    pt_pct     = (df["exit_reason"] == "profit_target").mean() * 100
    return {
        "label":       label,
        "n":           n,
        "win_rate":    round(win_rate, 1),
        "avg_win":     round(avg_win, 2),
        "avg_loss":    round(avg_loss, 2),
        "profit_factor": round(pf, 2),
        "total_pnl":   round(total_pnl, 2),
        "final_equity": round(eq.iloc[-1], 2),
        "cagr":        round(cagr, 1),
        "sharpe":      round(sharpe, 2),
        "max_dd":      round(max_dd, 2),
        "avg_credit":  round(avg_credit, 2),
        "pt_pct":      round(pt_pct, 1),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("0DTE Iron Condor Backtest -- SPY  (VIX filter + high-move analysis)")
    print("=" * 70)

    print("Fetching SPY 1-min bars (this may take a few minutes)...")
    spy = fetch_spy_minutes(START, END)
    print(f"  {len(spy):,} minute bars  ({spy.index[0].date()} -> {spy.index[-1].date()})")

    print("Fetching VIX daily data from Yahoo Finance...")
    vix = fetch_vix()
    print(f"  {len(vix)} VIX observations")

    # Get SPY daily bars for prev-close high-move detection
    spy_daily = spy.groupby(spy.index.normalize().tz_localize(None))["close"].last()

    # Group bars by trading day
    spy["_date"] = spy.index.normalize().tz_localize(None)
    trading_days = sorted(spy["_date"].unique())
    print(f"  {len(trading_days)} trading days to simulate\n")

    trades = []
    prev_close = None
    for d in trading_days:
        day_bars = spy[spy["_date"] == d].copy()
        iv_date  = pd.Timestamp(d)
        if iv_date not in vix.index:
            for lag in range(1, 5):
                alt = iv_date - pd.Timedelta(days=lag)
                if alt in vix.index:
                    iv_date = alt
                    break
            else:
                prev_close = float(spy_daily.get(pd.Timestamp(d), np.nan))
                continue
        iv      = float(vix.loc[iv_date])
        vix_raw = iv * 100.0   # store as e.g. 18.5

        result = simulate_day(day_bars, iv)
        if result:
            # Tag high-move days: compare 9:30 open to previous day's close
            if prev_close and not np.isnan(prev_close):
                gap_pct = abs(result["S_open930"] - prev_close) / prev_close
                result["high_move"] = gap_pct >= HIGH_MOVE_PCT
                result["gap_pct"]   = round(gap_pct * 100, 2)
            else:
                result["high_move"] = False
                result["gap_pct"]   = 0.0
            result["vix"] = round(vix_raw, 2)
            trades.append(result)

        prev_close = float(spy_daily.get(pd.Timestamp(d), prev_close or np.nan))

        if len(trades) % 100 == 0 and len(trades) > 0:
            print(f"  ... {len(trades)} trades (day {d})")

    if not trades:
        print("No trades generated!")
        return

    df = pd.DataFrame(trades)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_DIR / "trades.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HIGH-MOVE DAY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    normal_days = df[~df["high_move"]]
    highmove_days = df[df["high_move"]]
    m_all    = compute_metrics(df,            "All days (no filter)")
    m_normal = compute_metrics(normal_days,   f"Normal days (gap <{HIGH_MOVE_PCT*100:.0f}%)")
    m_highmv = compute_metrics(highmove_days, f"High-move days (gap >={HIGH_MOVE_PCT*100:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VIX FILTER GRID
    # ─────────────────────────────────────────────────────────────────────────
    vix_metrics = []
    for thresh in VIX_THRESHOLDS:
        subset = df[df["vix"] < thresh]
        vix_metrics.append(compute_metrics(subset, f"VIX < {thresh}"))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PRINT SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    W = 112
    header = (f"{'Filter':<28}  {'N':>5}  {'WinR%':>6}  {'AvgW':>7}  {'AvgL':>7}  "
              f"{'PF':>5}  {'Sharpe':>7}  {'CAGR%':>7}  {'MaxDD':>9}  {'TotPnL':>10}  {'AvgCrd':>7}")
    divider = "-" * W

    def fmt_row(m):
        if m["n"] == 0:
            return f"  {m['label']:<28}  {'0':>5}  -- no trades --"
        return (f"  {m['label']:<28}  {m['n']:>5}  {m['win_rate']:>6.1f}  "
                f"${m['avg_win']:>6.2f}  ${m['avg_loss']:>6.2f}  "
                f"{m['profit_factor']:>5.2f}  {m['sharpe']:>7.2f}  {m['cagr']:>7.1f}  "
                f"${m['max_dd']:>8,.0f}  ${m['total_pnl']:>9,.0f}  ${m['avg_credit']:>6.2f}")

    print()
    print("HIGH-MOVE DAY ANALYSIS")
    print("=" * W)
    print("  " + header)
    print("  " + divider)
    for m in [m_all, m_normal, m_highmv]:
        print(fmt_row(m))
    print()

    print("VIX FILTER GRID  (only open trade when VIX < threshold)")
    print("=" * W)
    print("  " + header)
    print("  " + divider)
    for m in vix_metrics:
        print(fmt_row(m))
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SAVE SUMMARY TEXT
    # ─────────────────────────────────────────────────────────────────────────
    lines = [
        f"0DTE Iron Condor -- {TICKER}  |  PCS 0.15d 10-wide / CCS 0.10d 5-wide",
        f"Entry {ENTRY_TIME} ET  |  Exit {PROFIT_TARGET:.0%} profit OR {EXIT_TIME} ET",
        f"Period: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}",
        "",
        "HIGH-MOVE DAY BREAKDOWN",
        header, divider,
        fmt_row(m_all), fmt_row(m_normal), fmt_row(m_highmv),
        "",
        "VIX FILTER GRID",
        header, divider,
    ] + [fmt_row(m) for m in vix_metrics]
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    palette = plt.cm.tab10(np.linspace(0, 0.6, len(VIX_THRESHOLDS)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: equity all days + high-move overlay
    ax = axes[0, 0]
    eq_all    = INIT_CAP + df["pnl"].cumsum()
    eq_normal = INIT_CAP + normal_days["pnl"].cumsum()
    ax.plot(df["date"], eq_all, color="steelblue", linewidth=1.5, label=f"All days  (Sharpe {m_all['sharpe']:.2f})")
    ax.plot(normal_days["date"], eq_normal, color="green", linewidth=1.2, linestyle="--",
            label=f"Normal days only  (Sharpe {m_normal['sharpe']:.2f})")
    ax.axhline(INIT_CAP, color="gray", linestyle=":", linewidth=0.8)
    # Shade high-move days
    for _, row in highmove_days.iterrows():
        ax.axvspan(row["date"], row["date"], alpha=0.15, color="red")
    ax.set_title("Equity: All vs Normal-Move Days\n(red = high-move day)", fontsize=10)
    ax.set_ylabel("Portfolio ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Top-right: high-move vs normal P&L distribution
    ax2 = axes[0, 1]
    ax2.hist(normal_days["pnl"], bins=40, alpha=0.6, color="steelblue", label=f"Normal ({len(normal_days)})")
    ax2.hist(highmove_days["pnl"], bins=20, alpha=0.6, color="crimson", label=f"High-move ({len(highmove_days)})")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_title("P&L Distribution: Normal vs High-Move Days", fontsize=10)
    ax2.set_xlabel("P&L per trade ($)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # Bottom-left: VIX filter equity curves
    ax3 = axes[1, 0]
    for i, thresh in enumerate(VIX_THRESHOLDS):
        subset = df[df["vix"] < thresh]
        if subset.empty:
            continue
        eq = INIT_CAP + subset["pnl"].cumsum()
        m  = vix_metrics[i]
        ax3.plot(subset["date"], eq, color=palette[i], linewidth=1.4,
                 label=f"VIX<{thresh}  n={m['n']}  Sh={m['sharpe']:.2f}")
    ax3.axhline(INIT_CAP, color="gray", linestyle=":", linewidth=0.8)
    ax3.set_title("Equity Curves by VIX Filter", fontsize=10)
    ax3.set_ylabel("Portfolio ($)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, alpha=0.25)

    # Bottom-right: VIX filter comparison bar chart (Sharpe + CAGR)
    ax4 = axes[1, 1]
    labels  = [f"VIX<{t}" for t in VIX_THRESHOLDS]
    sharpes = [m["sharpe"] for m in vix_metrics]
    cagrs   = [m["cagr"]   for m in vix_metrics]
    x = np.arange(len(labels))
    w = 0.35
    ax4.bar(x - w/2, sharpes, w, label="Sharpe", color="steelblue", alpha=0.8)
    ax4r = ax4.twinx()
    ax4r.bar(x + w/2, cagrs, w, label="CAGR %", color="darkorange", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel("Sharpe Ratio", color="steelblue")
    ax4r.set_ylabel("CAGR %", color="darkorange")
    ax4.set_title("VIX Filter: Sharpe vs CAGR", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4r.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.25, axis="y")

    fig.suptitle(f"0DTE Iron Condor -- {TICKER}  |  PCS 0.15d 10-wide / CCS 0.10d 5-wide  |  "
                 f"Exit: {PROFIT_TARGET:.0%} profit or {EXIT_TIME} ET",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "analysis.png", dpi=150)
    plt.close(fig)

    print(f"Saved to {OUT_DIR}/")
    print("  summary.txt  |  trades.csv  |  analysis.png")


if __name__ == "__main__":
    main()
