#!/usr/bin/env python3
"""
breakout_scanner.py  –  Daily Consolidation Breakout Scanner

Scans the S&P 500 + S&P 400 universe for stocks breaking out of tight
consolidation bases on above-average volume.

Schedule: Run after 4:30 PM ET on trading days (market close + data settlement).
  Unix cron:    30 16 * * 1-5  /usr/bin/python /path/to/breakout_scanner.py
  Windows Task: schtasks /create /tn "BreakoutScanner" /tr "C:\\path\\run_scanner.bat"
                         /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 16:30
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tabulate import tabulate

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

CACHE_DIR       = Path("scanner_cache")
RESULTS_DIR     = Path("results/scanner")
CACHE_MAX_AGE_H = 12          # hours before per-symbol parquet cache is stale
LOOKBACK_DAYS   = 500         # calendar days to request (~340 trading days; covers 200-SMA + 6-month ATR dist)
MIN_CACHE_BARS  = 220         # invalidate cache if it has fewer rows (prevents short-lookback cache reuse)
BATCH_SIZE      = 50          # symbols per Alpaca API call

# Pre-filter thresholds
MIN_PRICE          = 10.0
MIN_AVG_DOLLAR_VOL = 10_000_000   # $10M/day average dollar volume

# Consolidation parameters
BASE_WINDOW        = 20     # trading days defining the consolidation base
ATR_WINDOW         = 10     # days for current short-term ATR
ATR_HIST_BARS      = 126    # bars (~6 months) for ATR percentile distribution
MAX_BASE_RANGE_PCT = 0.15   # max (max_close - min_close) / min_close over base
VOL_CONTRACTION    = 0.80   # recent 10d vol must be < 80% of prior 30d vol
NEAR_TOP_PCT       = 0.05   # yesterday close within 5% of base high (coiling)

# Breakout thresholds
VOL_BREAKOUT_RATIO = 1.5    # today vol > 1.5× 20d average

# Output
TOP_N = 20

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────────────

_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _wiki_table(url: str, symbol_col: str, name_col: str) -> pd.DataFrame:
    r = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0][[symbol_col, name_col]]
    df.columns = ["symbol", "name"]
    return df


def get_universe() -> pd.DataFrame:
    log.info("Building universe ...")
    frames: list[pd.DataFrame] = []

    for label, url, sym_col, name_col in [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
         "Symbol", "Security"),
        ("S&P 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
         "Symbol", "Security"),
    ]:
        try:
            df = _wiki_table(url, sym_col, name_col)
            log.info("  %-8s  %d symbols", label, len(df))
            frames.append(df)
        except Exception as exc:
            log.warning("  %s fetch failed: %s", label, exc)

    if not frames:
        log.error("Could not fetch any universe list — aborting.")
        sys.exit(1)

    universe = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("symbol")
        .reset_index(drop=True)
    )
    log.info("Universe total: %d symbols", len(universe))
    return universe


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _cache_is_fresh(symbol: str) -> bool:
    p = _cache_path(symbol)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < CACHE_MAX_AGE_H


def _load_cache(symbol: str) -> pd.DataFrame | None:
    if not _cache_is_fresh(symbol):
        return None
    df = pd.read_parquet(_cache_path(symbol))
    if len(df) < MIN_CACHE_BARS:
        return None   # cached with a shorter lookback window; re-fetch
    return df


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(symbol))


# ──────────────────────────────────────────────────────────────────────────────
# Data fetch
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_batch(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    raw = client.get_stock_bars(req).df.reset_index()
    result: dict[str, pd.DataFrame] = {}
    for sym, grp in raw.groupby("symbol"):
        result[sym] = grp.sort_values("timestamp").reset_index(drop=True)
    return result


def fetch_all_bars(
    client: StockHistoricalDataClient,
    universe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    symbols = universe["symbol"].tolist()
    end   = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    bars_dict: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        cached = _load_cache(sym)
        if cached is not None:
            bars_dict[sym] = cached
        else:
            to_fetch.append(sym)

    log.info("Cache: %d fresh  |  Fetching: %d symbols in batches of %d",
             len(bars_dict), len(to_fetch), BATCH_SIZE)

    total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i : i + BATCH_SIZE]
        batch_n = i // BATCH_SIZE + 1
        log.info("  Batch %d/%d  (%d symbols)", batch_n, total_batches, len(batch))
        try:
            fetched = _fetch_batch(client, batch, start, end)
            for sym, df in fetched.items():
                bars_dict[sym] = df
                _save_cache(sym, df)
            missing = set(batch) - set(fetched)
            if missing:
                log.debug("    No data for: %s", ", ".join(sorted(missing)))
        except Exception as exc:
            log.warning("    Batch %d failed: %s — skipping", batch_n, exc)
        time.sleep(0.15)   # light rate-limit courtesy

    return bars_dict


def fetch_spy(client: StockHistoricalDataClient) -> pd.Series:
    """SPY daily adjusted closes as a Series indexed by Python date objects."""
    req = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Day,
        start=date.today() - timedelta(days=LOOKBACK_DAYS),
        end=date.today(),
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(req).df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df.set_index("date")["close"].sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["sma_50"]  = c.rolling(50).mean()
    df["sma_200"] = c.rolling(200).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr_10"]  = tr.rolling(ATR_WINDOW).mean()
    df["atr_pct"] = df["atr_10"] / c            # ATR as fraction of close

    df["dolvol_20"]    = (c * v).rolling(20).mean()
    df["vol_avg_20"]   = v.rolling(20).mean()
    df["vol_avg_10"]   = v.rolling(10).mean()
    df["vol_prior_30"] = v.rolling(30).mean().shift(10)  # 30d avg ending 10 bars ago
    df["bar_range"]    = h - l
    df["avg_range_10"] = df["bar_range"].rolling(10).mean()

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Scanner core
# ──────────────────────────────────────────────────────────────────────────────

def _spy_at(spy: pd.Series, dt: date) -> float | None:
    """Nearest SPY close on or before dt."""
    candidates = spy[spy.index <= dt]
    return float(candidates.iloc[-1]) if not candidates.empty else None


def scan_symbol(sym: str, df: pd.DataFrame, spy: pd.Series) -> dict | None:
    """
    Scan one symbol. Returns a result dict with signal_type="breakout" or
    signal_type="coiling" (setup in place but not yet broken out), or None
    if no setup is detected.
    """
    if len(df) < 215:
        return None

    today = df.iloc[-1]
    prior = df.iloc[:-1]             # all bars before today
    base  = prior.tail(BASE_WINDOW)  # 20-bar consolidation window

    price = float(today["close"])

    # ── Pre-filters ───────────────────────────────────────────────────────────
    if price < MIN_PRICE:
        return None
    if pd.isna(today["dolvol_20"]) or float(today["dolvol_20"]) < MIN_AVG_DOLLAR_VOL:
        return None
    sma200 = today["sma_200"]
    sma50  = today["sma_50"]
    if pd.isna(sma200) or price <= float(sma200):
        return None
    if pd.isna(sma50) or float(sma50) <= float(sma200):
        return None

    # ── Consolidation: tight price range ─────────────────────────────────────
    if len(base) < BASE_WINDOW:
        return None

    base_closes = base["close"].astype(float)
    base_min = base_closes.min()
    base_max = base_closes.max()
    if base_min <= 0:
        return None

    range_pct = (base_max - base_min) / base_min
    if range_pct >= MAX_BASE_RANGE_PCT:
        return None

    # ── Consolidation: ATR squeeze (current ATR% in bottom 30% of 6-month dist) ──
    atr_hist        = prior["atr_pct"].tail(ATR_HIST_BARS).dropna()
    current_atr_pct = float(today["atr_pct"]) if not pd.isna(today["atr_pct"]) else None
    if len(atr_hist) < 40 or current_atr_pct is None:
        return None
    if float((atr_hist < current_atr_pct).mean()) > 0.30:
        return None

    # ── Consolidation: volume contraction ─────────────────────────────────────
    vol_10      = today["vol_avg_10"]
    vol_prior30 = today["vol_prior_30"]
    if pd.isna(vol_10) or pd.isna(vol_prior30) or float(vol_prior30) <= 0:
        return None
    if float(vol_10) >= VOL_CONTRACTION * float(vol_prior30):
        return None

    # ── Consolidation: coiling near range top ─────────────────────────────────
    yesterday_close = float(prior.iloc[-1]["close"])
    if (base_max - yesterday_close) / base_max > NEAR_TOP_PCT:
        return None

    # ── Breakout conditions ───────────────────────────────────────────────────
    vol_today  = float(today["volume"])
    vol_avg_20 = float(today["vol_avg_20"])
    if pd.isna(today["vol_avg_20"]) or vol_avg_20 <= 0:
        return None
    vol_ratio = vol_today / vol_avg_20

    is_breakout = (
        price > base_max
        and vol_ratio >= VOL_BREAKOUT_RATIO
        and float(today["bar_range"]) > float(base["bar_range"].mean())
    )
    signal_type = "breakout" if is_breakout else "coiling"

    # ── Shared scoring components ─────────────────────────────────────────────
    stock_ret_20d = (price / float(base_closes.iloc[0])) - 1.0
    today_dt      = pd.Timestamp(today["timestamp"]).date()
    base_start_dt = pd.Timestamp(base.iloc[0]["timestamp"]).date()

    spy_now  = _spy_at(spy, today_dt)
    spy_base = _spy_at(spy, base_start_dt)
    rs_20d   = stock_ret_20d - (spy_now / spy_base - 1.0) if (spy_now and spy_base and spy_base) else stock_ret_20d

    high_52w = float(prior.tail(252)["high"].max()) if len(prior) >= 252 else float(prior["high"].max())
    dist_52w = (price - high_52w) / high_52w   # 0 = at 52w high, negative = below

    base_days = BASE_WINDOW
    for n in (30, 40, 50, 60):
        if len(prior) >= n:
            wider = prior.tail(n)["close"].astype(float)
            w_min, w_max = wider.min(), wider.max()
            if w_min > 0 and (w_max - w_min) / w_min < MAX_BASE_RANGE_PCT:
                base_days = n

    dist_to_breakout = ((base_max - price) / price) * 100.0   # % needed to break out; 0 or neg = broken

    return {
        "SignalType":    signal_type,
        "Symbol":        sym,
        "Price":         round(price, 2),
        "PctFromBase":   round(((price / base_max) - 1.0) * 100.0, 2),
        "DistToBreak":   round(dist_to_breakout, 2),
        "VolumeRatio":   round(vol_ratio, 2),
        "BaseDays":      base_days,
        "BaseTightness": round(range_pct * 100.0, 2),
        "RS_vs_SPY_20d": round(rs_20d * 100.0, 2),
        "Dist52wHigh":   round(dist_52w * 100.0, 2),
        "_vol_ratio": vol_ratio,
        "_range_pct": range_pct,
        "_rs":        rs_20d,
        "_dist_52w":  dist_52w,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank in [0, 1]. ascending=True: higher value -> higher rank."""
    return series.rank(ascending=ascending, pct=True)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Normalize each component to [0, 1]; higher score = better
    s_vol   = _pct_rank(df["_vol_ratio"])                    # higher vol ratio = better
    s_tight = _pct_rank(df["_range_pct"], ascending=False)   # lower range% = tighter base = better
    s_rs    = _pct_rank(df["_rs"])                           # higher RS vs SPY = better
    s_52w   = _pct_rank(df["_dist_52w"])                     # closer to 52w high = better

    df["Score"] = (0.30 * s_vol + 0.25 * s_tight + 0.25 * s_rs + 0.20 * s_52w) * 100
    df["Score"] = df["Score"].round(1)
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    today_str = date.today().strftime("%Y%m%d")
    out_path  = RESULTS_DIR / f"breakouts_{today_str}.csv"

    log.info("=" * 64)
    log.info("Consolidation Breakout Scanner  –  %s", date.today().strftime("%Y-%m-%d"))
    log.info("=" * 64)

    if not ALPACA_KEY or not ALPACA_SECRET:
        log.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    client   = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    universe = get_universe()
    name_map = dict(zip(universe["symbol"], universe["name"]))

    log.info("Fetching SPY for relative-strength baseline ...")
    spy = fetch_spy(client)

    t0 = time.time()
    bars_dict = fetch_all_bars(client, universe)
    log.info("Data ready in %.1fs  (%d symbols)", time.time() - t0, len(bars_dict))

    log.info("Running scanner ...")
    signals: list[dict] = []
    errors = 0
    for sym, raw_df in bars_dict.items():
        try:
            df  = add_indicators(raw_df)
            hit = scan_symbol(sym, df, spy)
            if hit:
                hit["CompanyName"] = name_map.get(sym, "")
                signals.append(hit)
        except Exception as exc:
            log.debug("Error on %s: %s", sym, exc)
            errors += 1

    breakouts = [s for s in signals if s["SignalType"] == "breakout"]
    coiling   = [s for s in signals if s["SignalType"] == "coiling"]
    log.info("Scanned %d symbols  ->  %d breakouts  |  %d coiling setups  (%d errors)",
             len(bars_dict), len(breakouts), len(coiling), errors)

    # ── Build scored DataFrames ───────────────────────────────────────────────
    def build_output(rows: list[dict], price_col: str = "PctFromBase") -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df_r = pd.DataFrame(rows)
        df_s = compute_scores(df_r)
        cols = [
            "Symbol", "CompanyName", "Price", price_col,
            "VolumeRatio", "BaseDays", "BaseTightness",
            "RS_vs_SPY_20d", "Dist52wHigh", "Score",
        ]
        return df_s[[c for c in cols if c in df_s.columns]]

    df_breakouts = build_output(breakouts, price_col="PctFromBase")
    df_coiling   = build_output(coiling,   price_col="DistToBreak")

    # ── Save CSV with both signal types ──────────────────────────────────────
    all_rows = pd.concat([
        df_breakouts.assign(SignalType="breakout"),
        df_coiling.assign(SignalType="coiling"),
    ], ignore_index=True) if (not df_breakouts.empty or not df_coiling.empty) else pd.DataFrame()

    if all_rows.empty:
        print("\nNo signals found today.")
        return

    all_rows.to_csv(out_path, index=False)
    log.info("Saved -> %s", out_path)

    # Ensure stdout handles UTF-8 on Windows
    out = sys.stdout if sys.stdout.encoding and sys.stdout.encoding.lower() in ("utf-8", "utf8") \
          else open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)

    SEP_HEAVY = "=" * 74
    SEP_LIGHT = "-" * 74

    # ── Print breakout signals ────────────────────────────────────────────────
    print(f"\n{SEP_HEAVY}", file=out)
    print(f"  BREAKOUT SIGNALS  --  {date.today().strftime('%A, %B %d %Y')}", file=out)
    print(f"  {len(df_breakouts)} confirmed breakouts", file=out)
    print(SEP_HEAVY, file=out)
    if df_breakouts.empty:
        print("  (none today)", file=out)
    else:
        top_b = df_breakouts.head(TOP_N).copy()
        top_b.insert(0, "#", range(1, len(top_b) + 1))
        top_b["CompanyName"] = top_b["CompanyName"].str[:20]
        print(tabulate(
            top_b,
            headers=["#", "Symbol", "Company", "Price", "%Break", "Vol x",
                     "BaseDays", "Tight%", "RS%", "52w%", "Score"],
            tablefmt="simple",
            floatfmt=(".0f", "s", "s", ".2f", ".2f", ".2f", ".0f", ".2f", ".2f", ".2f", ".1f"),
            showindex=False,
        ), file=out)

    # ── Print coiling watchlist ───────────────────────────────────────────────
    print(f"\n{SEP_LIGHT}", file=out)
    print(f"  COILING SETUPS (watchlist for tomorrow)  --  {len(df_coiling)} candidates", file=out)
    print(SEP_LIGHT, file=out)
    if df_coiling.empty:
        print("  (none today)", file=out)
    else:
        top_c = df_coiling.head(TOP_N).copy()
        top_c.insert(0, "#", range(1, len(top_c) + 1))
        top_c["CompanyName"] = top_c["CompanyName"].str[:20]
        print(tabulate(
            top_c,
            headers=["#", "Symbol", "Company", "Price", "%ToBreak", "Vol x",
                     "BaseDays", "Tight%", "RS%", "52w%", "Score"],
            tablefmt="simple",
            floatfmt=(".0f", "s", "s", ".2f", ".2f", ".2f", ".0f", ".2f", ".2f", ".2f", ".1f"),
            showindex=False,
        ), file=out)

    print(f"\n  Full results -> {out_path}\n", file=out)

    _send_email(df_breakouts, df_coiling, out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Email notification
# ──────────────────────────────────────────────────────────────────────────────

def _send_email(df_breakouts, df_coiling, csv_path):
    """
    Send results to NOTIFY_EMAIL via Gmail SMTP.
    Add to .env to enable:
      GMAIL_USER      = your.address@gmail.com
      GMAIL_APP_PASS  = xxxx xxxx xxxx xxxx   (Gmail App Password, 16 chars)
      NOTIFY_EMAIL    = rmenaar@gmail.com      (recipient; defaults to GMAIL_USER)
    Create App Password: myaccount.google.com -> Security -> 2-Step -> App passwords
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_pass = os.getenv("GMAIL_APP_PASS", "").replace(" ", "").strip()
    notify_to  = os.getenv("NOTIFY_EMAIL", gmail_user).strip()

    if not gmail_user or not gmail_pass:
        log.info("Email skipped — add GMAIL_USER + GMAIL_APP_PASS to .env to enable")
        return

    today_label = date.today().strftime("%A %b %d")
    n_b = len(df_breakouts)
    n_c = len(df_coiling)
    subject = f"Breakout Scanner {today_label} — {n_b} breakout{'s' if n_b!=1 else ''}, {n_c} coiling"

    def _rows_txt(df, n=15):
        if df.empty:
            return "  (none)\n"
        out = [f"  {'#':<3} {'Symbol':<8} {'Price':>8} {'Vol':>6} {'Base%':>7} {'RS%':>6} {'Score':>6}",
               "  " + "-" * 50]
        for i, r in df.head(n).iterrows():
            out.append(f"  {i+1:<3} {r['Symbol']:<8} ${r['Price']:>7.2f}"
                       f"  {r['VolumeRatio']:>5.2f}x  {r['BaseTightness']:>5.1f}%"
                       f"  {r['RS_vs_SPY_20d']:>+5.1f}%  {r['Score']:>5.1f}")
        return "\n".join(out) + "\n"

    def _rows_html(df, n=15):
        if df.empty:
            return "<tr><td colspan='7' style='color:#888;padding:6px'>None today</td></tr>"
        rows = ""
        for i, r in df.head(n).iterrows():
            bg = "#f7f7f7" if i % 2 else "#ffffff"
            rows += (f"<tr style='background:{bg}'>"
                     f"<td style='padding:5px 8px'>{i+1}</td>"
                     f"<td style='padding:5px 8px'><b>{r['Symbol']}</b></td>"
                     f"<td style='padding:5px 8px;text-align:right'>${r['Price']:.2f}</td>"
                     f"<td style='padding:5px 8px;text-align:right'>{r['VolumeRatio']:.2f}x</td>"
                     f"<td style='padding:5px 8px;text-align:right'>{r['BaseTightness']:.1f}%</td>"
                     f"<td style='padding:5px 8px;text-align:right'>{r['RS_vs_SPY_20d']:+.1f}%</td>"
                     f"<td style='padding:5px 8px;text-align:right'><b>{r['Score']:.1f}</b></td></tr>")
        return rows

    th  = "style='background:#1a1a2e;color:#fff;padding:6px 8px;text-align:right'"
    thl = "style='background:#1a1a2e;color:#fff;padding:6px 8px;text-align:left'"
    hdr = (f"<tr><th {thl}>#</th><th {thl}>Symbol</th><th {th}>Price</th>"
           f"<th {th}>Vol</th><th {th}>Base%</th><th {th}>RS%</th><th {th}>Score</th></tr>")

    html = (f"<html><body style='font-family:Arial,sans-serif;font-size:13px;color:#222;max-width:700px'>"
            f"<h2 style='color:#1a1a2e;margin-bottom:4px'>Breakout Scanner — {today_label}</h2>"
            f"<p style='color:#555;margin-top:0'><b>{n_b}</b> breakout(s) &nbsp;|&nbsp; <b>{n_c}</b> coiling</p>"
            f"<h3 style='color:#1a1a2e'>✅ Confirmed Breakouts</h3>"
            f"<table style='border-collapse:collapse;width:100%;font-size:12px'>{hdr}{_rows_html(df_breakouts)}</table>"
            f"<h3 style='color:#1a1a2e;margin-top:20px'>👀 Coiling Watchlist</h3>"
            f"<table style='border-collapse:collapse;width:100%;font-size:12px'>{hdr}{_rows_html(df_coiling)}</table>"
            f"<p style='color:#aaa;font-size:11px;margin-top:16px'>CSV attached | {csv_path}</p>"
            f"</body></html>")

    plain = (f"Breakout Scanner — {today_label}\n"
             f"{n_b} breakout(s) | {n_c} coiling\n\n"
             f"BREAKOUTS:\n{_rows_txt(df_breakouts)}\n"
             f"COILING WATCHLIST:\n{_rows_txt(df_coiling)}\nCSV: {csv_path}\n")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = notify_to
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    if csv_path.exists():
        with open(csv_path, "rb") as f:
            att = MIMEBase("application", "octet-stream")
            att.set_payload(f.read())
            encoders.encode_base64(att)
            att.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
            msg.attach(att)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pass)
            smtp.sendmail(gmail_user, notify_to, msg.as_string())
        log.info("Email sent to %s", notify_to)
    except Exception as e:
        log.warning("Email failed: %s", e)


if __name__ == "__main__":
    main()
