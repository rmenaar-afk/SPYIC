#!/usr/bin/env python3
"""
histdata_downloader.py
──────────────────────
Downloads 1-minute OHLCV bar data from histdata.com for the requested
symbols and stores each year as a CSV file ready for backtesting.

Symbols available on histdata.com (confirmed):
  EURUSD, GBPUSD, USDJPY, XAUUSD, SPXUSD (S&P 500)

NOT available on histdata.com:
  US30 / Dow Jones (DJUSD) — no data exists on the site

Output structure:
  data/histdata/
    EURUSD/
      EURUSD_M1_2015.csv
      EURUSD_M1_2016.csv
      ...
    SPXUSD/
      SPXUSD_M1_2015.csv
      ...

CSV column format (semicolon-separated, no header row):
  DateTime(YYYYMMDD HHMMSS) ; Open ; High ; Low ; Close ; Volume

Usage:
  python histdata_downloader.py
  python histdata_downloader.py --symbols EURUSD GBPUSD --years 2022 2023 2024
"""

import argparse
import re
import sys
import time
import zipfile
import io
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found.")
    print("Install it with:  pip install requests")
    sys.exit(1)

# ── Default configuration ─────────────────────────────────────────────────────

# User-friendly names → histdata.com symbol codes
SYMBOL_MAP = {
    "US500":  "SPXUSD",
    "SP500":  "SPXUSD",
    "SPX":    "SPXUSD",
    "SPXUSD": "SPXUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "XAUUSD": "XAUUSD",
    "GOLD":   "XAUUSD",
    # Add more as needed — full list at histdata.com
}

NOT_AVAILABLE = {"US30", "DJ30", "DJUSD", "NAS100"}  # known missing symbols

DEFAULT_SYMBOLS = ["US500", "EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
DEFAULT_YEARS   = list(range(2015, 2025))   # 2015 – 2024 inclusive
TIMEFRAME       = "M1"
OUTPUT_DIR      = Path("data/histdata")
DELAY_SEC       = 2.5    # polite pause between requests (be kind to the server)

BASE_URL  = "https://www.histdata.com"
PAGE_URL  = BASE_URL + "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{symbol}/{year}"
POST_URL  = BASE_URL + "/get.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": BASE_URL + "/",
    "Accept":  "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Core download logic ───────────────────────────────────────────────────────

def get_token(hd_symbol: str, year: int, session: requests.Session):
    """
    GET the histdata page and extract the hidden 'tk' token required for
    the POST download request.  Returns (token_str, referer_url) or (None, None).
    """
    url = PAGE_URL.format(symbol=hd_symbol, year=year)
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:
        print(f"    [ERR] Network error fetching page: {exc}")
        return None, None

    if resp.status_code != 200:
        print(f"    [WARN] HTTP {resp.status_code} for {url}")
        return None, None

    match = re.search(
        r'<input[^>]+name=["\']tk["\'][^>]+value=["\']([a-f0-9]+)["\']',
        resp.text,
    )
    if not match:
        # Symbol might not exist on histdata.com
        print(f"    [WARN] No download token found — {hd_symbol}/{year} may not exist on histdata.com")
        return None, None

    return match.group(1), resp.url


def download_year(hd_symbol: str, year: int, session: requests.Session, out_dir: Path) -> bool:
    """
    Download one year of M1 data for hd_symbol.
    Returns True on success (including skip if file already exists).
    """
    out_file = out_dir / f"{hd_symbol}_M1_{year}.csv"

    if out_file.exists() and out_file.stat().st_size > 1024:
        print(f"    [SKIP] {out_file.name} already on disk")
        return True

    # Step 1 — get token
    tk, referer = get_token(hd_symbol, year, session)
    if not tk:
        return False

    # Step 2 — POST to download ZIP
    payload = {
        "tk":        tk,
        "date":      str(year),
        "datemonth": str(year),
        "platform":  "ASCII",
        "timeframe": TIMEFRAME,
        "fxpair":    hd_symbol,
    }
    dl_headers = {**HEADERS, "Referer": referer}

    try:
        resp = session.post(POST_URL, data=payload, headers=dl_headers,
                            stream=True, timeout=120)
    except requests.RequestException as exc:
        print(f"    [ERR] Network error downloading ZIP: {exc}")
        return False

    if resp.status_code != 200:
        print(f"    [WARN] POST returned HTTP {resp.status_code}")
        return False

    content = resp.content
    if len(content) < 2048:
        print(f"    [WARN] Response too small ({len(content)} bytes) — likely not a ZIP")
        return False

    # Step 3 — extract CSV from ZIP (in memory, no temp file needed)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist()
                         if n.lower().endswith((".csv", ".txt"))
                         and not n.startswith("__MACOSX")]
            if not csv_names:
                print(f"    [WARN] No CSV/TXT inside ZIP for {hd_symbol}/{year}")
                return False
            csv_data = zf.read(csv_names[0])
    except zipfile.BadZipFile:
        print(f"    [WARN] Bad ZIP received for {hd_symbol}/{year}")
        return False

    out_file.write_bytes(csv_data)
    size_mb = out_file.stat().st_size / 1_048_576
    print(f"    [OK]   {out_file.name}  ({size_mb:.1f} MB)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download histdata.com 1-min Forex/Index data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        metavar="SYM",
                        help="Symbols to download (default: US500 EURUSD GBPUSD USDJPY XAUUSD)")
    parser.add_argument("--years",   nargs="+", type=int, default=DEFAULT_YEARS,
                        metavar="YYYY",
                        help="Years to download (default: 2015–2024)")
    parser.add_argument("--outdir",  default=str(OUTPUT_DIR),
                        help="Output directory (default: data/histdata)")
    args = parser.parse_args()

    out_root = Path(args.outdir)

    # ── Resolve and deduplicate symbols ──────────────────────────────────────
    jobs = {}   # hd_symbol → display_name
    skipped = []
    for sym in args.symbols:
        sym_upper = sym.upper()
        if sym_upper in NOT_AVAILABLE:
            skipped.append(sym_upper)
            continue
        hd = SYMBOL_MAP.get(sym_upper, sym_upper)
        if hd not in jobs:
            jobs[hd] = sym_upper

    print("=" * 60)
    print("  histdata.com 1-Minute Data Downloader")
    print("=" * 60)
    print(f"  Symbols   : {', '.join(f'{v} -> {k}' for k, v in jobs.items())}")
    if skipped:
        print(f"  SKIPPED   : {', '.join(skipped)} (not available on histdata.com)")
    print(f"  Years     : {min(args.years)} - {max(args.years)}")
    print(f"  Output    : {out_root.resolve()}")
    print(f"  Downloads : {len(jobs) * len(args.years)} files")
    print("=" * 60)

    total_ok = 0
    total_fail = 0

    with requests.Session() as session:
        for hd_sym, display_name in jobs.items():
            sym_dir = out_root / hd_sym
            sym_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{display_name} -> {hd_sym}]")

            for year in sorted(args.years):
                ok = download_year(hd_sym, year, session, sym_dir)
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
                time.sleep(DELAY_SEC)

    print("\n" + "=" * 60)
    print(f"  Done - {total_ok} files saved, {total_fail} failed/skipped")
    print(f"  Data folder: {out_root.resolve()}")
    if total_fail:
        print("  Re-run the script to retry failed files (already-downloaded files are skipped).")
    print("=" * 60)


if __name__ == "__main__":
    main()
