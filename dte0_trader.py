"""
dte0_trader.py — Automated 0DTE SPY Iron Condor Trader (Alpaca Paper)

Strategy:
  - PCS: 0.15-delta put, 10-wide
  - CCS: 0.10-delta call, 5-wide
  - Entry: 10:00 AM ET
  - Exit:  75% take-profit (resting buy-back order) OR 3:00 PM ET force-close
  - Filters: VIX < 17 AND gap < 1% from prior close

Sizing: 25% of account equity per day
  contracts = floor(0.25 * equity / (max_spread_width * 100))
  max_spread_width = 10 (put spread is the wider leg)

Run modes (python dte0_trader.py <mode>):
  entry  — wait to 10:00 ET, apply filters, enter the condor, attach the 75% TP.
  close  — wake ~1h before the close, cancel the TP, flatten anything still open.
  all    — (default) entry then close in one process; for local/manual runs.

In production each mode runs as its own short GitHub Actions job
(.github/workflows/dte0_entry.yml, dte0_close.yml). Splitting them keeps each job
far from GitHub's 6-hour cap and lets the entry fire hours early to absorb the
scheduler's jitter while still entering exactly at the open.
"""

import sys
import os
import time
import math
import logging
import smtplib
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderClass,
    TimeInForce,
    AssetStatus,
    ContractType,
    PositionSide,
)
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# ── config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASS", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", GMAIL_USER)

ET = ZoneInfo("America/New_York")

TICKER           = "SPY"
PUT_DELTA        = 0.15
CALL_DELTA       = 0.10
PUT_WIDTH        = 10.0    # $
CALL_WIDTH       = 5.0     # $
PROFIT_TARGET    = 0.75    # take profit at 75% of credit (resting buy-back order)
# Alpaca has no market order for multi-leg options, so we enter with a MARKETABLE
# limit: priced well below model-mid so it crosses the market and fills immediately
# (a synthetic market order), with a floor to bound worst-case slippage.
ENTRY_LIMIT_HAIRCUT = 0.50 # first limit = model-mid credit × (1-this); 0.50 = aggressive/marketable
ENTRY_CREDIT_FLOOR  = 0.30 # never accept less than this fraction of model-mid credit
ENTRY_CHASE_STEP    = 0.05 # if still unfilled, get more aggressive by $0.05/share per retry
ENTRY_FILL_WAIT     = 20   # seconds to wait for the marketable fill before retrying
VIX_MAX          = 17.0
GAP_MAX          = 0.01    # 1%
ACCOUNT_RISK_PCT = 0.25    # 25% of equity per day
RISK_FREE        = 0.04
ENTRY_TIME       = (10, 0)   # 10:00 AM ET
EXIT_TIME        = (15, 0)   # 3:00 PM ET — 1 hour before the 4:00 PM close
POLL_SECS        = 60        # check P&L every 60 seconds (only used by --mode all)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"dte0_trader_{date.today()}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Alpaca clients ─────────────────────────────────────────────────────────────
trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
opt_data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)

# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_put_price(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_price(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_put_delta(S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1.0

def bs_call_delta(S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def strike_for_put_delta(S, T, r, sigma, target_delta):
    """Find K such that put delta == -target_delta (target_delta positive, e.g. 0.15).

    For 0DTE the target-delta strike sits very close to spot, so the bracket must
    reach right up to spot — a hi of S*0.99 is too far OTM and misses the root.
    """
    def objective(K):
        return bs_put_delta(S, K, T, r, sigma) + target_delta  # want delta == -target
    lo, hi = S * 0.50, S * 0.9999
    try:
        return brentq(objective, lo, hi)
    except ValueError:
        log.warning(f"Put strike solve failed (bracket [{lo:.0f},{hi:.0f}]); using ATM fallback.")
        return S * 0.99

def strike_for_call_delta(S, T, r, sigma, target_delta):
    """Find K such that call delta == target_delta (e.g. 0.10).

    For 0DTE the target-delta strike sits very close to spot, so the bracket must
    start right at spot.
    """
    def objective(K):
        return bs_call_delta(S, K, T, r, sigma) - target_delta
    lo, hi = S * 1.0001, S * 1.50
    try:
        return brentq(objective, lo, hi)
    except ValueError:
        log.warning(f"Call strike solve failed (bracket [{lo:.0f},{hi:.0f}]); using ATM fallback.")
        return S * 1.01

def minutes_to_T(mins_remaining):
    return max(mins_remaining / (252.0 * 390.0), 1e-8)

# ── market data ───────────────────────────────────────────────────────────────

def get_vix():
    """Return today's VIX as a decimal (e.g. 16.5 → 0.165)."""
    try:
        vix = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        close = vix["Close"].squeeze()  # flatten MultiIndex column to Series if needed
        last = float(close.dropna().iloc[-1])
        log.info(f"VIX = {last:.2f}")
        return last / 100.0
    except Exception as e:
        log.error(f"VIX fetch failed: {e}")
        return None

def get_spy_prices():
    """Return (prev_close, current_price) for SPY."""
    try:
        bars = yf.download("SPY", period="5d", interval="1d", auto_adjust=True, progress=False)
        closes = bars["Close"].squeeze().dropna()
        prev_close = float(closes.iloc[-2])
        # For current price, grab the latest 1-min bar
        req = StockBarsRequest(
            symbol_or_symbols=[TICKER],
            timeframe=TimeFrame.Minute,
            start=datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0),
        )
        intraday = data_client.get_stock_bars(req).df
        if intraday.empty:
            current = float(closes.iloc[-1])
        else:
            current = float(intraday["close"].iloc[-1])
        log.info(f"SPY prev_close={prev_close:.2f}  current={current:.2f}")
        return prev_close, current
    except Exception as e:
        log.error(f"SPY price fetch failed: {e}")
        return None, None

def get_account_info():
    """Return (equity, options_buying_power)."""
    acct = trading.get_account()
    equity = float(acct.equity)
    opt_bp = float(getattr(acct, "options_buying_power", None) or acct.buying_power)
    log.info(f"Account equity = ${equity:,.2f}  options_BP = ${opt_bp:,.2f}")
    return equity, opt_bp

# ── option chain helpers ──────────────────────────────────────────────────────

def fetch_option_chain(expiry: date):
    """Return list of option contracts for SPY expiring on expiry (puts + calls)."""
    all_contracts = []
    for ct in (ContractType.PUT, ContractType.CALL):
        req = GetOptionContractsRequest(
            underlying_symbols=[TICKER],
            expiration_date=expiry,
            status=AssetStatus.ACTIVE,
            type=ct,
            limit=1000,
        )
        resp = trading.get_option_contracts(req)
        batch = resp.option_contracts if hasattr(resp, "option_contracts") else list(resp)
        all_contracts.extend(batch)
    return all_contracts

def nearest_strike(chain, target_strike, contract_type: ContractType):
    """Find nearest available strike for the given type."""
    filtered = [c for c in chain if c.type == contract_type]
    if not filtered:
        return None
    return min(filtered, key=lambda c: abs(float(c.strike_price) - target_strike))

def option_symbol(ticker, expiry: date, contract_type: str, strike: float) -> str:
    """Build OCC symbol: SPY240601P00540000"""
    exp_str = expiry.strftime("%y%m%d")
    cp = "C" if contract_type == "call" else "P"
    strike_int = round(strike * 1000)
    return f"{ticker}{exp_str}{cp}{strike_int:08d}"

# ── pricing at runtime ─────────────────────────────────────────────────────────

def spread_cost(S, k_ps, k_pl, k_cs, k_cl, T, sigma):
    """
    Cost to close the iron condor (what we'd pay to buy it back).
    PCS: short k_ps put, long k_pl put  (k_pl < k_ps)
    CCS: short k_cs call, long k_cl call (k_cl > k_cs)
    """
    pcs_cost = bs_put_price(S, k_ps, T, RISK_FREE, sigma) - bs_put_price(S, k_pl, T, RISK_FREE, sigma)
    ccs_cost = bs_call_price(S, k_cs, T, RISK_FREE, sigma) - bs_call_price(S, k_cl, T, RISK_FREE, sigma)
    return max(pcs_cost + ccs_cost, 0.0)

def get_option_quotes(symbols: list) -> dict:
    """Fetch live bid/ask/mid quotes for option symbols. Returns {} on failure."""
    try:
        req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
        raw = opt_data_client.get_option_latest_quote(req)
        result = {}
        for sym, q in raw.items():
            bid = float(q.bid_price or 0)
            ask = float(q.ask_price or 0)
            mid = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else 0.0
            result[sym] = {"bid": bid, "ask": ask, "mid": mid}
        return result
    except Exception as e:
        log.warning(f"Option quote fetch failed: {e}")
        return {}

# ── order placement ───────────────────────────────────────────────────────────

def place_iron_condor(
    put_short_sym, put_long_sym, call_short_sym, call_long_sym, contracts, limit_price
):
    """Submit 4-leg iron condor via REST API (SDK lacks OptionLeg in older versions).

    Alpaca multi-leg orders must be LIMIT orders; limit_price is a positive value
    (the net credit per share we want to collect, e.g. 0.25 for $25/contract).
    """
    import requests as _requests
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "type": "limit",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(contracts),
        "limit_price": f"{limit_price:.2f}",
        "legs": [
            {"symbol": put_short_sym,  "side": "sell", "ratio_qty": "1", "position_intent": "sell_to_open"},
            {"symbol": put_long_sym,   "side": "buy",  "ratio_qty": "1", "position_intent": "buy_to_open"},
            {"symbol": call_short_sym, "side": "sell", "ratio_qty": "1", "position_intent": "sell_to_open"},
            {"symbol": call_long_sym,  "side": "buy",  "ratio_qty": "1", "position_intent": "buy_to_open"},
        ],
    }
    try:
        resp = _requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code >= 400:
            log.error(f"Order rejected ({resp.status_code}): {resp.text}")
            log.error(f"Payload sent: {payload}")
            return None
        order = resp.json()
        log.info(f"Iron condor order submitted: id={order.get('id')}")
        return order
    except Exception as e:
        log.error(f"Order submission failed: {e}")
        return None

def place_take_profit(
    put_short_sym, put_long_sym, call_short_sym, call_long_sym, contracts, debit_limit
):
    """Submit a resting CLOSING mleg limit order = take-profit at PROFIT_TARGET.

    This buys the iron condor back: buy_to_close the two shorts, sell_to_close the
    two longs. It is a NET DEBIT, so limit_price is the positive maximum debit we
    will pay (= PROFIT_TARGET-adjusted fraction of the credit). The order rests
    (TIF=day) and fills the moment the condor can be bought back for <= that debit,
    i.e. once PROFIT_TARGET of the credit has decayed away. Direction (debit) is
    inferred by Alpaca from the legs; limit_price stays positive.
    """
    import requests as _requests
    url = f"{ALPACA_PAPER_BASE}/v2/orders"
    payload = {
        "type": "limit",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(contracts),
        "limit_price": f"{debit_limit:.2f}",
        "legs": [
            {"symbol": put_short_sym,  "side": "buy",  "ratio_qty": "1", "position_intent": "buy_to_close"},
            {"symbol": put_long_sym,   "side": "sell", "ratio_qty": "1", "position_intent": "sell_to_close"},
            {"symbol": call_short_sym, "side": "buy",  "ratio_qty": "1", "position_intent": "buy_to_close"},
            {"symbol": call_long_sym,  "side": "sell", "ratio_qty": "1", "position_intent": "sell_to_close"},
        ],
    }
    try:
        resp = _requests.post(url, json=payload, headers=_alpaca_headers(), timeout=30)
        if resp.status_code >= 400:
            log.error(f"Take-profit order rejected ({resp.status_code}): {resp.text}")
            log.error(f"Payload sent: {payload}")
            return None
        order = resp.json()
        log.info(f"Take-profit (buy-back) order submitted: id={order.get('id')}  "
                 f"debit_limit=${debit_limit:.2f}/share")
        return order
    except Exception as e:
        log.error(f"Take-profit order submission failed: {e}")
        return None

# ── Alpaca REST helpers (SDK multi-leg support is unreliable) ──────────────────
ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"

def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
        "Content-Type": "application/json",
    }

def get_order(order_id):
    import requests as _requests
    r = _requests.get(f"{ALPACA_PAPER_BASE}/v2/orders/{order_id}?nested=true",
                      headers=_alpaca_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def wait_for_fill(order_id, timeout_s=180, poll_s=3):
    """Poll until the entry order fills. Cancel + return None on timeout/terminal."""
    import requests as _requests
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        o = get_order(order_id)
        status = o.get("status")
        if status != last_status:
            log.info(f"Order {order_id} status: {status}")
            last_status = status
        if status == "filled":
            return o
        if status in ("canceled", "rejected", "expired", "done_for_day", "stopped"):
            log.error(f"Entry order terminal without fill: {status}")
            return None
        time.sleep(poll_s)
    # Timed out — cancel, then re-check in case it filled in the race window.
    try:
        _requests.delete(f"{ALPACA_PAPER_BASE}/v2/orders/{order_id}",
                         headers=_alpaca_headers(), timeout=30)
    except Exception as e:
        log.warning(f"Cancel request error: {e}")
    try:
        final = get_order(order_id)
        if final.get("status") == "filled":
            log.info("Order filled in the cancel race window — keeping fill.")
            return final
    except Exception as e:
        log.warning(f"Post-cancel status check failed: {e}")
    log.warning("Entry order not filled within timeout — canceled.")
    return None

def net_credit_from_fill(order):
    """Net credit per share from filled legs: sum(sell fills) - sum(buy fills)."""
    legs = order.get("legs") or []
    if not legs:
        return None
    credit = 0.0
    for leg in legs:
        fp = leg.get("filled_avg_price")
        if fp is None:
            return None  # incomplete fill data — caller falls back to model
        fp = float(fp)
        credit += fp if leg.get("side") == "sell" else -fp
    return credit

def get_positions_map():
    import requests as _requests
    r = _requests.get(f"{ALPACA_PAPER_BASE}/v2/positions",
                      headers=_alpaca_headers(), timeout=30)
    r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}

def close_position_by_symbol(symbol):
    try:
        trading.close_position(symbol)
        log.info(f"Closed position: {symbol}")
    except Exception as e:
        log.warning(f"Close {symbol} failed (may already be closed): {e}")

def close_all_legs(legs):
    for sym in legs:
        close_position_by_symbol(sym)

def list_open_orders():
    """Return all open orders (nested, so mleg legs are visible)."""
    import requests as _requests
    r = _requests.get(f"{ALPACA_PAPER_BASE}/v2/orders?status=open&nested=true",
                      headers=_alpaca_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def cancel_order_by_id(order_id):
    import requests as _requests
    try:
        _requests.delete(f"{ALPACA_PAPER_BASE}/v2/orders/{order_id}",
                         headers=_alpaca_headers(), timeout=30)
        log.info(f"Canceled order {order_id}")
    except Exception as e:
        log.warning(f"Cancel order {order_id} failed (may already be terminal): {e}")

def cancel_open_orders_for(symbols):
    """Cancel any open order that touches one of `symbols` (e.g. the resting TP).

    Must run before flattening: Alpaca rejects close_position while a working
    order exists on that symbol. Only cancels orders whose legs intersect our
    leg set, so other strategies on the account are left untouched.
    """
    sym_set = set(symbols)
    try:
        orders = list_open_orders()
    except Exception as e:
        log.warning(f"Could not list open orders to cancel: {e}")
        return
    for o in orders:
        o_syms = {o.get("symbol")} | {l.get("symbol") for l in (o.get("legs") or [])}
        if o_syms & sym_set:
            cancel_order_by_id(o["id"])

def find_open_condor_legs():
    """For the close job: return option-leg symbols currently held (our SPY 0DTE
    condor), reconstructed from Alpaca positions. Empty if nothing is open
    (e.g. the take-profit already filled)."""
    try:
        pos = get_positions_map()
    except Exception as e:
        log.warning(f"Position fetch failed during close: {e}")
        return []
    today_occ = date.today().strftime("%y%m%d")
    legs = []
    for sym, p in pos.items():
        # OCC option symbols look like SPY260604P00540000; match our ticker + today's expiry.
        if (p.get("asset_class") == "us_option" or sym.startswith(TICKER)) and today_occ in sym:
            legs.append(sym)
    return legs

# ── notification ──────────────────────────────────────────────────────────────

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS:
        log.info("Email not configured, skipping notification.")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = GMAIL_USER
        msg["To"] = NOTIFY_EMAIL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [NOTIFY_EMAIL], msg.as_string())
        log.info(f"Email sent: {subject}")
    except Exception as e:
        log.error(f"Email failed: {e}")

# ── wait until target time ────────────────────────────────────────────────────

def wait_until(hour, minute, tz=ET):
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        return
    secs = (target - now).total_seconds()
    log.info(f"Waiting {secs/60:.1f} min until {hour:02d}:{minute:02d} ET ...")
    time.sleep(max(secs - 2, 0))

def et_now():
    return datetime.now(ET)

# ── entry phase ────────────────────────────────────────────────────────────────

def run_entry():
    """ENTRY job: run filters, enter the iron condor at market open, and attach a
    resting take-profit (buy-back) order at PROFIT_TARGET. Returns a context dict
    on success (used by --mode all to chain into the close), or None if no
    position was taken. This job is short, so its workflow can be scheduled well
    before the open to absorb GitHub's scheduler jitter."""
    log.info("=" * 60)
    log.info("0DTE Iron Condor Trader — ENTRY phase")
    today = date.today()
    log.info(f"Date: {today}  Weekday: {today.strftime('%A')}")

    # ── 0. Wait for entry time ──────────────────────────────────────────────────
    # The entry workflow fires hours early (to outrun GitHub's scheduler jitter),
    # so hold here until 10:00 ET. Running the VIX/gap filters AFTER this means they
    # evaluate live data at entry time, not stale pre-market data.
    wait_until(*ENTRY_TIME)

    # ── 1. Pre-checks ──────────────────────────────────────────────────────────
    vix_decimal = get_vix()
    if vix_decimal is None:
        send_email("0DTE Trader: SKIP (VIX fetch error)", f"Could not fetch VIX on {today}. No trade placed.")
        return

    vix_level = vix_decimal * 100
    if vix_level >= VIX_MAX:
        log.info(f"VIX {vix_level:.2f} >= {VIX_MAX} — skipping today.")
        send_email(
            f"0DTE Trader: SKIP (VIX {vix_level:.1f})",
            f"VIX={vix_level:.1f} exceeds threshold {VIX_MAX}. No trade placed on {today}."
        )
        return

    prev_close, current_price = get_spy_prices()
    if prev_close is None or current_price is None:
        send_email("0DTE Trader: SKIP (price fetch error)", "Could not fetch SPY prices. No trade placed.")
        return

    gap_pct = abs(current_price - prev_close) / prev_close
    if gap_pct >= GAP_MAX:
        log.info(f"Gap {gap_pct*100:.2f}% >= {GAP_MAX*100:.0f}% — high-move day, skipping.")
        send_email(
            f"0DTE Trader: SKIP (gap {gap_pct*100:.1f}%)",
            f"SPY gapped {gap_pct*100:.1f}% from prior close. High-move day filter triggered. No trade on {today}."
        )
        return

    log.info(f"Filters passed: VIX={vix_level:.1f}  gap={gap_pct*100:.2f}%")

    # ── 2. Fetch the entry price (already at/after the open) ─────────────────────
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[TICKER],
            timeframe=TimeFrame.Minute,
            start=et_now().replace(second=0, microsecond=0) - timedelta(minutes=2),
        )
        bars = data_client.get_stock_bars(req).df
        S = float(bars["close"].iloc[-1]) if not bars.empty else current_price
    except Exception:
        S = current_price
    log.info(f"Entry SPY price: {S:.2f}")

    equity, options_bp = get_account_info()

    # ── 3. Strike calculation ──────────────────────────────────────────────────
    sigma = vix_decimal
    # Real minutes from now until the 4:00 PM ET close (0DTE expiry), floored at 5.
    now_et = et_now()
    mins_to_close = max((16 * 60) - (now_et.hour * 60 + now_et.minute), 5)
    T_entry = minutes_to_T(mins_to_close)
    log.info(f"Minutes to close: {mins_to_close}  (T={T_entry:.6f})")

    k_put_short_ideal  = strike_for_put_delta(S, T_entry, RISK_FREE, sigma, PUT_DELTA)
    k_put_long_ideal   = k_put_short_ideal - PUT_WIDTH
    k_call_short_ideal = strike_for_call_delta(S, T_entry, RISK_FREE, sigma, CALL_DELTA)
    k_call_long_ideal  = k_call_short_ideal + CALL_WIDTH

    log.info(f"Ideal strikes: PS={k_put_short_ideal:.2f}  PL={k_put_long_ideal:.2f}  "
             f"CS={k_call_short_ideal:.2f}  CL={k_call_long_ideal:.2f}")

    # ── 4. Option chain — find real strikes ───────────────────────────────────
    log.info("Fetching option chain...")
    chain = fetch_option_chain(today)
    if not chain:
        log.error("Empty option chain — cannot trade.")
        send_email("0DTE Trader: ERROR (empty chain)", f"No option contracts returned for {today}.")
        return

    log.info(f"Chain returned {len(chain)} contracts. Sample types: {list({str(c.type) for c in chain[:20]})}")

    # Normalise type comparison — Alpaca may return string or enum
    def _is_type(contract, ct: ContractType):
        t = contract.type
        return t == ct or str(t).upper() in (ct.value.upper(), ct.name.upper())

    def nearest_strike_safe(ch, target, ct):
        filtered = [c for c in ch if _is_type(c, ct)]
        log.info(f"  {ct.value} contracts available: {len(filtered)}")
        if not filtered:
            return None
        return min(filtered, key=lambda c: abs(float(c.strike_price) - target))

    ps_contract = nearest_strike_safe(chain, k_put_short_ideal,  ContractType.PUT)
    pl_contract = nearest_strike_safe(chain, k_put_long_ideal,   ContractType.PUT)
    cs_contract = nearest_strike_safe(chain, k_call_short_ideal, ContractType.CALL)
    cl_contract = nearest_strike_safe(chain, k_call_long_ideal,  ContractType.CALL)

    if not all([ps_contract, pl_contract, cs_contract, cl_contract]):
        log.error("Could not find all 4 legs in chain.")
        send_email("0DTE Trader: ERROR (chain lookup)", "Could not find all 4 option legs.")
        return

    k_ps = float(ps_contract.strike_price)
    k_pl = float(pl_contract.strike_price)
    k_cs = float(cs_contract.strike_price)
    k_cl = float(cl_contract.strike_price)

    # Use actual symbols from chain
    ps_sym = ps_contract.symbol
    pl_sym = pl_contract.symbol
    cs_sym = cs_contract.symbol
    cl_sym = cl_contract.symbol

    log.info(f"Actual strikes: PS={k_ps}  PL={k_pl}  CS={k_cs}  CL={k_cl}")
    log.info(f"Symbols: {ps_sym}  {pl_sym}  {cs_sym}  {cl_sym}")

    # ── 5. Size the position ──────────────────────────────────────────────────
    max_width = max(k_ps - k_pl, k_cl - k_cs)
    max_loss_per_contract = max_width * 100
    risk_contracts = math.floor((ACCOUNT_RISK_PCT * equity) / max_loss_per_contract)
    # Cap by actual options buying power. Alpaca's margin runs ~2% above our max-loss
    # estimate, so use a 10% buffer to stay clear of "insufficient buying power".
    bp_contracts = math.floor((options_bp * 0.90) / max_loss_per_contract)
    contracts = min(risk_contracts, bp_contracts)
    log.info(f"max_width={max_width:.2f}  max_loss/contract=${max_loss_per_contract:.0f}  "
             f"risk_cap={risk_contracts}  bp_cap={bp_contracts}  → contracts={contracts}")
    if contracts < 1:
        log.error(f"Insufficient buying power for even 1 contract "
                  f"(need ${max_loss_per_contract:.0f}, have ${options_bp:.2f}).")
        send_email("0DTE Trader: SKIP (no buying power)",
                   f"Options BP ${options_bp:.2f} < ${max_loss_per_contract:.0f} needed for 1 contract. "
                   f"Check for leftover open positions on {today}.")
        return

    # ── 6. Calculate entry credit (live quotes, BS as fallback) ──────────────
    bs_entry_credit = spread_cost(S, k_ps, k_pl, k_cs, k_cl, T_entry, sigma)
    log.info(f"BS model estimate: ${bs_entry_credit:.4f}/share  (${bs_entry_credit*100:.2f}/contract)")

    entry_quotes = get_option_quotes([ps_sym, pl_sym, cs_sym, cl_sym])
    if all(entry_quotes.get(s, {}).get("mid", 0) > 0 for s in [ps_sym, pl_sym, cs_sym, cl_sym]):
        market_mid_credit = ((entry_quotes[ps_sym]["mid"] - entry_quotes[pl_sym]["mid"]) +
                             (entry_quotes[cs_sym]["mid"] - entry_quotes[cl_sym]["mid"]))
        log.info(f"Live market mid credit: ${market_mid_credit:.4f}/share  "
                 f"(${market_mid_credit*100:.2f}/contract)")
        entry_credit = market_mid_credit
    else:
        log.warning("Some option quotes missing/zero — falling back to BS estimate for entry pricing.")
        market_mid_credit = None
        entry_credit = bs_entry_credit

    # ── 7. Enter via a MARKETABLE limit (Alpaca has no market order for mleg) ────
    # With live quotes: start $0.05 below market mid for a fast fill.
    # With BS fallback: use 50% haircut to make the order marketable.
    # Chase down to ENTRY_CREDIT_FLOOR if not filled on first try.
    legs = [ps_sym, pl_sym, cs_sym, cl_sym]
    if market_mid_credit is not None:
        start_credit = max(round(market_mid_credit - 0.05, 2), 0.05)
        floor_credit = max(round(market_mid_credit * ENTRY_CREDIT_FLOOR, 2), 0.05)
    else:
        start_credit = max(round(entry_credit * (1 - ENTRY_LIMIT_HAIRCUT), 2), 0.05)
        floor_credit = max(round(entry_credit * ENTRY_CREDIT_FLOOR, 2), 0.05)
    attempt = start_credit
    filled = None
    while attempt >= floor_credit - 1e-9:
        log.info(f"Submitting IC limit: net credit >= ${attempt:.2f}/share")
        order = place_iron_condor(ps_sym, pl_sym, cs_sym, cl_sym, contracts, attempt)
        if order is None:
            send_email("0DTE Trader: ORDER FAILED", f"Order submission error on {today}. Check logs.")
            return
        filled = wait_for_fill(order["id"], timeout_s=ENTRY_FILL_WAIT, poll_s=3)
        if filled:
            break
        attempt = round(attempt - ENTRY_CHASE_STEP, 2)
        if attempt >= floor_credit - 1e-9:
            log.info(f"Not filled — chasing credit down to ${attempt:.2f}/share")

    if filled is None:
        send_email("0DTE Trader: NOT FILLED",
                   f"Iron condor did not fill on {today} down to ${floor_credit:.2f}/share credit. "
                   f"No position taken.")
        return

    actual_credit = net_credit_from_fill(filled)
    if actual_credit is None or actual_credit <= 0:
        log.warning(f"Could not read net credit from fill (got {actual_credit}); "
                    f"falling back to model estimate ${entry_credit:.4f}/share.")
        actual_credit = entry_credit

    credit_dollars = actual_credit * 100 * contracts
    log.info(f"FILLED. Net credit ${actual_credit:.4f}/share  (${credit_dollars:.2f} total).")

    # ── 8. Attach take-profit: resting buy-back at PROFIT_TARGET of the credit ───
    # 50% profit ⇒ buy the condor back once it can be closed for (1-PROFIT_TARGET)
    # of the credit. limit_price is the positive max debit we will pay.
    tp_debit = max(round(actual_credit * (1 - PROFIT_TARGET), 2), 0.01)
    tp_order = place_take_profit(ps_sym, pl_sym, cs_sym, cl_sym, contracts, tp_debit)
    if tp_order is None:
        log.warning("Take-profit order failed to submit — position is OPEN with no "
                    "resting TP. The close job will still flatten at 3:00 PM ET.")
    else:
        log.info(f"Take-profit resting at ${tp_debit:.2f}/share debit "
                 f"(= {PROFIT_TARGET*100:.0f}% of ${actual_credit:.2f} credit).")

    # ── 9. Entry email ──────────────────────────────────────────────────────────
    mkt_credit_line = (
        f"Mkt mid credit: ${market_mid_credit*100:.2f}/contract  "
        f"(BS est: ${bs_entry_credit*100:.2f}/contract)\n"
        if market_mid_credit is not None else
        f"BS est. credit:  ${bs_entry_credit*100:.2f}/contract  "
        f"(live quotes unavailable)\n"
    )
    tp_line = (
        f"Take-profit:  buy-back @ ${tp_debit*100:.2f}/contract "
        f"({PROFIT_TARGET*100:.0f}% target)  [{'resting' if tp_order else 'FAILED — see logs'}]\n"
    )
    entry_email_body = (
        f"Iron condor FILLED on {today}\n\n"
        f"SPY @ {S:.2f}\n"
        f"VIX: {vix_level:.1f}\n"
        f"Contracts: {contracts}\n\n"
        f"Put spread:  {k_ps:.0f}/{k_pl:.0f}  [{ps_sym}]\n"
        f"Call spread: {k_cs:.0f}/{k_cl:.0f}  [{cs_sym}]\n\n"
        + mkt_credit_line
        + f"Net credit:  ${actual_credit*100:.2f}/contract  "
        f"(${credit_dollars:.2f} total)\n"
        + tp_line
        + f"Max risk:    ${max_loss_per_contract*contracts:.2f}\n"
    )
    send_email(f"0DTE Trader: ENTERED ({contracts} contracts)", entry_email_body)

    return {
        "legs": legs,
        "contracts": contracts,
        "actual_credit": actual_credit,
        "credit_dollars": credit_dollars,
        "tp_debit": tp_debit,
        "today": today,
    }

# ── close phase ────────────────────────────────────────────────────────────────

def run_close():
    """CLOSE job: wake ~1 hour before the 4:00 PM ET close. Cancel the resting
    take-profit and flatten whatever is still open. If the take-profit already
    filled, there is nothing to do. State is reconstructed from Alpaca positions,
    so this runs as a fully independent job from the entry."""
    log.info("=" * 60)
    log.info("0DTE Iron Condor Trader — CLOSE phase")
    today = date.today()

    # Hold until the close time in case the runner fired early (jitter buffer).
    wait_until(*EXIT_TIME)

    legs = find_open_condor_legs()
    if not legs:
        log.info("No open condor legs found — take-profit likely already filled.")
        send_email(
            "0DTE Trader: CLOSE (nothing open)",
            f"No open SPY 0DTE legs at {EXIT_TIME[0]:02d}:{EXIT_TIME[1]:02d} ET on {today}. "
            f"The {PROFIT_TARGET*100:.0f}% take-profit either filled earlier or no trade was placed."
        )
        log.info("Done.")
        return

    log.info(f"{len(legs)} open legs at close time: {legs}")

    # Mark-to-market P&L before flattening (Alpaca tracks unrealized_pl per leg).
    try:
        pos = get_positions_map()
        last_unreal = sum(float(pos[s].get("unrealized_pl", 0.0)) for s in legs if s in pos)
    except Exception as e:
        log.warning(f"Could not read unrealized P&L before close: {e}")
        last_unreal = float("nan")

    # Cancel the resting take-profit first — Alpaca rejects close while a working
    # order exists on the symbol.
    cancel_open_orders_for(legs)
    time.sleep(2)

    log.info(f"Force-closing {len(legs)} legs: {legs}")
    close_all_legs(legs)
    time.sleep(5)  # brief pause for closing fills

    pnl_str = f"${last_unreal:+.2f}" if last_unreal == last_unreal else "unknown"
    exit_email = (
        f"0DTE Trade force-closed — {today}\n\n"
        f"Exit reason: time (3:00 PM ET)\n"
        f"Est. P&L (mark-to-market): {pnl_str}\n\n"
        f"Legs flattened: {len(legs)}\n"
    )
    log.info(f"Trade force-closed at time stop. Est. P&L={pnl_str}")
    send_email(f"0DTE Trader: CLOSED (time) est. {pnl_str}", exit_email)
    log.info("Done.")

# ── dispatch ────────────────────────────────────────────────────────────────────

def run_all():
    """Manual end-to-end run: enter at open, then hold (the resting TP may fill on
    its own) until the close job flattens any remainder. Used for local/manual
    runs; production uses the two scheduled jobs (entry, close) instead."""
    ctx = run_entry()
    if ctx is None:
        log.info("No position taken — skipping close phase.")
        return
    run_close()

def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "all")
    if mode == "entry":
        run_entry()
    elif mode == "close":
        run_close()
    elif mode == "all":
        run_all()
    else:
        log.error(f"Unknown mode '{mode}'. Use one of: entry | close | all")
        sys.exit(2)


if __name__ == "__main__":
    main()
