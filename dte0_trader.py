"""
dte0_trader.py — Automated 0DTE SPY Iron Condor Trader (Alpaca Paper)

Strategy (from backtest):
  - PCS: 0.15-delta put, 10-wide
  - CCS: 0.10-delta call, 5-wide
  - Entry: 10:00 AM ET
  - Exit:  75% profit target OR 3:00 PM ET force-close
  - Filters: VIX < 17 AND gap < 1% from prior close

Sizing: 25% of account equity per day
  contracts = floor(0.25 * equity / (max_spread_width * 100))
  max_spread_width = 10 (put spread is the wider leg)
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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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
PROFIT_TARGET    = 0.75    # close when 75% of credit collected
VIX_MAX          = 17.0
GAP_MAX          = 0.01    # 1%
ACCOUNT_RISK_PCT = 0.25    # 25% of equity per day
RISK_FREE        = 0.04
ENTRY_TIME       = (10, 0)   # 10:00 AM ET
EXIT_TIME        = (15, 0)   # 3:00 PM ET
POLL_SECS        = 60        # check P&L every 60 seconds

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
    """Find K such that put delta == -target_delta (target_delta positive, e.g. 0.15)."""
    def objective(K):
        return bs_put_delta(S, K, T, r, sigma) + target_delta  # want delta == -target
    lo, hi = S * 0.50, S * 0.99
    try:
        return brentq(objective, lo, hi)
    except ValueError:
        return S * (1 - target_delta * 0.5)

def strike_for_call_delta(S, T, r, sigma, target_delta):
    """Find K such that call delta == target_delta (e.g. 0.10)."""
    def objective(K):
        return bs_call_delta(S, K, T, r, sigma) - target_delta
    lo, hi = S * 1.001, S * 1.50
    try:
        return brentq(objective, lo, hi)
    except ValueError:
        return S * (1 + target_delta * 0.5)

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

def get_account_equity():
    acct = trading.get_account()
    equity = float(acct.equity)
    log.info(f"Account equity = ${equity:,.2f}")
    return equity

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

# ── order placement ───────────────────────────────────────────────────────────

def place_iron_condor(
    put_short_sym, put_long_sym, call_short_sym, call_long_sym, contracts
):
    """Submit 4-leg iron condor via REST API (SDK lacks OptionLeg in older versions)."""
    import requests as _requests
    url = "https://paper-api.alpaca.markets/v2/orders"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "type": "market",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(contracts),
        "legs": [
            {"symbol": put_short_sym,  "side": "sell", "ratio_qty": "1"},
            {"symbol": put_long_sym,   "side": "buy",  "ratio_qty": "1"},
            {"symbol": call_short_sym, "side": "sell", "ratio_qty": "1"},
            {"symbol": call_long_sym,  "side": "buy",  "ratio_qty": "1"},
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

def close_position_by_symbol(symbol):
    try:
        trading.close_position(symbol)
        log.info(f"Closed position: {symbol}")
    except Exception as e:
        log.warning(f"Close {symbol} failed (may already be closed): {e}")

def close_all_legs(legs):
    for sym in legs:
        close_position_by_symbol(sym)

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

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("0DTE Iron Condor Trader — starting")
    today = date.today()
    log.info(f"Date: {today}  Weekday: {today.strftime('%A')}")

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

    # ── 2. Wait for entry ──────────────────────────────────────────────────────
    wait_until(*ENTRY_TIME)

    # Re-fetch current price at entry time
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

    equity = get_account_equity()

    # ── 3. Strike calculation ──────────────────────────────────────────────────
    sigma = vix_decimal
    # Minutes from 10:00 to 4:00 PM close = 360 min
    T_entry = minutes_to_T(360)

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
    contracts = max(1, math.floor((ACCOUNT_RISK_PCT * equity) / max_loss_per_contract))
    log.info(f"max_width={max_width:.2f}  max_loss/contract=${max_loss_per_contract:.0f}  contracts={contracts}")

    # ── 6. Calculate entry credit (BS model) ─────────────────────────────────
    entry_credit = spread_cost(S, k_ps, k_pl, k_cs, k_cl, T_entry, sigma)
    log.info(f"Estimated entry credit: ${entry_credit:.4f}/share  (${entry_credit*100:.2f}/contract)")

    # ── 7. Place order ────────────────────────────────────────────────────────
    order = place_iron_condor(ps_sym, pl_sym, cs_sym, cl_sym, contracts)
    if order is None:
        send_email("0DTE Trader: ORDER FAILED", f"Order submission failed for {today}. Check logs.")
        return

    legs = [ps_sym, pl_sym, cs_sym, cl_sym]
    entry_email_body = (
        f"Iron condor placed on {today}\n\n"
        f"SPY @ {S:.2f}\n"
        f"VIX: {vix_level:.1f}\n"
        f"Contracts: {contracts}\n\n"
        f"Put spread:  {k_ps:.0f}/{k_pl:.0f}  [{ps_sym}]\n"
        f"Call spread: {k_cs:.0f}/{k_cl:.0f}  [{cs_sym}]\n\n"
        f"Est. credit: ${entry_credit*100*contracts:.2f} total\n"
        f"Max risk:    ${max_loss_per_contract*contracts:.2f}\n"
    )
    send_email(f"0DTE Trader: ENTERED ({contracts} contracts)", entry_email_body)

    # ── 8. Monitor loop ───────────────────────────────────────────────────────
    profit_target_cost = entry_credit * (1 - PROFIT_TARGET)  # 25% of credit = close here
    log.info(f"Monitoring until 3PM or profit target (cost <= ${profit_target_cost:.4f}/share)")

    exit_reason = "time"
    final_cost = None

    while True:
        now = et_now()

        # Force close at EXIT_TIME
        if (now.hour, now.minute) >= EXIT_TIME:
            log.info("3:00 PM — force closing position.")
            exit_reason = "time"
            break

        # Current cost to close
        mins_remaining = (15 * 60 + 0) - (now.hour * 60 + now.minute)  # mins until 3 PM
        T_now = minutes_to_T(max(mins_remaining, 1))

        # Re-fetch SPY price
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[TICKER],
                timeframe=TimeFrame.Minute,
                start=now.replace(second=0, microsecond=0) - timedelta(minutes=2),
            )
            bars = data_client.get_stock_bars(req).df
            S_now = float(bars["close"].iloc[-1]) if not bars.empty else S
        except Exception:
            S_now = S

        current_cost = spread_cost(S_now, k_ps, k_pl, k_cs, k_cl, T_now, sigma)
        pnl_per_share = entry_credit - current_cost
        pnl_total = pnl_per_share * 100 * contracts
        pct_profit = pnl_per_share / entry_credit if entry_credit > 0 else 0

        log.info(
            f"  {now.strftime('%H:%M')}  SPY={S_now:.2f}  cost={current_cost:.4f}  "
            f"PnL=${pnl_total:.2f}  ({pct_profit*100:.1f}%)"
        )

        if current_cost <= profit_target_cost:
            log.info(f"Profit target hit: {pct_profit*100:.1f}% — closing.")
            exit_reason = "profit_target"
            final_cost = current_cost
            break

        time.sleep(POLL_SECS)

    # ── 9. Close legs ──────────────────────────────────────────────────────────
    close_all_legs(legs)
    time.sleep(3)  # brief pause for fill

    # Estimate final P&L
    if final_cost is None:
        now = et_now()
        mins_remaining = max((15 * 60) - (now.hour * 60 + now.minute), 1)
        T_close = minutes_to_T(mins_remaining)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[TICKER],
                timeframe=TimeFrame.Minute,
                start=now.replace(second=0, microsecond=0) - timedelta(minutes=2),
            )
            bars = data_client.get_stock_bars(req).df
            S_close = float(bars["close"].iloc[-1]) if not bars.empty else S
        except Exception:
            S_close = S
        final_cost = spread_cost(S_close, k_ps, k_pl, k_cs, k_cl, T_close, sigma)

    final_pnl = (entry_credit - final_cost) * 100 * contracts
    exit_email = (
        f"0DTE Trade closed — {today}\n\n"
        f"Exit reason: {exit_reason}\n"
        f"Estimated P&L: ${final_pnl:.2f}\n\n"
        f"Entry credit:  ${entry_credit*100:.2f}/contract\n"
        f"Exit cost:     ${final_cost*100:.2f}/contract\n"
        f"Contracts:     {contracts}\n"
    )
    log.info(f"Trade closed. Reason={exit_reason}  Est. P&L=${final_pnl:.2f}")
    send_email(
        f"0DTE Trader: CLOSED ({exit_reason}) est. ${final_pnl:+.2f}",
        exit_email
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
