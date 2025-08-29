# runner_ib_price.py
from ib_insync import IB
from datetime import datetime
from ib_insync import Stock, Option, util
from typing import List, Dict, Optional, Tuple
import math
# --- paste your get_underlying_price_ib(...) here or import it ---
# from yourmodule import get_underlying_price_ib

def _get_underlying_price(ib: IB, ticker: str, exchange: str = "SMART", timeout_sec: float = 2.0) -> Optional[float]:
    from ib_insync import Stock
    contract = Stock(ticker, exchange, "USD")
    ib.qualifyContracts(contract)
    tk = ib.reqMktData(contract, snapshot=True)
    ib.sleep(2)
    deadline = util.time.time() + timeout_sec
    while util.time.time() < deadline and not any([tk.bid, tk.ask, tk.last, tk.close]):
        ib.sleep(0.1)

    if tk.bid and tk.ask and tk.bid > 0 and tk.ask > 0:
        return float((tk.bid + tk.ask) / 2.0)
    if tk.last and tk.last > 0:
        return float(tk.last)
    if tk.close and tk.close > 0:
        return float(tk.close)

    bars = ib.reqHistoricalData(
        contract, endDateTime=datetime.now(), durationStr="2 D",
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=False, formatDate=1
    )
    if bars:
        return float(bars[-1].close)
    return None

from typing import List, Dict
from ib_insync import IB, Stock

def _list_contracts_for_expiry(ib: IB, ticker: str, expiry: str, exchange: str = "SMART") -> List[Dict]:
    """
    Return minimal info for all contracts at a given expiry using IBKR metadata.

    Args:
        ib: connected IB() instance
        ticker: underlying symbol, e.g. 'AMZN'
        expiry: 'YYYY-MM-DD' or 'YYYYMMDD' (IB also supports monthly 'YYYYMM')
        exchange: typically 'SMART'

    Returns: [{'ticker': <OCC>, 'strike': <float>, 'type': 'call'|'put'}, ...]
    Notes:
      - OCC root is up to 6 chars; for >6-char underlyings we best-effort pad/truncate.
      - No market data required; just contract definitions.
    """
    # Normalize expiry to IB format (YYYYMMDD or YYYYMM)
    e = expiry.replace("-", "")
    if len(e) not in (6, 8):
        raise ValueError(f"Unexpected expiry format: {expiry}. Use YYYY-MM-DD or YYYYMMDD (or YYYYMM).")

    # Qualify underlying to get its conId
    und = Stock(ticker, exchange, "USD")
    ib.qualifyContracts(und)

    # Get option params (expirations/strikes/multiplier/etc.)
    params = ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
    chosen = next((p for p in params if p.exchange == exchange), None) or (params[0] if params else None)
    if not chosen:
        return []

    # Validate the requested expiry exists
    if e not in chosen.expirations:
        # Sometimes IB returns monthlies (YYYYMM) alongside dailies; allow a monthly match if given
        has_close = any(exp.startswith(e) or e.startswith(exp) for exp in chosen.expirations)
        if not has_close:
            return []

    strikes = sorted(float(s) for s in chosen.strikes)

    def occ_symbol(root: str, yyyymmdd_or_mm: str, right: str, strike: float) -> str:
        """
        Build OCC 21-char (or 15-char for some venues) style like:
          'AMZN  250829C00232500'
        Format: ROOT(<=6, space-padded) + YYMMDD + C/P + 8-digit strike (3 decimals implied)
        For monthly codes 'YYYYMM' we approximate '01' as the day.
        """
        # root up to 6 chars, right- or left-pad with spaces so total is exactly 6
        r = (root[:6]).ljust(6)
        yyyymmdd = yyyymmdd_or_mm if len(yyyymmdd_or_mm) == 8 else (yyyymmdd_or_mm + "01")
        y, m, d = yyyymmdd[:4], yyyymmdd[4:6], yyyymmdd[6:8]
        yymmdd = y[2:] + m + d
        strike_int = int(round(strike * 1000))  # 3 implied decimals
        strike_field = f"{strike_int:08d}"
        return f"{r}{yymmdd}{right}{strike_field}"

    contracts: List[Dict] = []
    for k in strikes:
        # Call
        contracts.append({
            "ticker": occ_symbol(ticker, e, "C", k),
            "strike": k,
            "type": "call"
        })
        # Put
        contracts.append({
            "ticker": occ_symbol(ticker, e, "P", k),
            "strike": k,
            "type": "put"
        })

    return contracts




def connect_ib_client(client_id=42):
    ib = IB()
    # Connect to Gateway live first
    ib.connect("127.0.0.1", 4001, clientId=client_id, timeout=8)

    # Try LIVE data
    #ib.reqMarketDataType(1)  # 1 = real-time
    ib.reqMarketDataType(1)  # 3 = delayed data
    
    return ib
    
    # price = get_underlying_price_ib(ib, symbol)
    #if price is None:
        # Fall back to DELAYED if you lack live subs
    #    ib.reqMarketDataType(3)  # 3 = delayed
    #    price = get_underlying_price_ib(ib, symbol)

    # print(f"{symbol} price:", price)
    # ib.disconnect()

def disconnect_ib_client(ib):
    ib.disconnect()





def _get_option_quote_greeks_2(
    ib: IB,
    symbol: str,                  # e.g. "AMZN"
    expiry: str,                  # "YYYYMMDD" or "YYYY-MM-DD"
    strike: float,                # e.g. 232.5
    right: str,                   # "C" or "P"
    exchange: str = "SMART",
    timeout_sec: float = 6.0,
    cancel_after: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Live OPRA path: returns (bid, ask, iv) without fixed sleeps.
    Requires:
      ib.reqMarketDataType(1) and an OPRA subscription.
    """

    # Normalize expiry to YYYYMMDD
    exp = expiry.replace("-", "")
    if len(exp) not in (6, 8):
        raise ValueError(f"Unexpected expiry format: {expiry} (use YYYYMMDD or YYYY-MM-DD)")

    # 1) Build & qualify the contract
    opt = Option(symbol, exp, float(strike), right.upper(), exchange)
    ib.qualifyContracts(opt)

    # 2) Start streaming with option computations
    tk = ib.reqMktData(opt, genericTickList="106", snapshot=False)

    # 3) Event-driven wait: update on *this* ticker only
    bid = ask = iv = None

    def clean(x):
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    def on_update(updated_tk):
        nonlocal bid, ask, iv
        # Quotes
        if bid is None:
            bid = clean(updated_tk.bid)
        if ask is None:
            ask = clean(updated_tk.ask)
        # Greeks buckets: prefer model IV, but accept others if that’s what arrives first
        if iv is None:
            if getattr(updated_tk, "modelGreeks", None) and updated_tk.modelGreeks.impliedVol is not None:
                iv = float(updated_tk.modelGreeks.impliedVol)
            elif getattr(updated_tk, "bidGreeks", None) and updated_tk.bidGreeks.impliedVol is not None:
                iv = float(updated_tk.bidGreeks.impliedVol)
            elif getattr(updated_tk, "askGreeks", None) and updated_tk.askGreeks.impliedVol is not None:
                iv = float(updated_tk.askGreeks.impliedVol)
            elif getattr(updated_tk, "lastGreeks", None) and updated_tk.lastGreeks.impliedVol is not None:
                iv = float(updated_tk.lastGreeks.impliedVol)
            elif getattr(updated_tk, "closeGreeks", None) and updated_tk.closeGreeks.impliedVol is not None:
                iv = float(updated_tk.closeGreeks.impliedVol)

    # attach handler
    tk.updateEvent += on_update

    try:
        # Wait until we have *something useful* or we time out
        ib.waitUntil(lambda: (bid is not None) or (ask is not None) or (iv is not None),
                     timeout=timeout_sec)
    finally:
        # Always detach handler
        tk.updateEvent -= on_update
        if cancel_after:
            ib.cancelMktData(opt)

    return bid, ask, iv

def _get_option_quote_greeks(
    ib: IB,
    symbol: str,                  # e.g. "AMZN"
    expiry: str,                  # "YYYYMMDD" or "YYYY-MM-DD"
    strike: float,                # e.g. 232.5
    right: str,                   # "C" or "P"
    exchange: str = "SMART",
    timeout_sec: float = 6.0,
    cancel_after: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Live OPRA path: returns (bid, ask, iv).
    Requires ib.reqMarketDataType(1) and an OPRA subscription.
    Waits on actual updates (ib.waitOnUpdate) instead of fixed sleeps.
    """
    exp = expiry.replace("-", "")
    if len(exp) not in (6, 8):
        raise ValueError(f"Unexpected expiry format: {expiry}")

    # Build & qualify
    opt = Option(symbol, exp, float(strike), right.upper(), exchange)
    ib.qualifyContracts(opt)

    # Stream market data with option computations
    tk = ib.reqMktData(opt, genericTickList="106", snapshot=False)
    ib.sleep(2)
    def clean(x):
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    deadline = util.time.time() + timeout_sec
    bid = ask = iv = None

    while util.time.time() < deadline:
        ib.waitOnUpdate(timeout=0.25)  # wait for real ticks

        # Quotes
        if bid is None: bid = clean(tk.bid)
        if ask is None: ask = clean(tk.ask)

        # IV candidates from all greeks buckets
        iv_candidates = {}
        if getattr(tk, "modelGreeks", None) and tk.modelGreeks.impliedVol is not None:
            iv_candidates["model"] = tk.modelGreeks.impliedVol
        if getattr(tk, "bidGreeks", None) and tk.bidGreeks.impliedVol is not None:
            iv_candidates["bid"] = tk.bidGreeks.impliedVol
        if getattr(tk, "askGreeks", None) and tk.askGreeks.impliedVol is not None:
            iv_candidates["ask"] = tk.askGreeks.impliedVol
        if getattr(tk, "lastGreeks", None) and tk.lastGreeks.impliedVol is not None:
            iv_candidates["last"] = tk.lastGreeks.impliedVol
        if getattr(tk, "closeGreeks", None) and tk.closeGreeks.impliedVol is not None:
            iv_candidates["close"] = tk.closeGreeks.impliedVol

        # Choose IV: prefer model; else mid of bid/ask; else any one available
        if iv is None:
            if "model" in iv_candidates:
                iv = float(iv_candidates["model"])
            elif "bid" in iv_candidates and "ask" in iv_candidates:
                iv = float((iv_candidates["bid"] + iv_candidates["ask"]) / 2.0)
            else:
                for k in ("last", "close", "bid", "ask"):
                    if k in iv_candidates:
                        iv = float(iv_candidates[k])
                        break

        # Return as soon as we have either side of the market OR IV
        if bid is not None or ask is not None or iv is not None:
            break

    if cancel_after:
        ib.cancelMktData(opt)

    return bid, ask, iv


import pandas as pd
from datetime import datetime
from typing import Optional
from ib_insync import IB, Stock, util

def _get_stock_history_df(
    ib: IB,
    ticker: str,
    days: int = 100,
    exchange: str = "SMART",
    what_to_show: str = "TRADES",  # or "MIDPOINT" etc.
    rth: bool = False              # include pre/post if False
) -> pd.DataFrame:
    """
    Fetch ~N days of daily OHLCV for `ticker` via IBKR and return a DataFrame:
    index: DatetimeIndex, columns: ['Open','High','Low','Close','Volume'].
    """
    # Cap at 365 in a single request (IB limit for 1-day bars)
    duration = f"{min(days, 365)} D"

    # Qualify the underlying
    contract = Stock(ticker, exchange, "USD")
    ib.qualifyContracts(contract)

    # Request historical bars
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=datetime.now(),   # now
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what_to_show,
        useRTH=rth,
        formatDate=1,
        keepUpToDate=False
    )

    if not bars:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    df = util.df(bars)  # columns: date, open, high, low, close, volume, etc.
    # Normalize column names & index
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low":  "Low",
        "close":"Close",
        "volume":"Volume",
    }, inplace=True)

    df.set_index("Date", inplace=True)
    df = df[["Open","High","Low","Close","Volume"]].sort_index()

    # If you prefer an explicitly timezone-aware index, uncomment:
    # df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    return df



def _list_expirations(ib: IB, ticker: str, exchange: str = "SMART") -> List[str]:
    """
    Get a sorted list of option expiration dates for an underlying using IBKR.
    Returns ISO strings like 'YYYY-MM-DD'.
    Notes:
      - IB sometimes returns monthly codes 'YYYYMM' for certain markets.
        We normalize those to 'YYYY-MM-01'.
      - Requires only contract metadata (no live market-data subscription).
    """
    # Qualify the underlying first
    underlying = Stock(ticker, exchange, "USD")
    ib.qualifyContracts(underlying)

    # Query option security-definition params
    params = ib.reqSecDefOptParams(
        underlying.symbol, "", underlying.secType, underlying.conId
    )

    # Prefer the requested exchange (e.g., SMART); fall back to the first available
    chosen = next((p for p in params if p.exchange == exchange), None) or (params[0] if params else None)
    if not chosen:
        return []

    def _fmt(exp: str) -> str:
        # IB gives 'YYYYMMDD' (typical) or 'YYYYMM' (monthly)
        if len(exp) == 8:
            return f"{exp[0:4]}-{exp[4:6]}-{exp[6:8]}"
        if len(exp) == 6:
            return f"{exp[0:4]}-{exp[4:6]}-01"  # normalize monthly to first of month
        return exp  # unexpected format; return as-is

    # Deduplicate, format, and sort
    exps = sorted({_fmt(e) for e in chosen.expirations})
    return exps


def testGetUnderlyingPrice():
    print("test get underlying price")
    symbol = "AMZN"
    ib = connect_ib_client()
    price = _get_underlying_price(ib, symbol)
    print(f"{symbol} price:", price)
    disconnect_ib_client(ib)

def testListExpirations():
    print("test expirations")
    symbol = "AMZN"
    ib = connect_ib_client()
    expirations = _list_expirations(ib, symbol)
    print(expirations)
    disconnect_ib_client(ib)

def testGetStockHistory():
    print("Testing stock history")
    ib = connect_ib_client()
    df = _get_stock_history_df(ib, "AMZN", days=100)
    print(df.tail())
    disconnect_ib_client(ib)

def testGetGreeks():
    print("Testing greeks")
    ib = connect_ib_client()
    bid, ask, iv = _get_option_quote_greeks(ib, "AMZN", "20250829", 232.5, "C", timeout_sec=4.0)
    print("Bid:", bid, "Ask:", ask, "IV:", iv)
    disconnect_ib_client(ib)



if __name__ == "__main__":
    
    testGetGreeks()
    # symbol = "AMZN"
    # ib = connect_ib_client()
    # price = _get_underlying_price(ib, symbol)
    # print(f"{symbol} price:", price)
    # expirations = _list_expirations(ib, symbol)
    # print(expirations)
    
    # contracts = _list_contracts_for_expiry(ib, "AMZN", "2025-09-05")
    # print(contracts[:6])  # first few rows: [{'ticker': 'AMZN  250829C00....', 'strike': ..., 'type': 'call'}, ...]

    # call_tkr = "AMZN  250905C00230000"
    # put_tkr  = "AMZN  250905P00230000"

    # bid, ask, iv = _get_option_quote_greeks(ib, "AMZN", "20250829", 232.5, "C", timeout_sec=4.0)
    # print("Bid:", bid, "Ask:", ask, "IV:", iv)
    # bid, ask, iv = _get_option_quote_greeks(ib, "AMZN  250829C00232500")
    # print(bid, ask, iv)

    # df = _get_stock_history_df(ib, "AMZN", days=100)
    # print(df.tail())

    # disconnect_ib_client(ib)