# runner_ib_price.py
from datetime import datetime, timedelta, timezone
import time
# from ib_insync import IB, Stock, Option, util
from ib_async import IB, Stock, Option, MarketOrder, LimitOrder, Contract, ContractDetails, util
import asyncio
from typing import List, Dict, Optional, Tuple
import math
import pandas as pd



from commons import mid, yang_zhang, build_term_structure, filter_dates, nearest_strike_contract

MINIMUM_VOLUME = 1_500_000
MINIMUM_IV_RV_RATIO = 1.25
MAXIMUM_TERM_STRUCTURE_SLOPE = -0.00406
# --- paste your get_underlying_price_ib(...) here or import it ---
# from yourmodule import get_underlying_price_ib

async def _get_underlying_price(
    ib: IB,
    ticker: str,
    exchange: str = "SMART",
    timeout_sec: float = 3.0
) -> Optional[float]:
    """
    Returns mid(bid,ask), falling back to last or close, else None.
    No historical fallback; hard timeouts at each step.
    """
    # 1) Resolve a fully-qualified contract (includes primaryExchange)
    try:
        cds = await asyncio.wait_for(
            ib.reqContractDetailsAsync(Stock(ticker, exchange, "USD")),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        return None

    if not cds:
        return None

    contract = cds[0].contract
    # 2) Qualify (guarantees a tradable/quoteable conId)
    try:
        [contract] = await asyncio.wait_for(
            ib.qualifyContractsAsync(contract),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        return None

    # 3) Snapshot request (one-shot). Give it a moment to populate fields.
    try:
        tk = await asyncio.wait_for(
            ib.reqMktDataAsync(contract, genericTickList="", snapshot=True),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        return None

    try:
        await asyncio.sleep(0.3)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not any([tk.bid, tk.ask, tk.last, tk.close]):
            # If data is delayed/live, ticks may trickle in
            await ib.waitOnUpdateAsync(timeout=0.25)

        # 4) Pick the best available price
        if tk.bid and tk.ask and tk.bid > 0 and tk.ask > 0:
            return float((tk.bid + tk.ask) / 2.0)
        if tk.last and tk.last > 0:
            return float(tk.last)
        if tk.close and tk.close > 0:
            return float(tk.close)
        return None
    finally:
        # For snapshots this is usually unnecessary, but safe to cancel
        ib.cancelMktData(contract)

  
                
def compute_recommendation(ib, ticker, max_expiries=6):
    print(f"Computing recommendation for {ticker}")
    try:
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return "No stock symbol provided."

        # 1) Get current stock price
        spot = _get_underlying_price(ib, symbol)  # assumes your IB version
        print(f"1. Underlying price = {spot}")
        if spot is None:
            return "Error: unable to retrieve stock price"
        
        # 2) Expirations -> Filter -> cap
        expirations = _list_expirations(ib, symbol)  # returns e.g. ['YYYY-MM-DD', ...]
        if not expirations:
            return f"Error: no options found for symbol {symbol}"
        try:
            exp_dates = filter_dates(expirations)
        except Exception:
            return "Error: not enough option data"
        exp_dates = exp_dates[:max_expiries]
        print(f"2. Retrieved expiration dates within the next 45 days")
        
        # 3) For each expiry, choose ATM call/put and fetch bid/ask and IV
        atm_iv = {}
        straddle_mid = None
        for i, exp in enumerate(exp_dates):
            print(f"3. {i}, {exp}")
            contracts = _list_contracts_for_expiry(ib, symbol, exp)  
            if not contracts:
                continue

            call_ctr = nearest_strike_contract(contracts, spot, "call")
            put_ctr  = nearest_strike_contract(contracts, spot, "put")

            print(f"Nearest strike contract for {exp}", call_ctr)

            if not call_ctr or not put_ctr:
                continue

            print(f"Found nearest strike (in abs val) contracts for exp {exp}")
            # ✅ updated calls: (ib, symbol, expiry, strike, right)
            c_bid, c_ask, c_iv = _get_option_quote_greeks(ib, symbol, exp, call_ctr["strike"], "C")
            p_bid, p_ask, p_iv = _get_option_quote_greeks(ib, symbol, exp, put_ctr["strike"], "P")

            print("Implied volatilties", c_iv, p_iv)

            # compute mids for earliest expiry for the straddle
            if i == 0:
                c_mid = mid(c_bid, c_ask)
                p_mid = mid(p_bid, p_ask)
                if c_mid is not None and p_mid is not None:
                    straddle_mid = c_mid + p_mid
            if c_iv is not None and p_iv is not None:
                atm_iv[exp] = (c_iv + p_iv) / 2.0;
            
            print(f"Implied volatilities", atm_iv)
            # Be polite to rate limits
            time.sleep(0.08)
        if not atm_iv:
            print("errror 1`")
            return "Error: Could not determin ATM IV for any expiration dates"
        print(f"3. atm_iv={atm_iv}")

        # 4) Build term structure spline and slope
        today = datetime.now(timezone.utc).date()
        dtes, ivs = [], []
        for exp, iv in atm_iv.items():
            d=datetime.strptime(exp, "%Y-%m-%d").date()
            dtes.append((d-today).days)
            ivs.append(float(iv))

        if len(dtes)<2:
            return "Error: Not enough expirations to build term structure."
        term_spline = build_term_structure(dtes,ivs);
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45-min(dtes))

        print(f"4. ts_slope_0_45=${ts_slope_0_45}")

        # 5) Daily OHLCV (~3 months) for Yang-Zhang + avg vol
        price_history = _get_stock_history_df(ib, symbol, days=100)
        if price_history.empty:
            return "Error: no historical data"
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        avg_volume = price_history["Volume"].rolling(30).mean().dropna().iloc[-1]
        expected_move = f"{round(straddle_mid / spot * 100, 2)}%" if straddle_mid else None

        print("")
        if avg_volume>=MINIMUM_VOLUME:
            print(f"GREEN.  Avg volume of {round(avg_volume)} exceeds the minimum of {MINIMUM_VOLUME}")
        else:
            print(f"RED.  Avg volume of {round(avg_volume)} is below the minimum of {MINIMUM_VOLUME}")

        if iv30_rv30 >= MINIMUM_IV_RV_RATIO:
            print(f"GREEN. iv-to-rv of {iv30_rv30} exceeds the minimum of {MINIMUM_IV_RV_RATIO}")
        else:
            print(f"RED. iv_to_rv of {iv30_rv30} is below the minimum of {MINIMUM_IV_RV_RATIO}")

        if ts_slope_0_45<=MAXIMUM_TERM_STRUCTURE_SLOPE:
            print(f"GREEN.  Term structure slope of {ts_slope_0_45} falls below the maximum of {MAXIMUM_TERM_STRUCTURE_SLOPE}")
        else:
            print(f"RED.  Term structure slope of {ts_slope_0_45} exceeds the maximum of {MAXIMUM_TERM_STRUCTURE_SLOPE}")
        print("")

        resultPackage = {
                "avg_volume" : avg_volume>=MINIMUM_VOLUME,
                "iv30_rv30" : iv30_rv30 >= MINIMUM_IV_RV_RATIO,
                "ts_slope_0_45" : ts_slope_0_45 <= MAXIMUM_TERM_STRUCTURE_SLOPE,
                "expected_move": expected_move
            }
            
        print(resultPackage)

        return resultPackage;
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"{ticker}: Processing failed: {e}"

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
        ib.waitOnUpdate(timeout=1)  # wait for real ticks

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


async def testGetUnderlyingPrice():
    print("test get underlying price")
    symbol = "AMZN"
    ib = await connect_ib_client()
    price = await _get_underlying_price(ib, symbol)
    print(f"{symbol} price:", price)
    await disconnect_ib_client(ib)

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

async def testConnection():
    ib = await connect_ib_client();
    await disconnect_ib_client(ib)

async def connect_ib_client(client_id=43):
    ib = IB()
    # Connect to Gateway live first
    await ib.connectAsync("127.0.0.1", 4001, clientId=client_id, timeout=8)

    # Try LIVE data
    #ib.reqMarketDataType(1)  # 1 = real-time
    ib.reqMarketDataType(3)  # 3 = delayed data
    
    return ib
   
async def disconnect_ib_client(ib):
    ib.disconnect()

if __name__ == "__main__":
    symbol = "AMZN"
    print(f"starting test")
    ib = asyncio.run(connect_ib_client())
    print(f"client retrieved")
    price = asyncio.run(_get_underlying_price(ib, symbol))
    print("retrieved price")
    print(f"{symbol} price:", price)
    asyncio.run(disconnect_ib_client(ib))
    print("disconnected client")
    # asyncio.run(testConnection())
    # asyncio.run(testGetUnderlyingPrice())
    # testGetGreeks()
    # ib = connect_ib_client()
    # compute_recommendation(ib, "AMZN")
    # disconnect_ib_client(ib)