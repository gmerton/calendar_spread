import os

from typing import Optional, List, Dict, Any
import os, asyncio
import aiohttp
from commons import mid, yang_zhang, build_term_structure, filter_dates, nearest_strike_contract
from datetime import datetime, timedelta, timezone
import time
import math
import pandas as pd

TRADIER_BASE = "https://api.tradier.com/v1"
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
MINIMUM_VOLUME = 1_500_000
MINIMUM_IV_RV_RATIO = 1.25
MAXIMUM_TERM_STRUCTURE_SLOPE = -0.00406

def _tradier_headers():
    return {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}

async def _get_underlying_price(
    ticker: str,
    *,
    timeout_sec: float = 4.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> Optional[float]:
    """
    Fetches a stock quote from Tradier and returns a single float price:
      - mid(bid, ask) if both > 0
      - else last
      - else close/prevclose
      - else None

    Requires env var TRADIER_TOKEN.
    """
    
    base = TRADIER_BASE
    url = f"{base}/markets/quotes"
    params = {"symbols": ticker}
    headers = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}

    # Use provided session or create a short-lived one
    close_session = False
    if session is None:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        close_session = True
    else:
        # Merge headers without clobbering existing
        session.headers.update(headers)

    try:
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        q = (data or {}).get("quotes", {}).get("quote")
        if q is None:
            return None
        if isinstance(q, list):
            q = q[0]

        bid = q.get("bid")
        ask = q.get("ask")
        last = q.get("last")
        close = q.get("close") or q.get("prevclose")

        if bid and ask and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        if last and last > 0:
            return float(last)
        if close and close > 0:
            return float(close)
        return None
    finally:
        if close_session:
            await session.close()

async def _list_expirations(
    symbol: str,
    *,
    include_all_roots: bool = True,
    timeout_sec: float = 6.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[str]:
    """Return sorted list of YYYY-MM-DD expirations from Tradier /markets/options/expirations."""
    close_session = False
    if session is None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
            headers=_tradier_headers()
        )
        close_session = True
    else:
        session.headers.update(_tradier_headers())

    try:
        url = f"{TRADIER_BASE}/markets/options/expirations"
        params = {
            "symbol": symbol,
            "includeAllRoots": "true" if include_all_roots else "false",
            "strikes": "false",
            "contractSize": "false",
            "expirationType": "false",
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        exp_root = (data or {}).get("expirations") or {}
        dates: List[str] = []

        maybe_dates = exp_root.get("date")
        if maybe_dates:
            dates = [str(d) for d in (maybe_dates if isinstance(maybe_dates, list) else [maybe_dates])]
        else:
            exps = exp_root.get("expiration")
            if exps:
                if isinstance(exps, dict):
                    exps = [exps]
                dates = [str(x.get("date")) for x in exps if x.get("date")]

        return sorted(set(filter(None, dates)))
    finally:
        if close_session:
            await session.close()



async def _list_contracts_for_expiry(
    symbol: str,
    expiration: str,                  # 'YYYY-MM-DD'
    *,
    option_type: Optional[str] = None,  # 'call' | 'put' | None (both)
    include_greeks: bool = True,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    timeout_sec: float = 8.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[Dict[str, Any]]:
    """
    Return a normalized list of option contracts for the given symbol+expiration
    from Tradier /markets/options/chains.

    Each item includes:
      - symbol (OCC option symbol)
      - option_type ('call' or 'put')
      - strike (float)
      - expiration_date (YYYY-MM-DD)
      - root_symbol, underlying
      - bid, ask, last, volume, open_interest, bid_size, ask_size
      - greeks (dict) if include_greeks=True and available
    """
    close_session = False
    if session is None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
            headers=_tradier_headers()
        )
        close_session = True
    else:
        session.headers.update(_tradier_headers())

    try:
        url = f"{TRADIER_BASE}/markets/options/chains"
        params = {
            "symbol": symbol,
            "expiration": expiration,
            "greeks": "true" if include_greeks else "false",
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        raw = (data or {}).get("options", {}).get("option")
        if not raw:
            return []

        # Normalize to list
        options = raw if isinstance(raw, list) else [raw]

        # Optional filters
        if option_type in ("call", "put"):
            options = [o for o in options if o.get("option_type") == option_type]

        if min_strike is not None:
            options = [o for o in options if o.get("strike") is not None and float(o["strike"]) >= min_strike]
        if max_strike is not None:
            options = [o for o in options if o.get("strike") is not None and float(o["strike"]) <= max_strike]

        # Normalize fields we commonly care about
        out: List[Dict[str, Any]] = []
        for o in options:
            out.append({
                "symbol": o.get("symbol"),
                "option_type": o.get("option_type"),                 # 'call' | 'put'
                "strike": float(o["strike"]) if o.get("strike") is not None else None,
                "expiration_date": o.get("expiration_date"),
                "root_symbol": o.get("root_symbol"),
                "underlying": o.get("underlying"),
                "bid": o.get("bid"),
                "ask": o.get("ask"),
                "last": o.get("last"),
                "volume": o.get("volume"),
                "open_interest": o.get("open_interest"),
                "bid_size": o.get("bid_size"),
                "ask_size": o.get("ask_size"),
                "greeks": o.get("greeks") if include_greeks else None,
            })

        # Sort by strike, then calls before puts (or vice versa if you prefer)
        out.sort(key=lambda x: (x["strike"] if x["strike"] is not None else float("inf"),
                                0 if x["option_type"] == "call" else 1))
        return out
    finally:
        if close_session:
            await session.close()

def _to_ymd(d: Optional[object]) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date().strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    # assume already 'YYYY-MM-DD' string
    return str(d)

async def _get_stock_history_df(
    symbol: str,
    *,
    start: Optional[object] = None,     # datetime/date/'YYYY-MM-DD'
    end: Optional[object] = None,       # datetime/date/'YYYY-MM-DD'
    interval: str = "daily",            # 'daily' | 'weekly' | 'monthly'
    timeout_sec: float = 10.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV for `symbol` from Tradier /markets/history.

    Returns a DataFrame with columns: ['date','open','high','low','close','volume', ('vwap' if provided)]
    Sorted ascending by date. Empty DataFrame if no data.
    """
    if interval not in {"daily", "weekly", "monthly"}:
        raise ValueError("interval must be one of: 'daily', 'weekly', 'monthly'")

    params = {
        "symbol": symbol,
        "interval": interval,
    }
    s = _to_ymd(start)
    e = _to_ymd(end)
    if s: params["start"] = s
    if e: params["end"] = e

    close_session = False
    if session is None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
            headers=_tradier_headers()
        )
        close_session = True
    else:
        session.headers.update(_tradier_headers())

    try:
        url = f"{TRADIER_BASE}/markets/history"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # Tradier shape: {"history":{"day":[ {...}, ... ]}} or a single object.
        hist = (data or {}).get("history") or {}
        days = hist.get("day")
        if not days:
            return pd.DataFrame(columns=["date","open","high","low","close","volume"])

        if isinstance(days, dict):
            days = [days]

        # Normalize to rows
        rows = []
        for d in days:
            rows.append({
                "date": d.get("date"),
                "open": d.get("open"),
                "high": d.get("high"),
                "low":  d.get("low"),
                "close": d.get("close"),
                "volume": d.get("volume"),
                # Some responses include vwap; include if present
                "vwap": d.get("vwap"),
            })

        df = pd.DataFrame(rows)
        # Coerce types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("open","high","low","close","vwap"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

        # Drop entirely-missing vwap column if not provided
        if "vwap" in df.columns and df["vwap"].isna().all():
            df = df.drop(columns=["vwap"])

        df = df.sort_values("date").reset_index(drop=True)
        return df
    finally:
        if close_session:
            await session.close()

async def compute_recommendation(ticker, max_expiries=6):
    print(f"Computing recommendation for {ticker}")
    try:
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return "No stock symbol provided."

        # 1) Get current stock price
        spot = await _get_underlying_price(symbol)  # assumes your IB version
        print(f"1. Underlying price = {spot}")
        if spot is None:
            return "Error: unable to retrieve stock price"
        
        # 2) Expirations -> Filter -> cap
        expirations = await _list_expirations(symbol)  # returns e.g. ['YYYY-MM-DD', ...]
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
            contracts = await _list_contracts_for_expiry(symbol, exp)  
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

async def test():
    # price = await _get_underlying_price_tradier("AMZN")
    # print(price)
    # expirations = await _list_expirations("AMZN")
    # print(expirations)
    # contracts = await _list_contracts_for_expiry("AMZN", "2025-09-05")
    # print(contracts)
    df = await _get_stock_history_df("AMZN")
    print(df.head())
    
    


if __name__ == "__main__":
    asyncio.run(test())