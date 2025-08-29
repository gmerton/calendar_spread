# Issue is we can't get bid and asks without a $199 sub.  Still a chance with Tradier or IBKR

import os
import time
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from commons import yang_zhang, build_term_structure, filter_dates, nearest_strike_contract

POLYGON_API_KEY=os.getenv("POLYGON_API_KEY")
BASE_V3 = "https://api.polygon.io/v3"
BASE_V2 = "https://api.polygon.io/v2"


# Wrapper for REST API call
def _get(url, params=None):
    if params is None:
        params = {}
    params["apiKey"]=POLYGON_API_KEY
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _get_option_quote_greeks(underlying, option_ticker):
    bid = ask = iv = None
    try:
        s= _get(f"{BASE_V3}/snapshot/options/{underlying.upper()}/{option_ticker}")
        res = s.get("results") or {}
        print(f"{BASE_V3}/snapshot/options/{underlying.upper()}/{option_ticker}")
        print(s)
        if res is None:
            print("res is none")
        print("res")
        print(res)
        iv = res.get("implied_volatility")
        bid = res.get("last_quote").get("bid")
        ask = res.get("last_quote").get("ask")
      
        print(iv)
        print(bid)
        print(ask)
        
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        iv  = float(iv)  if iv  is not None else None
        
        sys.exit()
    except Exception as e:
        print(e)
        sys.exit()
    print(f"iv for {option_ticker}={iv}, bid={bid}, ask={ask}")
    return bid, ask, iv

"""
def _get_option_quote_greeks(underlying: str, option_ticker: str):
   
    bid = ask = iv = None

    # A) IV via single-contract snapshot (preferred)
    try:
        s = _get(f"{BASE_V3}/snapshot/options/{underlying.upper()}/{option_ticker}")
        res = s.get("results") or {}
        # IV is exposed as top-level 'implied_volatility' OR inside 'greeks'
        iv = res.get("implied_volatility")
        bid = res.get("last_quote").get("bid")
        ask = res.get("last_quote").get("ask")
        if iv is None:
            iv = (res.get("greeks") or {}).get("implied_volatility")


    
    except Exception as e:
        print(f"[snapshot] {option_ticker}: {e}")
        # Fallback: some older accounts may accept the contract-only path
        try:
            s = _get(f"{BASE_V3}/snapshot/options/{option_ticker}")
            res = s.get("results") or {}
            iv = iv or res.get("implied_volatility") or (res.get("greeks") or {}).get("implied_volatility")
        except Exception as ee:
            print(f"[snapshot fallback] {option_ticker}: {ee}")
    # B) NBBO via quotes/last (with underlying in path), then fallback
     try:
        q = _get(f"{BASE_V3}/quotes/options/{underlying}/{option_ticker}/last")
        r = q.get("results") or {}
        bid = r.get("bid_price") or (r.get("bid") or {}).get("price")
        ask = r.get("ask_price") or (r.get("ask") or {}).get("price")
    except Exception as e:
        print(f"[quotes/last] {option_ticker}: {e}")
        try:
            q = _get(f"{BASE_V3}/quotes/options/{option_ticker}/last")
            r = q.get("results") or {}
            bid = bid or r.get("bid_price") or (r.get("bid") or {}).get("price")
            ask = ask or r.get("ask_price") or (r.get("ask") or {}).get("price")
        except Exception as ee:
            print(f"[quotes/last fallback] {option_ticker}: {ee}")

    # normalize
    bid = float(bid) if bid is not None else None
    ask = float(ask) if ask is not None else None
    iv  = float(iv)  if iv  is not None else None
    print(f"iv for {option_ticker}={iv}, bid={bid}, ask={ask}")
    return bid, ask, iv
"""




def _mid(bid, ask):
    return (bid+ask)/2.0 if (bid and ask and bid>0 and ask>0) else None

def compute_recommendation(ticker, max_expiries=6):
    print(f"Computing recommendation for {ticker}")
    try:
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return "No stock symbol provided."
        # 1) Get current stock price
        spot = _get_underlying_price(symbol)
        print(f"1. Underlying price = {spot}")
        if spot is None:
            return "Error: unable to retrieve stock price"
        
        # 2) Expirations -> Filter -> cap
        
        expirations = _list_expirations(symbol)
        if not expirations:
            return f"Error: no options found for symbol {symbol}"
        try:
            exp_dates = filter_dates(expirations)
        except Exception:
            return "Error: not enough option data"
        exp_dates = exp_dates[:max_expiries]
        print(f"2. Exp dates {exp_dates}")
        
        # 3) For each expiry, choose ATM call/put and fetch bid/ask and IV
        atm_iv = {}
        straddle_mid = None
        for i, exp in enumerate(exp_dates):
            print(f"3. {i}, {exp}")
            # contracts for this expiry
            contracts = _list_contracts_for_expiry(symbol, exp)
            if not contracts:
                continue
            call_ctr = nearest_strike_contract(contracts, spot, "call")
            put_ctr = nearest_strike_contract(contracts, spot, "put")
            if not call_ctr or not put_ctr:
                continue
            print(call_ctr)
            print(put_ctr)
            c_bid, c_ask, c_iv = _get_option_quote_greeks(ticker,call_ctr['ticker'])
            p_bid, p_ask, p_iv = _get_option_quote_greeks(ticker,put_ctr['ticker'])

            # compute mids for earliest expiry for the straddle
            if i == 0:
                c_mid = _mid(c_bid, c_ask)
                p_mid = _mid(p_bid, p_ask)
                if c_mid is not None and p_mid is not None:
                    straddle_mid = c_mid + p_mid
            if c_iv is not None and p_iv is not None:
                atm_iv[exp] = (c_iv + p_iv) / 2.0;
            # Be polite to rate limits
            time.sleep(0.08)
        print("exiting loop")
        if not atm_iv:
            print("errror 1`")
            return "Error: Could not determin ATM IV for any expiration dates"
        print(f"3. atm_iv={atm_iv}")

        # 4) Build term structure spline and slope
        today = datetime.utcnow.date();
        dtes, ivs = [], []
        for exp, iv in atm_iv.items():
            d=datetime.strptime(exp, "%Y-%m-$d").date()
            dtes.append((d-today).days)
            ivs.append(float(iv))

        if len(dtes)<2:
            return "Error: Not enough expirations to build term structure."
        term_spline = build_term_structure(dtes,ivs);
        ts_slope_0_45 = (term_spline(45) - term_spline(min(dtes))) / (45-min(dtes))

        print(f"4. ts_slope_0_45=${ts_slope_0_45}")

        # 5) Daily OHLCV (~3 months) for Yang-Zhang + avg vol
        price_history = _get_stock_history_df(symbol, days=100)
        if price_history.empty:
            return "Error: no historical data"
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        avg_volume = price_history["Volume"].rolling(30).mean().dropna().iloc[-1]
        expected_move = f"{round(straddle_mid / spot * 100, 2)}%" if straddle_mid else None
            
        resultPackage = {
                "avg_volume" : avg_volume>=1_500_000,
                "iv30_rv30" : iv30_rv30 >= 1.25,
                "ts_slope_0_45" : ts_slope_0_45 <= -0.00406,
                "expected_move": expected_move
            }
            
        print(resultPackage)

        return resultPackage;
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"{ticker}: Processing failed: {e}"


        



def _list_contracts_for_expiry(ticker, expiry):
    """
    Return minimal info for all contracts at a given expiry
    """
    contracts = []
    url = f"{BASE_V3}/reference/options/contracts"
    
    params={
        "underlying_ticker":ticker,
        "expiration_date":expiry,
        "limit":1000,
        "order": "asc",
        "sort": "strike_price",
        "expired" : "false",
    }
   
    while True:
        j=_get(url, params)
        for c in j.get("results", []) or []:
            occt = c.get("ticker")
            k=c.get("strike_price")
            cp=c.get("contract_type") # call or put
            if occt and k and cp:
                contracts.append({
                    "ticker":occt,
                    "strike": float(k),
                    "type" : cp
                })
        next_url = j.get("next_url")
        if not next_url:
            break
        url = next_url
    return contracts


def _list_expirations(ticker):
    """
    Get a set of option expiration dates accross contracts for an underlying.
    """
    expirations = set()
    url=f"{BASE_V3}/reference/options/contracts"
    params={
        "underlying_ticker":ticker,
        "limit":100,
        "order": "asc",
        "sort" : "expiration_date",
        "contract_type" : "call", #supports call or put, we only care about exp dates..
        "expired": "false",
    }
    while True:
        j=_get(url,params)
        for c in j.get("results", []) or []:
            exp=c.get("expiration_date")
            if exp:
                expirations.add(exp)
        next_url=j.get("next_url")
        if not next_url:
            break
        url = next_url
    return sorted(expirations)

def _get_stock_history_df(ticker, days=100): 
    """
    Return ~3 months of dailys OHLCV as a DataFrame with columns:
    Open, High, Low, Close, Volume and DatetimeIndex
    """
    end =datetime.now(timezone.utc).date()
    start = end-timedelta(days=days)
    j=_get(
        f"{BASE_V2}/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        params={"adjusted":"true","limit":5000}
    )
    rows=j.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"]=pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={
        "o":"Open", "h":"High", "l":"Low", "c":"Close", "v":"Volume"
    }, inplace=True)
    df.set_index("date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return df


def _get_underlying_price(ticker):
    try:
        j = _get(f"{BASE_V2}/snapshot/locale/us/markets/stocks/tickers/{ticker}")
        snap = j.get("ticker", {})
        if "day" in snap and snap["day"].get("c"):
            return float(snap["day"]["c"])
        if "min" in snap and snap["min"].get("c"):
            return float(snap["min"]["c"])
        if "prevDay" in snap and snap["prevDay"].get("c"):
            return float(snap["prevDay"]["c"])
    except Exception:
        pass
    return None






# _get_tickers()

#x = _get_underlying_price('AAPL')
# print(f"Underlying price: {x}")
#stock_history_df = _get_stock_history_df('AAPL')
#print(stock_history_df.head(10))

#print(_list_expirations('AAPL'))
#contracts = _list_contracts_for_expiry('AAPL', '2025-08-29')