import os
import time
import math
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


# POLYGON_API_KEY ="REDACTED"
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

def _get_option_quote_greeks(option_ticker):
    """
    Get latest bid/ask and immplied volatility for a single option.
    """
    bid = ask = iv = None
    try:
        q=_get(f"{BASE_V3}/last/nbbo/options/{option_ticker}")
        quote = q.get("results") or {}
        bid = quote.get("bid",{}).get("price")
        ask = quote.get("ask",{}).get("price")
        iv = quote.get("implied_volatility")
        if iv is None:
            iv=quote.get("greeks", {}).get("implied_volatility")
    except Exception:
        pass
    if iv is None or bid is None or ask is None:
        try:
            s=_get(f"{BASE_V3}/snapshot/options/{option_ticker}")
            snap = s.get("results") or {}
            nb = snap.get("latestQuote",{})
            bid = bid if bid is not None else nb.get("bid", None)
            ask = ask if ask is not None else nb.get("ask", None)
            gr = snap.get("greeks",{}) or snap.get("day",{}).get("greeks", {})
            iv = iv if iv is not None else gr.get("implied_volatility")
        except Exception:
            pass
    bid = float(bid) if bid is not None else None
    ask = float(ask) if ask is not None else None
    iv = float(iv) if iv is not None else None
    return bid, ask, iv

def _nearest_strike_contract(contracts, spot, cp):
    """
    Pick the contract dict from _list_contracts_for_expiry nearest to spot
    """
    side = [c for c in contracts if c["type"]==cp]
    if not side:
        return None
    return min(side, key=lambda c: abs(c["strike"]-spot))

def _mid(bid, ask):
    return (bid+ask)/2.0 if (bid and ask and bid>0 and ask>0) else None

def commpute_recommendation_polygon(ticker, max_expiries=6):
    try:
        symbol = (ticker or "").strip().upper()
        if not symbol:
            return "No stock symbol provided."
        # 1) Get current stock price
        spot = _get_underlying_price(symbol)
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

        # 3) For each expiry, choose ATM call/put and fetch bid/ask and IV
        atm_iv = {}
        straddle_mid = None
        for i, exp in enumerate(exp_dates):
            # contracts for this expiry
            contracts = _list_contracts_for_expiry(symbol, exp)
            if not contracts:
                continue
            call_ctr = _nearest_strike_contract(contracts, spot, "call")
            put_ctr = _nearest_strike_contract(contracts, spot, "put")
            if not call_ctr or not put_ctr:
                continue
            c_bid, c_ask, c_iv = _get_option_quote_greeks(call_ctr["ticker"])
            p_bid, p_ask, p_iv = _get_option_quote_greeks(put_ctr["ticker"])

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
            if not atm_iv:
                return "Error: Could not determin ATM IV for any expiration dates"
            
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

            # 5) Daily OHLCV (~3 months) for Yang-Zhang + avg vol
            price_history = _get_stock_history_df(symbol, days=100)
            if price_history.empty:
                return "Error: no historical data"
            iv30_rv39 = term_spline(30) / yang_zhang(price_history)
            avg_volume = price_history["Volume"].rolling(30).mean().dropna().iloc[-1]
            expected_move = f"{round(straddle_mid / spot * 100, 2)}%" if straddle_mid else None
            return {
                "avg_volume" : avg_volume>=1_500_000,
                "iv30_rv30" : iv30_rv30 >= 1.25,
                "ts_slope_0_45" : ts_slope_0_45 <= -0.00406,
                "expected_move": expected_move
            }
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
        print(j)
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
    print(len(contracts))
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
    print(params)
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
contracts = _list_contracts_for_expiry('AAPL', '2025-08-29')