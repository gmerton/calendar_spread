import os
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