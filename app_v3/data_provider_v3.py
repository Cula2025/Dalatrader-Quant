from __future__ import annotations
import os, requests, pandas as pd
BASE = "https://apiservice.borsdata.se"
KEY = os.getenv("BORS_API_KEY") or os.getenv("BORSDATA_API_KEY")

def _ins_id_for_ticker(ticker: str) -> int:
    r = requests.get(f"{BASE}/v1/instruments", params={"authKey": KEY}, timeout=30); r.raise_for_status()
    tnorm = (ticker or "").replace(".ST","").replace(" ", "").upper()
    for it in r.json().get("instruments", []):
        if (it.get("ticker","") or "").replace(" ","").upper() == tnorm:
            return int(it["insId"])
    raise ValueError(f"Hittar inget instrument för {ticker!r}")

def get_ohlcv(ticker: str, start: str, end: str|None=None) -> pd.DataFrame:
    if not KEY: raise RuntimeError("BORS_API_KEY saknas i env")
    ins = _ins_id_for_ticker(ticker)
    params = {"authKey": KEY, "from": start}
    if end: params["to"] = end
    r = requests.get(f"{BASE}/v1/instruments/{ins}/stockprices", params=params, timeout=30); r.raise_for_status()
    data = r.json().get("stockPricesList", []) or []
    if not data: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = pd.DataFrame(data).rename(columns={"d":"Date","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    for c in ("Open","High","Low","Close","Volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"])[["Open","High","Low","Close","Volume"]]

def px_on_or_after(df: pd.DataFrame, ts: pd.Timestamp): 
    ts = pd.to_datetime(ts).normalize(); s = df[df.index>=ts]
    return (s.index[0], float(s["Close"].iloc[0])) if len(s) else (None, None)

def px_on_or_before(df: pd.DataFrame, ts: pd.Timestamp):
    ts = pd.to_datetime(ts).normalize(); s = df[df.index<=ts]
    return (s.index[-1], float(s["Close"].iloc[-1])) if len(s) else (None, None)

def buyhold_equity(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty: return pd.Series(dtype=float, name="BH")
    eq = (close / float(close.iloc[0])); eq.name="BH"; return eq
