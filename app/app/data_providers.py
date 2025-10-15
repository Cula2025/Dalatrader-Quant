from __future__ import annotations
import os, json, time, re
from typing import Optional, List, Dict
import requests
import pandas as pd

# Hårdkodad nyckel (kan överstyras via env BORSDATA_API_KEY / BORS_API_KEY)
API_KEY = os.environ.get("BORSDATA_API_KEY") or os.environ.get("BORS_API_KEY") or "85218870c1744409b0624920db023ba8"
BASE = "https://apiservice.borsdata.se"

# Cache-katalog
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage"))
os.makedirs(CACHE_DIR, exist_ok=True)

def _norm_ticker(t: str) -> str:
    # Normalisera: trimma, uppercase, kollapsa mellanrum
    return re.sub(r"\s+", " ", (t or "").strip().upper())

def _load_instruments(force: bool=False) -> List[Dict]:
    path = os.path.join(CACHE_DIR, "bd_instruments.json")
    fresh = os.path.isfile(path) and (time.time() - os.path.getmtime(path) < 24*3600)
    if fresh and not force:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("instruments", data if isinstance(data, list) else [])
        except Exception:
            pass

    r = requests.get(f"{BASE}/v1/instruments", params={"authKey": API_KEY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data.get("instruments", data if isinstance(data, list) else [])

def _resolve_ins_id(ticker: str) -> Optional[int]:
    target = _norm_ticker(ticker)
    items = _load_instruments()

    # 1) exakt match på 'ticker' inklusive mellanslag (GETI B)
    for it in items:
        if _norm_ticker(it.get("ticker","")) == target:
            return it.get("insId")

    # 2) fallback: ignorera mellanslag
    nospace = target.replace(" ", "")
    for it in items:
        if _norm_ticker(it.get("ticker","")).replace(" ", "") == nospace:
            return it.get("insId")

    return None

def get_ohlcv(ticker: str, start: Optional[str]=None, end: Optional[str]=None, freq: str="D") -> pd.DataFrame:
    ins = _resolve_ins_id(ticker)
    if ins is None:
        raise ValueError(f"Okänd ticker hos Börsdata: '{ticker}'")

    params = {"authKey": API_KEY, "maxCount": 50000}
    if start:
        params["from"] = str(pd.Timestamp(start).date())
    if end:
        params["to"] = str(pd.Timestamp(end).date())

    r = requests.get(f"{BASE}/v1/instruments/{ins}/stockprices", params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = j.get("stockPricesList") or []
    if not rows:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    recs = []
    for rec in rows:
        recs.append({
            "Date":   pd.to_datetime(rec["d"]),
            "Open":   float(rec["o"]),
            "High":   float(rec["h"]),
            "Low":    float(rec["l"]),
            "Close":  float(rec["c"]),
            "Volume": int(rec.get("v", 0)),
        })
    df = pd.DataFrame(recs).set_index("Date").sort_index()
    return df
