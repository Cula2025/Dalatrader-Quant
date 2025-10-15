"""
bd_modern_client.py (Börsdata-only, EOD, DEMO)
- Hårdkodad nyckel enligt din begäran för demo-lokal körning.
- Hämtar OHLCV via Börsdata REST-API v1.
- Primär väg: /v1/instruments/stockprices?instList=... (array-endpoint)
- Stödjer svar med nycklar:
    • stockPricesArrayList -> [{ instrument: <id>, stockPricesList: [...] }]
    • stockPrices / StockPrices -> [ ... ]
"""

from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import urllib.error as ue
import pandas as pd

_BASE = "https://apiservice.borsdata.se/v1"

# HÅRDKODAD DEMO-NYCKEL (lokal demo)
_HARDCODED_BORSDATA_KEY = "85218870c1744409b0624920db023ba8"

# ---------- Key / client ----------
def _resolve_bd_key() -> str:
    print("[DEMO] Ignorerar env och använder hårdkodad BORSDATA-nyckel.")
    return _HARDCODED_BORSDATA_KEY

def get_client() -> str:
    return _resolve_bd_key()

class BDModernAdapter:
    def __init__(self, _client: Any | None = None) -> None:
        self.client = _client or get_client()
    def ping(self) -> bool:
        return True

def using_hardcoded_key() -> bool:
    return True

# ---------- HTTP helper ----------
def _http_get_json(path: str, params: Dict[str, Any]) -> Any:
    qs = urlencode(params)
    url = f"{_BASE}{path}?{qs}"
    req = Request(url, headers={"User-Agent": "Dalatraderbot/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:
        if isinstance(e, ue.HTTPError):
            safe = url.replace(str(params.get("authKey", "")), "****")
            raise RuntimeError(f"HTTP {e.code} {e.reason} – {safe}") from e
        raise
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw)

# ---------- Instruments ----------
_INSTR_CACHE: Dict[str, Any] = {}
_INSTR_CACHE_TS = 0.0
_INSTR_TTL_SEC = 3600

def _load_instruments(auth_key: str) -> List[Dict[str, Any]]:
    global _INSTR_CACHE, _INSTR_CACHE_TS
    now = time.time()
    if _INSTR_CACHE and (now - _INSTR_CACHE_TS < _INSTR_TTL_SEC):
        return _INSTR_CACHE["data"]
    data = _http_get_json("/instruments", {"authKey": auth_key})
    if isinstance(data, dict):
        items = data.get("instruments") or data.get("Instruments") or []
        if not isinstance(items, list):
            items = []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    _INSTR_CACHE, _INSTR_CACHE_TS = {"data": items}, now
    return items

def get_instruments() -> List[Dict[str, Any]]:
    key = get_client()
    return _load_instruments(key)

def _norm_ticker_for_match(t: str) -> str:
    t = (t or "").upper().strip()
    if t.endswith(".ST"):
        t = t[:-3]
    return " ".join(t.replace("-", " ").replace(".", " ").split())

def _resolve_instrument_id(ticker_like: str, auth_key: str) -> Optional[int]:
    want_a = _norm_ticker_for_match(ticker_like)   # "SAAB B"
    want_b = want_a.replace(" ", "")               # "SAABB"
    instruments = _load_instruments(auth_key)
    cand: List[Tuple[int, str]] = []
    for it in instruments:
        ins_id = it.get("insId") or it.get("InsId") or it.get("id") or it.get("Id")
        if not ins_id:
            continue
        ticker = (it.get("ticker") or it.get("Ticker") or "").upper()
        yahoo  = (it.get("yahoo")  or it.get("Yahoo")  or "").upper()
        name   = (it.get("name")   or it.get("Name")   or "").upper()
        tic_sp   = ticker.replace(" ", "")
        yfs_norm = _norm_ticker_for_match(yahoo)
        yfs_sp   = yfs_norm.replace(" ", "")
        if ticker == want_a or tic_sp == want_b:
            cand.append((int(ins_id), ticker or yahoo or name)); continue
        if (yfs_norm == want_a or yfs_sp == want_b or yahoo == ticker_like.upper()):
            cand.append((int(ins_id), yahoo or ticker or name)); continue
        if (ticker.startswith(want_a) or tic_sp.startswith(want_b)
            or yfs_norm.startswith(want_a) or yfs_sp.startswith(want_b)):
            cand.append((int(ins_id), ticker or yahoo or name))
    if not cand:
        return None
    cand.sort(key=lambda x: len(x[1] or ""))
    return cand[0][0]

# ---------- Parsers ----------
def _parse_prices_rows(rows: Any) -> pd.DataFrame:
    recs = []
    for r in rows or []:
        d = r.get("d") or r.get("Date") or r.get("date")
        if d is None:
            continue
        recs.append({
            "Date":   pd.to_datetime(d),
            "Open":   r.get("o") or r.get("Open")  or r.get("open"),
            "High":   r.get("h") or r.get("High")  or r.get("high"),
            "Low":    r.get("l") or r.get("Low")   or r.get("low"),
            "Close":  r.get("c") or r.get("Close") or r.get("close"),
            "Volume": r.get("v") or r.get("Volume")or r.get("volume"),
        })
    if not recs:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = pd.DataFrame.from_records(recs).set_index("Date").sort_index()
    return df[["Open","High","Low","Close","Volume"]].dropna(how="all")

def _parse_stockprices_payload(payload: Any, prefer_ins_id: Optional[int]=None) -> pd.DataFrame:
    # list at top level?
    if isinstance(payload, list):
        return _parse_prices_rows(payload)
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    # classic keys
    for topk in ("stockPrices","StockPrices","stockprices","prices","data","Data","list","List"):
        val = payload.get(topk)
        if isinstance(val, list) and val and isinstance(val[0], dict) and (
            "d" in val[0] or "Date" in val[0] or "date" in val[0]
        ):
            return _parse_prices_rows(val)
    # array variant: stockPricesArrayList -> [{ instrument, stockPricesList }]
    arr = payload.get("stockPricesArrayList")
    if isinstance(arr, list) and arr and isinstance(arr[0], dict):
        chosen = None
        if prefer_ins_id is not None:
            for itm in arr:
                try:
                    if int(itm.get("instrument")) == int(prefer_ins_id):
                        chosen = itm; break
                except Exception:
                    pass
        if chosen is None:
            chosen = arr[0]
        inner = chosen.get("stockPricesList") or []
        if isinstance(inner, list):
            return _parse_prices_rows(inner)
    return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

# ---------- Public API ----------
def get_ohlcv(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    interval: str = "1d",
    source: str = "borsdata",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    if source.lower() != "borsdata":
        raise ValueError("Endast source='borsdata' stöds i denna build.")
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker måste vara en icke-tom sträng")
    auth_key = get_client()
    inst_id = _resolve_instrument_id(ticker, auth_key)
    if inst_id is None:
        raise RuntimeError(
            f"Kunde inte hitta InstrumentId för '{ticker}'. "
            f"Prova t.ex. 'SAAB B' eller kontrollera i /v1/instruments."
        )
    params: Dict[str, Any] = {"authKey": auth_key}
    if start: params["from"] = start
    if end:   params["to"]   = end
    if "from" not in params and "to" not in params:
        params["maxcount"] = 2000

    # Primary: array endpoint
    p_arr = dict(params); p_arr["instList"] = str(inst_id)
    payload = _http_get_json("/instruments/stockprices", p_arr)
    df = _parse_stockprices_payload(payload, prefer_ins_id=inst_id)
    if not df.empty:
        return df

    # Fallback: single
    payload2 = _http_get_json(f"/instruments/{inst_id}/stockprices", params)
    df2 = _parse_stockprices_payload(payload2)
    if not df2.empty:
        return df2

    raise RuntimeError(
        f"Ingen OHLCV-data för '{ticker}' (InstrumentId={inst_id}). "
        f"Prova: {_BASE}/instruments/stockprices?authKey=****&instList={inst_id}"
    )

__all__ = ["get_client","BDModernAdapter","get_ohlcv","using_hardcoded_key","get_instruments"]