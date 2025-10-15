# app/borsdata_client.py
# BÃ¶rsdata-klient med robust fallback:
# - LÃ¤ser API-nyckel frÃ¥n .env (override=True) och miljÃ¶n (BORSDATA_API_KEY/BD_API_KEY/BD_TOKEN/BORSDATA_KEY)
# - instruments() med 24h cache
# - choose() med robust ticker-/namnmatch + alias fÃ¶r Ticker/Name
# - prices(): fÃ¶rst singel-endpoint /instruments/{InsId}/stockprices,
#             vid fel/tomt â†’ fallback till array-endpoint /instruments/stockprices?instList=...
# - prices_by_ticker() stÃ¶djer start/end + fÃ¶rsiktigt MIC-filter

from __future__ import annotations

import os
import json
import time
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import HTTPError, RequestException

# --- Ladda .env; lÃ¥t .env vinna Ã¶ver tidigare env vars ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# --- TillÃ¥t Streamlit secrets (om det rÃ¥kar finnas) ---
try:
    import streamlit as st  # type: ignore
    _sec = (
        st.secrets.get("BORSDATA_API_KEY")
        or st.secrets.get("BD_API_KEY")
        or st.secrets.get("BD_TOKEN")
        or st.secrets.get("BORSDATA_KEY")
    )
    if _sec and not os.getenv("BORSDATA_API_KEY"):
        os.environ["BORSDATA_API_KEY"] = str(_sec)
except Exception:
    pass


def _canon_ticker(s: str) -> str:
    if not s:
        return ""
    return s.replace(".", "").replace(" ", "").replace("-", "").upper()


def _extract_ins_id(meta: dict) -> int:
    for k in ("InsId", "InstrumentId", "Id", "insId", "instrumentId", "id", "insid", "instrumentid"):
        if k in meta and meta[k] is not None:
            try:
                return int(meta[k])
            except Exception:
                pass
    raise KeyError("Instrument id key not found in instrument metadata")


def _alias_instrument_fields(it: dict) -> dict:
    out = dict(it)
    if out.get("Ticker") is None:
        for k in ("ticker", "Symbol", "symbol", "ShortName", "shortName"):
            if out.get(k):
                out["Ticker"] = out[k]
                break
    if out.get("Name") is None:
        for k in ("name", "CompanyName", "companyName", "Instrument", "instrument", "FullName", "fullName"):
            if out.get(k):
                out["Name"] = out[k]
                break
    return out


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


class BDClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://apiservice.borsdata.se/v1",
        cache_dir: str = os.path.join("app", ".cache", "borsdata"),
        timeout: int = 20,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("BORSDATA_API_KEY")
            or os.getenv("BD_API_KEY")
            or os.getenv("BD_TOKEN")
            or os.getenv("BORSDATA_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "BÃ¶rsdata API key saknas. SÃ¤tt BORSDATA_API_KEY (eller BD_API_KEY/BD_TOKEN/BORSDATA_KEY) i miljÃ¶n."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._instruments_cache: Optional[List[Dict[str, Any]]] = None

        self.cache_dir = cache_dir
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            pass

    # -------------------------------
    # HTTP
    # -------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        q = dict(params or {})
        q["authKey"] = self.api_key

        r = self.session.get(url, params=q, timeout=self.timeout)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After") or r.headers.get("retry-after")
            if ra:
                time.sleep(float(ra))
                r = self.session.get(url, params=q, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return r.text

    # -------------------------------
    # Instruments
    # -------------------------------
    def instruments(self, use_cache: bool = True, force_refresh: bool = False) -> List[Dict[str, Any]]:
        if self._instruments_cache is not None and not force_refresh:
            return self._instruments_cache

        cache_file = os.path.join(self.cache_dir, "instruments.json")
        if use_cache and not force_refresh and os.path.exists(cache_file):
            try:
                mtime = os.path.getmtime(cache_file)
                if (time.time() - mtime) < 24 * 3600:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items = data.get("instruments") if isinstance(data, dict) else data
                    if isinstance(items, list) and items:
                        self._instruments_cache = items
                        return items
            except Exception:
                pass

        data = self._get("/instruments")
        items = data.get("instruments") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError("Kunde inte lÃ¤sa instrumentlistan frÃ¥n BÃ¶rsdata.")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"cachedAt": _now_utc_iso(), "instruments": items}, f, ensure_ascii=False)
        except Exception:
            pass

        self._instruments_cache = items
        return items

    def choose(self, q: str) -> Optional[Dict[str, Any]]:
        if not q:
            return None
        cq = _canon_ticker(q)
        items = self.instruments()

        # 1) Exakt ticker
        for raw in items:
            t = _canon_ticker(raw.get("Ticker") or raw.get("ticker") or raw.get("Symbol") or raw.get("symbol") or "")
            if t == cq:
                return _alias_instrument_fields(raw)

        # 2) Hantera A/B-suffix
        if cq.endswith("A") or cq.endswith("B"):
            base = cq[:-1]
            for raw in items:
                t = _canon_ticker(raw.get("Ticker") or raw.get("ticker") or raw.get("Symbol") or raw.get("symbol") or "")
                if t == base or t.startswith(base):
                    return _alias_instrument_fields(raw)

        # 3) Name contains
        lq = (q or "").strip().lower()
        if lq:
            for raw in items:
                name = (raw.get("Name") or raw.get("name") or raw.get("CompanyName") or raw.get("companyName") or "").lower()
                if lq in name:
                    return _alias_instrument_fields(raw)

        return None

    # -------------------------------
    # Prices
    # -------------------------------
    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for r in rows:
            if "Date" not in r:
                if "d" in r:
                    r["Date"] = r["d"]
                elif "date" in r:
                    r["Date"] = r["date"]
            if "Open" not in r and "o" in r:
                r["Open"] = r["o"]
            if "High" not in r and "h" in r:
                r["High"] = r["h"]
            if "Low" not in r and "l" in r:
                r["Low"] = r["l"]
            if "Close" not in r and "c" in r:
                r["Close"] = r["c"]
            if "Volume" not in r and "v" in r:
                r["Volume"] = r["v"]
            if "mic" not in r and "m" in r:
                r["mic"] = r["m"]
        return rows

    def _prices_array_fallback(self, ins_id: int, start: Optional[str], end: Optional[str]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"instList": str(int(ins_id))}
        if start:
            params["from"] = start
        if end:
            params["to"] = end

        try:
            data = self._get("/instruments/stockprices", params=params)
        except (HTTPError, RequestException):
            return []

        # FÃ¶rvÃ¤ntat: {"stockPricesList":[{"insId":X,"stockPrices":[...]}, ...]}
        rows: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            spl = (
                data.get("stockPricesList")
                or data.get("stockpricesList")
                or data.get("StockPricesList")
            )
            if isinstance(spl, list) and spl:
                block = None
                for b in spl:
                    if (
                        b.get("insId") == ins_id
                        or b.get("InsId") == ins_id
                        or b.get("id") == ins_id
                    ):
                        block = b
                        break
                if block is None:
                    block = spl[0]
                cand = block.get("stockPrices") or block.get("StockPrices") or block.get("stockprices")
                if isinstance(cand, list):
                    rows = cand

        return self._normalize_rows(rows) if rows else []

    def prices(
        self,
        ins_id: int,
        max_count: int = 20000,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        PrimÃ¤r: /v1/instruments/{InsId}/stockprices
        Fallback: /v1/instruments/stockprices?instList={InsId}
        """
        params: Dict[str, Any] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        if not start and not end:
            mc = min(int(max_count or 20000), 20000)
            # Skicka bÃ¥da varianterna
            params["maxCount"] = mc
            params["maxcount"] = mc

        # 1) Singel-endpoint
        try:
            data = self._get(f"/instruments/{int(ins_id)}/stockprices", params=params)
            rows = data.get("stockPrices") if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                # tomt â‡’ prova fallback
                return self._prices_array_fallback(ins_id, start, end)
            return self._normalize_rows(rows)
        except (HTTPError, RequestException):
            # 500/andra fel â‡’ prova fallback
            return self._prices_array_fallback(ins_id, start, end)

    def prices_by_ticker(
        self,
        ticker: str,
        days: int = 20000,
        prefer_mic: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        it = self.choose(ticker)
        if not it:
            return []
        ins_id = _extract_ins_id(it)

        # Om start/end angivits, anvÃ¤nd dem; annars max_count
        if start or end:
            rows = self.prices(ins_id, start=start, end=end)
        else:
            rows = self.prices(ins_id, max_count=days)

        rows = rows or []
        if not rows:
            return rows

        if prefer_mic:
            any_mic = any(r.get("mic") for r in rows)
            if any_mic:
                filtered = [r for r in rows if r.get("mic") == prefer_mic]
                if filtered:
                    rows = filtered
        return rows
