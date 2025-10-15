# app/bd_modern_client.py
from __future__ import annotations

import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

# Ladda .env om mÃ¶jligt
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

try:
    from borsdata_client import BorsdataClient  # alexwox/Modern-Borsdata-Client
except Exception as e:
    raise RuntimeError(
        "Paketet 'borsdata-client' saknas. Installera med:\n"
        '  pip install "git+https://github.com/alexwox/Modern-Borsdata-Client.git#egg=borsdata-client"\n'
        f"Ursprungligt fel: {e}"
    )

def _canon_ticker(s: str) -> str:
    return (s or "").replace(".", "").replace("-", "").replace(" ", "").upper()

def _to_date(obj: Optional[str]) -> Optional[date]:
    if not obj:
        return None
    try:
        return datetime.strptime(obj, "%Y-%m-%d").date()
    except Exception:
        pass
    try:
        if len(obj) == 8 and obj.isdigit():
            return datetime.strptime(obj, "%Y%m%d").date()
    except Exception:
        pass
    try:
        from pandas import to_datetime
        return to_datetime(obj).date()
    except Exception:
        return None

class BDModernAdapter:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = (
            api_key
            or os.getenv("BORSDATA_API_KEY")
            or os.getenv("BD_API_KEY")
            or os.getenv("BD_TOKEN")
            or os.getenv("BORSDATA_KEY")
        )
        if not self.api_key:
            raise RuntimeError("BÃ¶rsdata API key saknas i miljÃ¶n (.env). SÃ¤tt BORSDATA_API_KEY.")
        self._instruments_cache: Optional[List[Any]] = None

    def instruments(self) -> List[Any]:
        if self._instruments_cache is not None:
            return self._instruments_cache
        with BorsdataClient(self.api_key) as client:
            self._instruments_cache = client.get_instruments()
        return self._instruments_cache

    def choose(self, q: str) -> Optional[Dict[str, Any]]:
        if not q:
            return None
        cq = _canon_ticker(q)
        items = self.instruments()

        for it in items:  # exakt ticker
            tick = getattr(it, "ticker", None) or getattr(it, "Ticker", None) or ""
            if _canon_ticker(tick) == cq:
                return {"InsId": getattr(it, "insId", None), "Ticker": tick, "Name": getattr(it, "name", "")}

        if cq.endswith("A") or cq.endswith("B"):  # HM B â†’ HMB/HMâ€¦
            base = cq[:-1]
            for it in items:
                tick = getattr(it, "ticker", None) or ""
                ct = _canon_ticker(tick)
                if ct == base or ct.startswith(base):
                    return {"InsId": getattr(it, "insId", None), "Ticker": tick, "Name": getattr(it, "name", "")}

        lq = (q or "").strip().lower()  # name contains
        for it in items:
            nm = (getattr(it, "name", None) or "").lower()
            if lq and lq in nm:
                return {"InsId": getattr(it, "insId", None), "Ticker": getattr(it, "ticker", ""), "Name": getattr(it, "name", "")}
        return None

    def prices_by_ticker(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        chosen = self.choose(ticker)
        if not chosen or not chosen.get("InsId"):
            return []
        ins_id = int(chosen["InsId"])

        d_from = _to_date(start) or date(2000, 1, 1)
        d_to: Optional[date] = _to_date(end) if end else None

        with BorsdataClient(self.api_key) as client:
            prices = client.get_stock_prices(instrument_id=ins_id, from_date=d_from, to_date=d_to)

        rows: List[Dict[str, Any]] = []
        for p in prices:
            get = (lambda k: getattr(p, k, None) if hasattr(p, k) else (p.get(k) if isinstance(p, dict) else None))
            d = get("d") or get("date") or get("Date")
            o = get("o") or get("open") or get("Open")
            h = get("h") or get("high") or get("High")
            l = get("l") or get("low") or get("Low")
            c = get("c") or get("close") or get("Close")
            v = get("v") or get("volume") or get("Volume")
            rows.append({"Date": d, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
        return rows
