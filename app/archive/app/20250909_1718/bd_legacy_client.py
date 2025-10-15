import os
import re
import time
import requests
from typing import Optional, Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ladda .env robust
_loaded = False
_env_path = find_dotenv(usecwd=True)
if _env_path:
    load_dotenv(_env_path)
    _loaded = True
else:
    proj_root = Path(__file__).resolve().parents[1]  # C:\trader
    root_env = proj_root / ".env"
    if root_env.exists():
        load_dotenv(root_env.as_posix())
        _loaded = True
if not _loaded:
    load_dotenv()

BASE = "https://apiservice.borsdata.se/v1"


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


class BDClient:
    """
    BÃ¶rsdata-klient (EOD):
      â€¢ Auth via env BORSDATA_KEY
      â€¢ Instrument-cache (24h)
      â€¢ Retry pÃ¥ 429/5xx/connection errors (exponentiell backoff)
      â€¢ Korrekt parsing av stockPricesList/stockPricesArrayList
    """

    def __init__(self, key: Optional[str] = None, base: str = BASE) -> None:
        self.key = (key or os.getenv("BORSDATA_KEY") or "").strip()
        if not self.key:
            raise RuntimeError("Missing BORSDATA_KEY in environment (.env)")
        self.base = base

        # HTTP-session med retries
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Tradbot/BD"})
        retry = Retry(
            total=6,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)

        self._instruments: Optional[List[Dict[str, Any]]] = None
        self._ts: float = 0.0

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        q = {"authKey": self.key}
        if params:
            q.update(params)

        # extra failsafe-retries pÃ¥ alla undantag
        for attempt in range(6):
            try:
                r = self.s.get(f"{self.base}/{path}", params=q, timeout=(5, 40))
                r.raise_for_status()
                if "application/json" in (r.headers.get("content-type") or ""):
                    return r.json()
                raise RuntimeError(f"Unexpected content-type: {r.headers.get('content-type')}")
            except requests.exceptions.RequestException as e:
                # pÃ¥ sista fÃ¶rsÃ¶ket: kasta vidare
                if attempt == 5:
                    raise
                # backoff (Ã¶kande)
                time.sleep(0.8 * (attempt + 1))

        # bÃ¶r ej hamna hÃ¤r
        return {}

    # --------------- instruments ----------------

    def instruments(self, force: bool = False) -> List[Dict[str, Any]]:
        if force or self._instruments is None or (time.time() - self._ts) > 86400:
            payload = self._get("instruments")
            lst = payload.get("instruments") or payload.get("Instruments") or payload
            if not isinstance(lst, list):
                raise RuntimeError("Unexpected instruments payload structure.")
            self._instruments = lst
            self._ts = time.time()
        return self._instruments

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

    def _ins_core(self, it: Dict[str, Any]) -> Dict[str, Any]:
        ik = _lower_keys(it)
        return {
            "ticker": ik.get("ticker") or ik.get("symbol"),
            "name": ik.get("name") or ik.get("companyname"),
            "mic": ik.get("mic") or ik.get("exchangemic") or ik.get("primarymic"),
            "insid": ik.get("insid") or ik.get("instrumentid") or ik.get("id"),
        }

    def choose(self, query: str, prefer_mic: Optional[str] = "XSTO") -> Optional[Dict[str, Any]]:
        qn = self._norm(query)
        best_exact: Optional[Dict[str, Any]] = None
        best_partial: Optional[Dict[str, Any]] = None
        pref = (prefer_mic or "").upper() if prefer_mic else None

        for it in self.instruments():
            core = self._ins_core(it)
            t_norm = self._norm(core["ticker"])
            n_norm = self._norm(core["name"])
            if not t_norm and not n_norm:
                continue
            if t_norm == qn:
                if pref and (str(core["mic"] or "").upper() == pref):
                    return it
                best_exact = best_exact or it
                continue
            if t_norm and (qn in t_norm or t_norm in qn):
                if pref and (str(core["mic"] or "").upper() == pref):
                    best_partial = best_partial or it
                best_partial = best_partial or it
                continue
            if n_norm and qn and (qn in n_norm):
                best_partial = best_partial or it
        return best_exact or best_partial

    # --------------- prices ----------------

    def _parse_prices_payload(self, data: Dict, ins_id: Optional[int] = None) -> List[Dict]:
        """
        StÃ¶d fÃ¶r:
          â€¢ {'instrument': 97, 'stockPricesList': [ ... ]}
          â€¢ {'stockPricesArrayList': [ {'instrument':97,'stockPricesList':[...]} , ... ] }
          â€¢ {'stockPrices': [ ... ]}  (fallback)
        """
        if not isinstance(data, dict):
            return []
        if isinstance(data.get("stockPricesList"), list):
            return data["stockPricesList"]
        arr = data.get("stockPricesArrayList")
        if isinstance(arr, list) and arr:
            if ins_id is None:
                first = arr[0]
                if isinstance(first, dict) and isinstance(first.get("stockPricesList"), list):
                    return first["stockPricesList"]
            else:
                for block in arr:
                    if isinstance(block, dict) and int(block.get("instrument", -1)) == int(ins_id):
                        lst = block.get("stockPricesList")
                        if isinstance(lst, list):
                            return lst
        if isinstance(data.get("stockPrices"), list):
            return data["stockPrices"]
        return []

    def prices(
        self,
        ins_id: int,
        max_count: Optional[int] = 20000,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict]:
        """
        HÃ¤mtar historik. AnvÃ¤nder 'maxcount' om start/end saknas, annars 'from'/'to'.
        Returnerar listan med dictar {'d','o','h','l','c','v'}.
        """
        params: Dict[str, Any] = {}
        if start or end:
            if start:
                params["from"] = start
            if end:
                params["to"] = end
        else:
            params["maxcount"] = int(max_count or 20000)

        data = self._get(f"instruments/{int(ins_id)}/stockprices", params)
        return self._parse_prices_payload(data, ins_id=ins_id)

    def prices_by_ticker(self, ticker: str, days: int = 20000, prefer_mic: Optional[str] = "XSTO") -> List[Dict]:
        it = self.choose(ticker, prefer_mic=prefer_mic)
        if not it:
            raise ValueError(f"No instrument found for '{ticker}'")
        core = self._ins_core(it)
        if core["insid"] is None:
            raise KeyError("Instrument lacks InsId / InstrumentId.")
        return self.prices(int(core["insid"]), max_count=days)


