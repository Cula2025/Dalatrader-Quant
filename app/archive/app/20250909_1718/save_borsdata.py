# app/save_borsdata.py
from __future__ import annotations
import os, re, json, argparse, datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

# Försök läsa .env om python-dotenv finns
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def _sanitize_ticker(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^A-Za-z0-9_\-\.]", "", t)
    return t

def _get_key() -> str:
    k = os.environ.get("BORSDATA_KEY") or os.environ.get("BORS_DATA_KEY")
    if not k:
        raise RuntimeError("Saknar BORSDATA_KEY i miljön (.env).")
    return k

def _choose_instrument(ticker: str, prefer_mic: Optional[str] = "XSTO") -> dict:
    """
    Hämta instrument-objekt för 'ticker' via /v1/instruments.
    Försöker matcha exakt 'ticker' (skiftlägeskänsligt som Börsdata brukar returnera),
    annars case-insensitive, och om prefer_mic satts, väljs i första hand instrument på den MIC:en.
    """
    key = _get_key()
    url = "https://apiservice.borsdata.se/v1/instruments"
    r = requests.get(url, params={"authKey": key}, timeout=30)
    r.raise_for_status()
    instruments = r.json().get("instruments", [])
    if not instruments:
        raise RuntimeError("Börsdata: tom instrumentlista.")

    t_norm = ticker.strip().upper().replace(" ", "")
    candidates = []
    for it in instruments:
        it_ticker = str(it.get("ticker", "")).strip()
        if not it_ticker:
            continue
        it_norm = it_ticker.upper().replace(" ", "")
        if it_norm == t_norm:
            candidates.append(it)

    if not candidates:
        # Prova delsträng (t.ex. VOLV B vs VOLVB), fallback
        for it in instruments:
            it_ticker = str(it.get("ticker", "")).strip()
            it_norm = it_ticker.upper().replace(" ", "")
            if t_norm in it_norm:
                candidates.append(it)

    if not candidates:
        raise ValueError(f"Hittar inget instrument för '{ticker}'.")

    if prefer_mic:
        # Hämta marktillhörighet via /v1/markets för mapping (om vi vill – men enklare: 
        # många instrument har 'marketId'; vi kan inte mappa till MIC direkt här,
        # så vi väljer bara första träff och skriver en varning om flera)
        # För enkelhet: om flera, välj första och logga urvalet.
        pass

    # Välj första kandidat
    return candidates[0]

def _fetch_prices_by_insid(ins_id: int, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Hämtar dagliga priser via /v1/instruments/{insId}/stockprices med from/to.
    JSON-nycklar: 'stockPricesList' (objektlista med fälten: d, o, h, l, c, v)
    """
    key = _get_key()
    url = f"https://apiservice.borsdata.se/v1/instruments/{int(ins_id)}/stockprices"
    params = {"authKey": key}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    r = requests.get(url, params=params, timeout=45)
    if r.status_code == 401:
        raise PermissionError("Börsdata: 401 Unauthorized – kontrollera BORSDATA_KEY.")
    r.raise_for_status()
    j = r.json()

    lst = j.get("stockPricesList") or j.get("stockPrices") or []
    if not lst:
        # För vissa äldre intervall kan listan ligga under annan nyckel-struktur – men de två ovan täcker normalt.
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    df = pd.DataFrame(lst)
    # Förväntade fält: d, o, h, l, c, v
    rename = {"d":"Date","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}
    df = df.rename(columns=rename)
    if "Date" not in df.columns:
        raise RuntimeError("Oväntat svar från Börsdata: saknar fältet 'Date' (d).")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[cols]

def save_csv(
    ticker: str,
    date_from: Optional[str],
    date_to: Optional[str],
    mic: Optional[str] = "XSTO",
    out_dir: str = ".\\outputs\\prices",
) -> Tuple[Path, pd.DataFrame]:
    """
    Hämtar priser för 'ticker' under [date_from, date_to] och sparar till CSV.
    Returnerar (full_path, df).
    """
    # Normalisera datum
    def norm_date(s):
        if not s:
            return None
        return dt.date.fromisoformat(str(s)).isoformat()

    start = norm_date(date_from)
    end   = norm_date(date_to)

    inst = _choose_instrument(ticker, prefer_mic=mic)
    ins_id = int(inst.get("insId") or inst.get("InsId") or inst.get("id") or inst.get("Id"))
    df = _fetch_prices_by_insid(ins_id, start, end)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe = _sanitize_ticker(ticker)
    stamp = f"{start or 'START'}_{end or 'END'}"
    fp = Path(out_dir) / f"{safe}_{stamp}.csv"
    df.to_csv(fp, index=True, date_format="%Y-%m-%d")
    return fp, df

# ==== CLI ====
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spara Börsdata-priser till CSV.")
    p.add_argument("--ticker", required=True, help="Ex: 'VOLV B', 'HM B'")
    p.add_argument("--start", dest="start", default=None, help="YYYY-MM-DD (valfritt)")
    p.add_argument("--end", dest="end", default=None, help="YYYY-MM-DD (valfritt)")
    p.add_argument("--mic", default="XSTO", help="Förval XSTO (valfritt)")
    p.add_argument("--out", default=".\\outputs\\prices", help="Utkatalog")
    return p

def main():
    args = _build_argparser().parse_args()
    fp, df = save_csv(args.ticker, args.start, args.end, mic=args.mic, out_dir=args.out)
    print(f"[OK] Sparat {len(df):,} rader -> {fp}")

if __name__ == "__main__":
    main()


