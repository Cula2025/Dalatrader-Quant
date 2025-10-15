from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.env_bootstrap import load_env

def get_ohlcv_any(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Försök via data_providers -> modern -> legacy."""
    try:
        from app.data_providers import get_ohlcv as dp_get
        return dp_get(ticker, start, end)
    except Exception as e1:
        try:
            from app.bd_modern_client import get_ohlcv as bd_get
            return bd_get(ticker, start, end)
        except Exception as e2:
            try:
                from app.bd_legacy_client import get_ohlcv as bd_legacy_get
                return bd_legacy_get(ticker, start, end)
            except Exception as e3:
                raise RuntimeError(
                    f"Kunde inte hämta data för '{ticker}'.\n"
                    f"data_providers: {e1}\nmodern_client: {e2}\nlegacy_client: {e3}"
                ) from e3

def _pick(df: pd.DataFrame, *cands: str) -> Optional[pd.Series]:
    cols = {c.strip().lower(): c for c in df.columns}
    for name in cands:
        if name in cols:
            return df[cols[name]]
    return None

def to_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalisera till OHLCV med Date-index.
    Accepterar alias: o/h/l/c/v, adj close, totalvolume*, turnover, etc.
    """
    if df_raw is None or df_raw.empty:
        raise RuntimeError("Tom DataFrame tillbaka.")

    # Sätt datumindex
    tmp = df_raw.copy()
    # Försök hitta datumkolumn
    date_col = None
    for cand in ("date","datum","d","time","timestamp"):
        for c in tmp.columns:
            if c.strip().lower() == cand:
                date_col = c
                break
        if date_col:
            break
    if date_col:
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", utc=False)
        tmp = tmp.set_index(date_col)
    # Om redan index men inte datetime, försök konvertera
    if not isinstance(tmp.index, pd.DatetimeIndex):
        try:
            tmp.index = pd.to_datetime(tmp.index, errors="coerce", utc=False)
        except Exception:
            pass
    # Rensa null-datum
    if isinstance(tmp.index, pd.DatetimeIndex):
        tmp = tmp[~tmp.index.isna()]

    # Plocka OHLC
    o = _pick(tmp, "open","o")
    h = _pick(tmp, "high","h")
    l = _pick(tmp, "low","l")

    # Close (prioritera "close", annars "adj close", annars "c")
    c = _pick(tmp, "close","adj close","adj_close","c","closing price")
    if c is None:
        raise RuntimeError("Saknar Close-kolumn.")

    # Volume: många varianter
    v = _pick(tmp, "volume","vol","v","totalvolume","total_volume",
              "totalvolumetraded","total_volume_traded")
    if v is None:
        # Försök räkna volym från omsättning/close
        turnover = _pick(tmp, "turnover","value_traded","omsatt","omsatt_sek","turnover_sek")
        if turnover is not None:
            with pd.option_context("mode.use_inf_as_na", True):
                v = (turnover / c).round()
        else:
            # Sista fallback: 0
            v = pd.Series(0, index=tmp.index, name="Volume")

    # Sätt namn och bygg slut-DF
    out = pd.DataFrame({
        "Open": o if o is not None else c,   # fallback till Close om Open saknas
        "High": h if h is not None else c,
        "Low":  l if l is not None else c,
        "Close": c,
        "Volume": v,
    }, index=tmp.index)

    # Sortera och släng rader som saknar Close
    out = out.sort_index()
    out = out.dropna(subset=["Close"])
    return out

def sanitize(t: str) -> str:
    return t.strip().replace("/", "-").replace("\\", "-").replace(" ", "_")

def main():
    ap = argparse.ArgumentParser(description="Hämta och normalisera OHLCV från Börsdata; spara CSV per ticker.")
    ap.add_argument("--tickers", type=str, default="", help='Kommaseparerat: "VOLV B,ABB,ERIC B,SHB A"')
    ap.add_argument("--file", type=str, default="", help="Textfil med en ticker per rad")
    ap.add_argument("--start", type=str, default="2020-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, default="", help="Tomt = senaste")
    ap.add_argument("--outdir", type=str, default="outputs/prices", help="Default outputs/prices")
    args = ap.parse_args()

    load_env()

    tickers: List[str] = []
    if args.tickers:
        tickers += [t for t in (x.strip() for x in args.tickers.split(",")) if t]
    if args.file:
        p = Path(args.file)
        if not p.exists():
            raise FileNotFoundError(p)
        tickers += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not tickers:
        raise SystemExit("Inga tickers angivna. Använd --tickers eller --file.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = args.end or None

    ok, fail = 0, 0
    for t in tickers:
        print(f"\n=== {t} ===")
        try:
            df_raw = get_ohlcv_any(t, start, end)
            df = to_ohlcv(df_raw)
            # Skriv CSV
            fname = f"{sanitize(t)}_{start}_{(end or 'latest')}.csv"
            out_path = outdir / fname
            df.index.name = "Date"
            df.to_csv(out_path, index=True, encoding="utf-8-sig")
            print(f"✅ Sparad: {out_path}  (rader: {len(df)})")
            ok += 1
        except Exception as e:
            print(f"❌ {t}: {e}")
            fail += 1

    print(f"\nKlart. OK: {ok}  |  Fel: {fail}  |  Mapp: {outdir}")

if __name__ == "__main__":
    main()
