from __future__ import annotations
import os, sys, json, math
from typing import Dict, Tuple
import pandas as pd

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def get_df(ticker: str, p: Dict) -> pd.DataFrame:
    from app.data_providers import get_ohlcv as GET
    try:
        return GET(ticker, start=p.get("from_date"), end=p.get("to_date"))
    except TypeError:
        return GET(ticker, p.get("from_date"), p.get("to_date"))

def _to_numeric_series(x) -> pd.Series:
    """Försök göra om vad som helst till en numerisk pd.Series."""
    if isinstance(x, pd.Series):
        s = pd.to_numeric(x, errors="coerce")
        return s.dropna()
    if isinstance(x, pd.DataFrame):
        # Försök kända kolumnnamn först
        for c in ("equity","Equity","value","values"):
            if c in x.columns:
                s = pd.to_numeric(x[c], errors="coerce").dropna()
                if s.size > 1:
                    return s
        # Annars första numeriska kolumn
        for c in x.columns:
            s = pd.to_numeric(x[c], errors="coerce").dropna()
            if s.size > 1:
                return s
        return pd.Series(dtype="float64")
    if isinstance(x, (list, tuple)):
        if len(x) and isinstance(x[0], (int, float)):
            return pd.Series(x, dtype="float64")
        if len(x) and isinstance(x[0], dict):
            for vk in ("y","value","equity","v"):
                vals = [row.get(vk) for row in x]
                s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
                if s.size > 1:
                    return s
        return pd.Series(dtype="float64")
    if isinstance(x, dict):
        # format med "values": [...]
        if "values" in x:
            s = pd.to_numeric(pd.Series(x["values"]), errors="coerce").dropna()
            return s
    # sista utväg: försök tolka direkt
    try:
        return pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype="float64")

def extract_equity(res) -> pd.Series:
    """Hämta numerisk equity-serie oavsett returformat från backtest.run_backtest."""
    candidates = []
    if isinstance(res, dict):
        for k in ("equity","equity_curve","equitySeries","Equity","curve","equity_vec"):
            if k in res:
                candidates.append(res[k])
                break
        else:
            # kanske hela res ÄR serien i annan nyckel – prova alla values
            candidates.extend(res.values())
    else:
        candidates.append(res)

    for v in candidates:
        s = _to_numeric_series(v)
        if s.size > 1:
            # sortera på index men använd värdena (skulle de råka vara timestamps)
            try:
                s = s.sort_index()
            except Exception:
                pass
            # normalisera om första är 0
            first = float(s.iloc[0]) if s.iloc[0] != 0 else 1.0
            if first == 0:
                first = 1.0
            return (s / first)
    raise ValueError("Kunde inte extrahera numerisk equity från run_backtest-resultatet.")

def buyhold_factor(df: pd.DataFrame) -> float:
    for c in ("Adj Close","adj_close","Close","close","c"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) >= 2:
                return float(s.iloc[-1] / s.iloc[0])
    raise ValueError("Hittade ingen pris-kolumn för BH.")

def strat_factor(df: pd.DataFrame, params: Dict) -> Tuple[float, int]:
    from backtest import run_backtest as RUN
    res = RUN(df, dict(params))
    s = extract_equity(res)
    return float(s.iloc[-1]), int(len(s))  # s är redan normaliserad till start=1

def near(a: float, b: float, tol=0.02) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    if b == 0:
        return abs(a-b) < tol
    return abs(a/b - 1.0) <= tol

def check_file(path: str) -> int:
    data = json.loads(open(path,"r",encoding="utf-8-sig").read())
    profs = data.get("profiles", [])
    if not profs:
        print(f"[FAIL] {path}: saknar profiler.")
        return 2

    rc = 0
    for i, p in enumerate(profs, 1):
        name   = p.get("name","?")
        ticker = p.get("ticker","?")
        params = dict(p.get("params", {}))
        met    = dict(p.get("metrics", {}))
        print(f"\n== {os.path.basename(path)} :: {i}/{len(profs)} :: {name} ({ticker}) ==")

        df = get_df(ticker, params)
        if df is None or len(df) < 2:
            print("[FAIL]  Ingen data från Börsdata.")
            rc = max(rc, 2); continue

        # Buy & Hold
        try:
            bh = buyhold_factor(df)
            bh_facit = (float(met.get("BuyHold", 0.0)) + 1.0) if met else None
            if bh_facit is not None:
                print(f"BH:     mätt={bh:.4f}×, facit={bh_facit:.4f}×, ok={near(bh,bh_facit)}")
            else:
                print(f"BH:     mätt={bh:.4f}× (facit saknas)")
        except Exception as e:
            print(f"[WARN] BH kunde inte räknas: {e}")
            bh = float('nan')

        # Strategi
        try:
            sf, n = strat_factor(df, params)
            tr_facit = (float(met.get("TotalReturn", float('nan'))) + 1.0) if met else None
            if tr_facit is not None and math.isfinite(tr_facit):
                is_ok = near(sf, tr_facit)
                print(f"STRAT:  mätt={sf:.4f}× över {n} steg, facit={tr_facit:.4f}×, ok={is_ok}")
                if not is_ok:
                    rc = max(rc, 1)
            else:
                print(f"STRAT:  mätt={sf:.4f}× över {n} steg (facit saknas)")
        except Exception as e:
            print(f"[FAIL] STRAT fel: {type(e).__name__}: {e}")
            rc = max(rc, 2)
    return rc

def main(args):
    paths = args or [p for p in sorted(os.listdir("profiles")) if p.endswith(".json")]
    paths = [p if p.startswith("profiles/") else f"profiles/{p}" for p in paths]
    worst = 0
    for p in paths:
        r = check_file(p)
        worst = max(worst, r)
    return worst

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
