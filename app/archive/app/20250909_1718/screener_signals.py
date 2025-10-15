# app/screener_signals.py
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from app.bd_legacy_client import BDClient


# ---------- indikatorer ----------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def breakout_high(high: pd.Series, window: int = 20) -> pd.Series:
    return high >= high.rolling(window).max().shift(1)


# ---------- scoring ----------

def score_today(df: pd.DataFrame) -> dict:
    if df.shape[0] < 50:
        return {"ok": False, "reason": "insufficient_data"}

    close = df["Close"]
    high = df["High"]
    vol = df["Volume"]

    # RSI / MACD
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi14 = (100 - (100 / (1 + rs))).fillna(50.0)

    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()

    i = df.index[-1]
    i_prev = df.index[-2]

    rsi_cross_up = (rsi14.loc[i_prev] <= 50) and (rsi14.loc[i] > 50)
    macd_cross_up = (macd_line.loc[i_prev] <= macd_sig.loc[i_prev]) and (macd_line.loc[i] > macd_sig.loc[i])
    is_breakout = bool((high >= high.rolling(20).max().shift(1)).loc[i])

    vol20 = vol.rolling(20).mean()
    vol_spike = 0.0
    if pd.notna(vol.loc[i]) and pd.notna(vol20.loc[i]) and vol20.loc[i] > 0:
        vol_spike = max(0.0, min(2.0, (vol.loc[i] / vol20.loc[i]) - 1.0))  # 0..2

    rsi_contrib = max(0.0, min(1.0, (rsi14.loc[i] - 50.0) / 30.0))  # 50→80 => 0..1

    w_rsi, w_macd, w_brk, w_vol = 0.40, 0.30, 0.20, 0.10
    score = w_rsi * rsi_contrib + w_macd * (1.0 if macd_cross_up else 0.0) + w_brk * (1.0 if is_breakout else 0.0) + w_vol * vol_spike

    return {
        "ok": True,
        "score": round(float(score), 4),
        "rsi": round(float(rsi14.loc[i]), 2),
        "rsi_cross_up": bool(rsi_cross_up),
        "macd_cross_up": bool(macd_cross_up),
        "breakout20": bool(is_breakout),
        "vol_spike": round(float(vol_spike), 3),
        "date": str(i.date()),
        "close": float(close.loc[i]),
    }


# ---------- dataladdning ----------

def df_from_bd_prices(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["d"])
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").set_index("Date")


def lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


def ins_core(it: Dict[str, Any]) -> Dict[str, Any]:
    ik = lower_keys(it)
    return {
        "ticker": ik.get("ticker") or ik.get("symbol"),
        "name": ik.get("name") or ik.get("companyname"),
        "mic": ik.get("mic") or ik.get("exchangemic") or ik.get("primarymic"),
        "insid": ik.get("insid") or ik.get("instrumentid") or ik.get("id"),
    }


def norm_mic(m: Optional[str]) -> Optional[str]:
    """
    Tomt, 'ALL', 'ANY' eller '*' betyder 'ingen filtrering'.
    """
    if m is None:
        return None
    m = m.strip()
    if m == "" or m.upper() in {"ALL", "ANY", "*"}:
        return None
    return m


# ---------- main ----------

def run(mic: Optional[str], limit: int, days: int, min_score: float) -> pd.DataFrame:
    bd = BDClient()
    inst = bd.instruments()
    rows: List[Dict[str, Any]] = []

    scanned = 0
    total = len(inst)
    print(f"[INFO] Startar screener: total={total}, mic={mic or 'ALL'}, limit={limit}, days={days}, min_score={min_score}")
    sys.stdout.flush()

    mic_u = norm_mic(mic)
    mic_u = mic_u.upper() if mic_u else None

    for it in inst:
        core = ins_core(it)

        # Snällt MIC-filter: filtrera endast om instrumentet HAR MIC och den inte matchar mic_u
        if mic_u:
            ins_mic = str(core["mic"] or "").upper()
            if ins_mic and ins_mic != mic_u:
                continue  # har mic och den matchar inte → hoppa
            # saknar mic → släpp igenom

        if core["insid"] is None or not core["ticker"]:
            continue

        try:
            pr = bd.prices(int(core["insid"]), max_count=days)
            df = df_from_bd_prices(pr)
            if df.empty or len(df) < 50:
                pass
            else:
                sc = score_today(df)
                if sc.get("ok") and sc["score"] >= min_score:
                    rows.append(
                        {
                            "InsId": int(core["insid"]),
                            "Ticker": core["ticker"],
                            "Name": core["name"],
                            "MIC": core["mic"],
                            "Date": sc["date"],
                            "Close": sc["close"],
                            "Score": sc["score"],
                            "RSI": sc["rsi"],
                            "RSI_cross_up": sc["rsi_cross_up"],
                            "MACD_cross_up": sc["macd_cross_up"],
                            "Breakout20": sc["breakout20"],
                            "Vol_spike": sc["vol_spike"],
                        }
                    )
        except Exception:
            time.sleep(0.2)

        scanned += 1
        if scanned % 10 == 0:
            print(f"[INFO] Skannat {scanned} instrument... (träffar hittills: {len(rows)})")
            sys.stdout.flush()
        if limit and scanned >= limit:
            break

    cols = ["Ticker", "Name", "MIC", "Date", "Close", "Score", "RSI", "RSI_cross_up", "MACD_cross_up", "Breakout20", "Vol_spike"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["Score", "Vol_spike", "RSI"], ascending=[False, False, False]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Börsdata-köpsignal-screener (EOD, idag)")
    p.add_argument("--mic", default="XSTO", help="Börsplats (MIC). Tom/ALL/ANY/* = ingen filtrering.")
    p.add_argument("--limit", type=int, default=150, help="Max antal instrument att skanna (sparar rate limits).")
    p.add_argument("--days", type=int, default=400, help="Antal dagar historik per instrument.")
    p.add_argument("--min-score", type=float, default=0.50, help="Minsta score för att tas med (0..1).")
    args = p.parse_args()

    os.makedirs(".\\outputs", exist_ok=True)
    today = dt.date.today().isoformat()
    out_path = os.path.abspath(f".\\outputs\\signals_{(args.mic or 'ALL').replace('*','ALL')}_{today}.csv")

    df = run(mic=(args.mic or None), limit=args.limit, days=args.days, min_score=args.min_score)
    df.to_csv(out_path, index=False)  # spara alltid

    if df.empty:
        print(f"[INFO] Inga starka köpsignaler — CSV skapad (tom): {out_path}")
        return

    print(df.head(20).to_string(index=False))
    print(f"\n[OK] Sparat: {out_path}")


if __name__ == "__main__":
    main()
