# -*- coding: utf-8 -*-
import json, time, sqlite3
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd

from app.data_providers import get_ohlcv as GET_OHLCV
from app.portfolio_signals import load_best_params_for_ticker, run_profile_positions

DB = Path(__file__).resolve().parents[1] / "storage" / "signals.db"
PROFILES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "opt_results"
TZ_STO = ZoneInfo("Europe/Stockholm")
TICKERS = ["HM-B.ST"]  # fyll på med fler senare

def in_trading_hours(dt):
    # Vardagar 09:00–17:35 (Stockholm)
    wd = dt.weekday()
    return wd < 5 and (dt.hour > 8 and (dt.hour < 17 or (dt.hour == 17 and dt.minute <= 35)))

def next_5min_sleep():
    now = datetime.now(tz=timezone.utc)
    return 300 - (now.minute*60 + now.second) % 300

def write_signal(con, ticker: str, signal: str, price=None, meta=None):
    con.execute(
        "INSERT INTO signals(ts,ticker,signal,price,meta) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), ticker, signal, price, json.dumps(meta or {}))
    )
    con.commit()

def last_signal(con, ticker: str):
    row = con.execute("SELECT signal FROM signals WHERE ticker=? ORDER BY id DESC LIMIT 1", (ticker,)).fetchone()
    return row[0] if row else None

def current_state_from_profile(ticker: str, start: date) -> tuple[str, float, dict]:
    """Returnerar (signal, price, meta) baserat på profilens dagsdata t.o.m. idag."""
    # 1) Hämta profilparametrar
    params, metrics, prof_name, prof_path = load_best_params_for_ticker(ticker, PROFILES_DIR)

    # 2) Beräkna positionsserie (daglig)
    s = run_profile_positions(ticker, params, start)
    in_pos = bool(len(s) and s.iloc[-1] >= 0.5)

    # 3) Pris = senaste Close från dagsdatan
    df = GET_OHLCV(ticker, start=start, source="borsdata")
    price = float(df["Close"].iloc[-1]) if len(df) else None

    meta = {
        "profile": prof_name,
        "profile_file": prof_path.name,
        "TotalReturn": float(metrics.get("TotalReturn", 0.0)),
        "source": "borsdata",
        "interval": "1D",
    }
    # signal = “state” just nu (BUY/FLAT). SELL hanteras via jämförelse mot förra raden.
    return ("BUY" if in_pos else "FLAT", price, meta)

def run_once():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL;")
    start = date(2020, 1, 1)

    now_sto = datetime.now(TZ_STO)
    # Vi skriver alltid en snapshot (även utanför handelstid), men du kan begränsa om du vill:
    # if not in_trading_hours(now_sto): return

    for t in TICKERS:
        state_now, price, meta = current_state_from_profile(t, start)
        prev = last_signal(con, t)

        # Mappa state->event: om prev var BUY och nu FLAT => SELL, om prev != BUY och nu BUY => BUY, annars FLAT
        if prev == "BUY" and state_now == "FLAT":
            signal = "SELL"
        elif prev != "BUY" and state_now == "BUY":
            signal = "BUY"
        else:
            signal = "FLAT"

        # Skriv alltid snapshot, markera ändring i meta
        m = dict(meta)
        if signal in ("BUY","SELL"):
            m["changed_from"] = prev
        m["snapshot"] = True

        write_signal(con, t, signal, price, m)
        print(f"[SNAP] {t}: prev={prev or '—'} state={state_now} -> write={signal} price={price}", flush=True)

if __name__ == "__main__":
    DB.parent.mkdir(parents=True, exist_ok=True)
    run_once()   # oneshot; systemd-timern kör detta var 5:e minut
