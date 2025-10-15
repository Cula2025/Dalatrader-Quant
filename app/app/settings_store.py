# app/settings_store.py
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_DIR = Path("storage")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "settings.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticker_settings (
  ticker TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_DEFAULTS: Dict[str, Any] = {
    # allm�nt
    "strategy": "breakout",      # "breakout" | "sma_cross" | "rsi_band"
    "start": "2018-01-01",
    "end": "",
    "commission_bps": 5.0,
    "slippage_bps": 5.0,

    # breakout
    "breakout_lookback": 20,
    "exit_lookback": 10,

    # SMA cross
    "sma_fast": 20,
    "sma_slow": 50,

    # RSI-filter (f�r breakout & SMA)
    "use_rsi_gate": False,
    "rsi_period": 14,
    "rsi_entry_min": 55.0,   # entry kr�ver RSI =
    "rsi_exit_max": 45.0,    # exit om RSI =

    # RSI-band (ren RSI-strategi)
    "rsi_band_period": 14,
    "rsi_band_entry_level": 50.0,
    "rsi_band_exit_level": 45.0,

    # Riskhantering
    "use_risk": False,           # SL/TP fr�n entry
    "stop_loss_pct": 0.08,       # 8%
    "take_profit_pct": 0.15,     # 15%

    # Trailing stop (golv fr�n h�gsta sedan entry)
    "use_trailing": False,
    "trailing_pct": 0.08,        # 8%

    # NYTT: Viloperiod efter STOP-LOSS (kalenderdagar)
    "use_cooldown": False,
    "cooldown_days": 14,         # 14�30 �r rimligt
}

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(_SCHEMA)
    return con

def get_settings(ticker: str) -> Dict[str, Any]:
    t = ticker.strip()
    with _conn() as con:
        row = con.execute("SELECT payload FROM ticker_settings WHERE ticker=?", (t,)).fetchone()
    data = json.loads(row[0]) if row else {}
    out = {**_DEFAULTS, **(data or {})}
    return out

def save_settings(ticker: str, data: Dict[str, Any]) -> None:
    t = ticker.strip()
    payload = json.dumps(data, ensure_ascii=False)
    with _conn() as con:
        con.execute(
            "INSERT INTO ticker_settings (ticker, payload) VALUES (?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET payload=excluded.payload, updated_at=CURRENT_TIMESTAMP",
            (t, payload),
        )

def delete_settings(ticker: str) -> None:
    t = ticker.strip()
    with _conn() as con:
        con.execute("DELETE FROM ticker_settings WHERE ticker=?", (t,))

def list_tickers() -> List[str]:
    with _conn() as con:
        rows = con.execute("SELECT ticker FROM ticker_settings ORDER BY ticker").fetchall()
    return [r[0] for r in rows]






