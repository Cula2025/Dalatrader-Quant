from __future__ import annotations
import random
from typing import Dict, Any

RANGES = {
    "RSI": {
        "rsi_window": (7, 21),
        "rsi_min": (20.0, 35.0),
        "rsi_max": (55.0, 80.0),
        "breakout_lookback": (40, 70),
        "exit_lookback": (10, 30),
    },
    "MA-CROSS": {
        "fast": (5, 20),
        "slow": (50, 150),
        "breakout_lookback": (30, 80),
        "exit_lookback": (10, 30),
    },
    "MACD-SIGNAL": {
        "macd_fast": (10, 14),
        "macd_slow": (22, 30),
        "macd_signal": (7, 12),
        "breakout_lookback": (30, 80),
        "exit_lookback": (10, 30),
    },
    "MOMENTUM": {
        "mom_window": (10, 60),
        "mom_thresh": (0.0, 0.02),
        "breakout_lookback": (30, 80),
        "exit_lookback": (10, 30),
    },
}

def _ri(rng, a, b): return rng.randint(a, b)
def _rf(rng, a, b): return rng.uniform(a, b)

def sample_params(strategy: str, rng: random.Random) -> Dict[str, Any]:
    s = strategy.upper()
    if s not in RANGES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    r = RANGES[s]
    p: Dict[str, Any] = {"strategy": s}

    if s == "RSI":
        p["rsi_window"] = _ri(rng, *RANGES["RSI"]["rsi_window"])
        p["rsi_min"] = _rf(rng, *RANGES["RSI"]["rsi_min"])
        p["rsi_max"] = _rf(rng, *RANGES["RSI"]["rsi_max"])
        p["breakout_lookback"] = _ri(rng, *RANGES["RSI"]["breakout_lookback"])
        p["exit_lookback"] = _ri(rng, *RANGES["RSI"]["exit_lookback"])
    elif s == "MA-CROSS":
        fast = _ri(rng, *RANGES["MA-CROSS"]["fast"])
        slow = _ri(rng, *RANGES["MA-CROSS"]["slow"])
        if fast >= slow:
            fast = max(5, slow - 5)
        p["fast"], p["slow"] = fast, slow
        p["breakout_lookback"] = _ri(rng, *RANGES["MA-CROSS"]["breakout_lookback"])
        p["exit_lookback"] = _ri(rng, *RANGES["MA-CROSS"]["exit_lookback"])
    elif s == "MACD-SIGNAL":
        p["macd_fast"] = _ri(rng, *RANGES["MACD-SIGNAL"]["macd_fast"])
        p["macd_slow"] = _ri(rng, *RANGES["MACD-SIGNAL"]["macd_slow"])
        p["macd_signal"] = _ri(rng, *RANGES["MACD-SIGNAL"]["macd_signal"])
        p["breakout_lookback"] = _ri(rng, *RANGES["MACD-SIGNAL"]["breakout_lookback"])
        p["exit_lookback"] = _ri(rng, *RANGES["MACD-SIGNAL"]["exit_lookback"])
    elif s == "MOMENTUM":
        p["mom_window"] = _ri(rng, *RANGES["MOMENTUM"]["mom_window"])
        p["mom_thresh"] = _rf(rng, *RANGES["MOMENTUM"]["mom_thresh"])
        p["breakout_lookback"] = _ri(rng, *RANGES["MOMENTUM"]["breakout_lookback"])
        p["exit_lookback"] = _ri(rng, *RANGES["MOMENTUM"]["exit_lookback"])

    # Trend gate (ofta på)
    p["use_trend_filter"] = rng.random() < 0.8
    p["trend_ma_type"] = "EMA"
    p["trend_ma_window"] = rng.randint(80, 140)

    # RSI-gate ibland även för andra strategier
    p["use_rsi_filter"] = (s != "RSI") and (rng.random() < 0.5)
    if p.get("use_rsi_filter", False):
        p.setdefault("rsi_window", rng.randint(7, 21))
        p.setdefault("rsi_min", rng.uniform(20.0, 35.0))
        p.setdefault("rsi_max", rng.uniform(55.0, 80.0))

    # BB-exit ibland
    p["use_bb_exit"] = rng.random() < 0.3
    if p["use_bb_exit"]:
        p["bb_window"] = rng.randint(18, 24)
        p["bb_nstd"] = 2.0
        p["bb_percent_b_min"] = rng.uniform(0.75, 0.9)

    return p