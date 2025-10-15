# -*- coding: utf-8 -*-
import json, glob
import pandas as pd
import numpy as np
from pathlib import Path

files = sorted(glob.glob(r".\outputs\grid_*_medium.csv"))
rows = []

for fp in files:
    try:
        df = pd.read_csv(fp)
        if df.empty:
            continue

        sort_cols = [c for c in ["SharpeD", "TotalReturn", "MaxDD"] if c in df.columns]
        if sort_cols:
            asc = [False, False, False][:len(sort_cols)]
            df = df.sort_values(sort_cols, ascending=asc)

        best = df.iloc[0].copy()

        # Risk 1-5: baserat pa MaxDD + PF-bonus
        maxdd = float(best.get("MaxDD", 0.0))
        pf = float(best.get("PF", 1.0))
        if maxdd > -0.10:
            dd_score = 1
        elif maxdd > -0.20:
            dd_score = 2
        elif maxdd > -0.30:
            dd_score = 3
        elif maxdd > -0.40:
            dd_score = 4
        else:
            dd_score = 5
        pf_bonus = -1 if pf >= 2.0 else (-0.5 if pf >= 1.5 else 0)
        risk = int(min(5, max(1, dd_score + pf_bonus)))

        p = best.to_dict()
        def get(k, default=None):
            return p.get(k, default)

        strat = str(get("strategy") or get("Strategy") or "breakout")

        def clean_int(x):
            try:
                return int(x) if not pd.isna(x) else None
            except Exception:
                return None

        def clean_float(x):
            try:
                return float(x) if not pd.isna(x) else None
            except Exception:
                return None

        def clean_bool(x, default=False):
            if pd.isna(x):
                return default
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            if isinstance(x, (int, float)):
                return bool(int(x))
            if isinstance(x, str):
                return x.strip().lower() in ("1", "true", "yes", "y")
            return default

        apply_params = {
            "strategy": strat,
            "breakout_lookback": clean_int(get("breakout_lookback")),
            "exit_lookback": clean_int(get("exit_lookback")),
            "use_trend_filter": clean_bool(get("use_trend_filter"), False),
            "trend_ma_window": clean_int(get("trend_ma_window")),
            "atr_period": int(get("atr_period")) if not pd.isna(get("atr_period")) else 14,
            "use_atr_momentum": clean_bool(get("use_atr_momentum"), False),
            "atr_momentum_mult": clean_float(get("atr_momentum_mult")) or 0.0,
            "breakeven_atr_mult": clean_float(get("breakeven_atr_mult")) or 0.0,
            "stop_loss_pct": clean_float(get("stop_loss_pct")),
            "take_profit_pct": clean_float(get("take_profit_pct")),
            "trailing_stop_pct": clean_float(get("trailing_stop_pct")),
            "cooldown_days_after_sl": int(get("cooldown_days_after_sl")) if not pd.isna(get("cooldown_days_after_sl")) else 0,
            "risk_per_trade_pct": float(get("risk_per_trade_pct")) if not pd.isna(get("risk_per_trade_pct")) else 1.0,
        }

        ticker_guess = best.get("Ticker")
        if not isinstance(ticker_guess, str) or ticker_guess.strip() == "":
            ticker_guess = Path(fp).stem.replace("grid_", "").replace("_medium", "").replace("_", " ")

        rows.append({
            "Ticker": ticker_guess,
            "Strategy": strat,
            "Trades": int(best.get("Trades", 0)),
            "WinRate": float(best.get("WinRate", 0.0)),
            "PF": float(best.get("PF", 0.0)),
            "TotalReturn": float(best.get("TotalReturn", 0.0)),
            "MaxDD": float(best.get("MaxDD", 0.0)),
            "SharpeD": float(best.get("SharpeD", 0.0)),
            "Risk": risk,
            "ApplyParams": json.dumps(apply_params),
            "Source": fp,
        })
    except Exception as e:
        print("[WARN] Skip", fp, ":", str(e))

res = pd.DataFrame(rows)
if not res.empty:
    res = res.sort_values(["SharpeD", "TotalReturn", "MaxDD"], ascending=[False, False, False])
    out_fp = r".\outputs\top10_summary.csv"
    res.to_csv(out_fp, index=False)
    print("OK ->", out_fp)
    print(res[["Ticker","Strategy","Trades","TotalReturn","MaxDD","SharpeD","Risk"]].to_string(index=False))
else:
    print("No results found - make sure grid files exist and are non-empty.")
