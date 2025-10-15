# app/grid_tuner.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools as it

import numpy as np
import pandas as pd

from app.backtest_simple import (
    load_ohlcv, run_backtest, perf_stats
)

# ---------- HjÃ¤lp: YoY & tradestatistik --------------------------------------

def yoy_stats(eq: pd.DataFrame) -> Dict[str, float]:
    """Year-over-Year statistik pÃ¥ equity-kurvan."""
    if eq.empty:
        return {"YoY_min": 0.0, "YoY_max": 0.0, "YoY_mean": 0.0, "YoY_neg_years": 0}
    yearly_last = eq["Equity"].resample("Y").last()
    yoy = yearly_last.pct_change().dropna()
    if yoy.empty:
        return {"YoY_min": 0.0, "YoY_max": 0.0, "YoY_mean": 0.0, "YoY_neg_years": 0}
    return {
        "YoY_min": float(yoy.min()),
        "YoY_max": float(yoy.max()),
        "YoY_mean": float(yoy.mean()),
        "YoY_neg_years": int((yoy < 0).sum()),
    }

def trade_stats(trades: List[dict]) -> Dict[str, float]:
    if not trades:
        return {"Trades": 0, "WinRate": 0.0, "PF": 0.0, "AvgRet": 0.0}
    df = pd.DataFrame(trades)
    wins = df[df["ret"] > 0]
    losses = df[df["ret"] <= 0]
    winrate = 0.0 if len(df) == 0 else len(wins) / len(df)
    gross_win = float(wins["pnl"].sum())
    gross_loss = float((-losses["pnl"]).sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (np.nan if gross_win == 0 else np.inf)
    return {
        "Trades": int(len(df)),
        "WinRate": float(winrate),
        "PF": float(pf if np.isfinite(pf) else 0.0),
        "AvgRet": float(df["ret"].mean()) if "ret" in df else 0.0,
    }

# ---------- Signaler (inbyggda hÃ¤r fÃ¶r att minska beroenden) -----------------

def build_signals(df: pd.DataFrame, strategy: str,
                  breakout_lookback: int = 20, exit_lookback: int = 10) -> pd.DataFrame:
    """Returnerar df med kolumnerna long_entry/long_exit."""
    if strategy == "breakout":
        out = df.copy()
        out["hi"] = out["High"].rolling(int(breakout_lookback)).max().shift(1)
        out["lo"] = out["Low"].rolling(int(exit_lookback)).min().shift(1)
        out["long_entry"] = (out["Close"] > out["hi"]).astype(int)
        out["long_exit"]  = (out["Close"] < out["lo"]).astype(int)
        return out
    elif strategy == "sma_cross":
        out = df.copy()
        out["sma20"] = out["Close"].rolling(20).mean()
        out["sma50"] = out["Close"].rolling(50).mean()
        prev_up = (out["sma20"].shift(1) > out["sma50"].shift(1))
        now_up  = (out["sma20"] > out["sma50"])
        out["long_entry"] = ((~prev_up) &  now_up).astype(int)
        out["long_exit"]  = ( prev_up  & (~now_up)).astype(int)
        return out
    else:
        raise ValueError("strategy mÃ¥ste vara 'breakout' eller 'sma_cross'")

# ---------- Griddef -----------------------------------------------------------

def grid_params(mode: str = "medium") -> Dict[str, List]:
    """
    Diskreta vÃ¤rdelistor fÃ¶r parametrar.
    'small' ~ snabb, 'medium' ~ rimlig, 'wide' ~ bred (kan ta lÃ¤ngre tid).
    """
    if mode == "small":
        return dict(
            strategy=["breakout"],
            breakout_lookback=[20],
            exit_lookback=[10],
            use_trend_filter=[True],
            trend_ma_window=[100],
            atr_period=[14],
            use_atr_momentum=[True],
            atr_momentum_mult=[0.5],
            breakeven_atr_mult=[1.0],
            stop_loss_pct=[0.08],
            take_profit_pct=[0.15],
            trailing_stop_pct=[None, 0.08],
            cooldown_days_after_sl=[14],
            risk_per_trade_pct=[0.5],
        )
    if mode == "wide":
        return dict(
            strategy=["breakout","sma_cross"],
            breakout_lookback=[10,20,30,50],
            exit_lookback=[5,10,20],
            use_trend_filter=[True, False],
            trend_ma_window=[50,100,200],
            atr_period=[14],
            use_atr_momentum=[True, False],
            atr_momentum_mult=[0.0, 0.3, 0.5, 0.8],
            breakeven_atr_mult=[0.0, 0.5, 1.0],
            stop_loss_pct=[None, 0.06, 0.08, 0.10],
            take_profit_pct=[None, 0.12, 0.15, 0.20],
            trailing_stop_pct=[None, 0.06, 0.08, 0.10],
            cooldown_days_after_sl=[0, 14, 30],
            risk_per_trade_pct=[0.0, 0.5],
        )
    # default: medium
    return dict(
        strategy=["breakout"],
        breakout_lookback=[20, 30],
        exit_lookback=[10],
        use_trend_filter=[True],
        trend_ma_window=[100],
        atr_period=[14],
        use_atr_momentum=[True],
        atr_momentum_mult=[0.3, 0.5, 0.8],
        breakeven_atr_mult=[0.0, 1.0],
        stop_loss_pct=[0.06, 0.08, 0.10],
        take_profit_pct=[0.12, 0.15, 0.20],
        trailing_stop_pct=[None, 0.06, 0.08, 0.10],
        cooldown_days_after_sl=[0, 14],
        risk_per_trade_pct=[0.5],
    )

# ---------- KÃ¶rning -----------------------------------------------------------

@dataclass
class Row:
    Ticker: str
    Strategy: str
    Start: str
    End: str
    breakout_lookback: int
    exit_lookback: int
    use_trend_filter: bool
    trend_ma_window: int
    atr_period: int
    use_atr_momentum: bool
    atr_momentum_mult: float
    breakeven_atr_mult: float
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]
    trailing_stop_pct: Optional[float]
    cooldown_days_after_sl: int
    risk_per_trade_pct: float
    Trades: int
    WinRate: float
    PF: float
    AvgRet: float
    TotalReturn: float
    CAGR: float
    MaxDD: float
    SharpeD: float
    YoY_min: float
    YoY_max: float
    YoY_mean: float
    YoY_neg_years: int

def main():
    ap = argparse.ArgumentParser(description="Grid-tuner fÃ¶r BÃ¶rsdata, inkl. Y/Y-analys")
    ap.add_argument("--ticker", required=True, help='Ex: "SEB A" eller "SEB C"')
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--mode", choices=["small","medium","wide"], default="medium")
    ap.add_argument("--out", default=None, help="CSV-sÃ¶kvÃ¤g (default: outputs/grid_<TICKER>_<MODE>.csv)")
    ap.add_argument("--max-combos", type=int, default=None, help="(valfritt) slumpmÃ¤ssigt tak pÃ¥ antal kombos")
    args = ap.parse_args()

    # HÃ¤mta data
    print(f"[INFO] HÃ¤mtar {args.ticker} frÃ¥n BÃ¶rsdataâ€¦")
    df = load_ohlcv(args.ticker, days=20000, start=args.start, end=args.end)
    if df.shape[0] < 60:
        raise RuntimeError(f"FÃ¶r fÃ¥ rader ({len(df)}) efter filtrering.")

    # Bygg grid
    G = grid_params(args.mode)
    keys = list(G.keys())
    combos = list(it.product(*[G[k] for k in keys]))
    print(f"[INFO] Totalt {len(combos)} kombinationer i '{args.mode}'-grid.")

    if args.max_combos and args.max_combos > 0 and len(combos) > args.max_combos:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(combos), size=args.max_combos, replace=False)
        combos = [combos[i] for i in idx]
        print(f"[INFO] Sampplar ned till {len(combos)} kombinationer.")

    rows: List[Row] = []

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))

        # Bygg signaler
        sig = build_signals(
            df, params["strategy"],
            breakout_lookback=int(params["breakout_lookback"]),
            exit_lookback=int(params["exit_lookback"]),
        )

        # KÃ¶r backtest
        eq, trades = run_backtest(
            sig,
            ticker=args.ticker,
            initial_capital=100_000.0,
            commission_bps=5.0,
            slippage_bps=5.0,
            stop_loss_pct=params["stop_loss_pct"],
            take_profit_pct=params["take_profit_pct"],
            trailing_stop_pct=params["trailing_stop_pct"],
            cooldown_days_after_sl=int(params["cooldown_days_after_sl"]),
            use_trend_filter=bool(params["use_trend_filter"]),
            trend_ma_window=int(params["trend_ma_window"]),
            atr_period=int(params["atr_period"]),
            use_atr_momentum=bool(params["use_atr_momentum"]),
            atr_momentum_mult=float(params["atr_momentum_mult"]),
            breakeven_atr_mult=float(params["breakeven_atr_mult"]),
            risk_per_trade_pct=float(params["risk_per_trade_pct"]),
        )

        stats = perf_stats(eq)
        yoy = yoy_stats(eq)
        tstats = trade_stats([t.__dict__ for t in trades])

        rows.append(Row(
            Ticker=args.ticker, Strategy=params["strategy"],
            Start=args.start, End=args.end or "latest",
            breakout_lookback=int(params["breakout_lookback"]),
            exit_lookback=int(params["exit_lookback"]),
            use_trend_filter=bool(params["use_trend_filter"]),
            trend_ma_window=int(params["trend_ma_window"]),
            atr_period=int(params["atr_period"]),
            use_atr_momentum=bool(params["use_atr_momentum"]),
            atr_momentum_mult=float(params["atr_momentum_mult"]),
            breakeven_atr_mult=float(params["breakeven_atr_mult"]),
            stop_loss_pct=(None if params["stop_loss_pct"] is None else float(params["stop_loss_pct"])),
            take_profit_pct=(None if params["take_profit_pct"] is None else float(params["take_profit_pct"])),
            trailing_stop_pct=(None if params["trailing_stop_pct"] is None else float(params["trailing_stop_pct"])),
            cooldown_days_after_sl=int(params["cooldown_days_after_sl"]),
            risk_per_trade_pct=float(params["risk_per_trade_pct"]),
            Trades=int(tstats["Trades"]), WinRate=float(tstats["WinRate"]),
            PF=float(tstats["PF"]), AvgRet=float(tstats["AvgRet"]),
            TotalReturn=float(stats["TotalReturn"]), CAGR=float(stats["CAGR"]),
            MaxDD=float(stats["MaxDD"]), SharpeD=float(stats["SharpeD"]),
            YoY_min=float(yoy["YoY_min"]), YoY_max=float(yoy["YoY_max"]),
            YoY_mean=float(yoy["YoY_mean"]), YoY_neg_years=int(yoy["YoY_neg_years"]),
        ))

        if i % 50 == 0:
            print(f"[INFO] KÃ¶rt {i}/{len(combos)} â€¦")

    # Spara CSV
    outdir = Path("outputs"); outdir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (outdir / f"grid_{args.ticker.replace(' ','_')}_{args.mode}.csv")
    df_out = pd.DataFrame([r.__dict__ for r in rows])

    # Sortera: bÃ¤sta fÃ¶rst (hÃ¶g Sharpe, mindre negativ MaxDD, hÃ¶g CAGR)
    df_out = df_out.sort_values(by=["SharpeD", "MaxDD", "CAGR"], ascending=[False, False, False])
    df_out.to_csv(out, index=False)
    print(f"[OK] Sparat grid-resultat: {out.resolve()}")

    # Visa topp-15 i konsolen
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df_out.head(15).to_string(index=False))

if __name__ == "__main__":
    main()


