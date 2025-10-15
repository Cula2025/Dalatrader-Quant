from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path

# ============== Hjälpfunktioner för indikatörer =================

def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False, min_periods=w).mean()

def _ma(s: pd.Series, w: int, kind: str = "SMA") -> pd.Series:
    return _ema(s, w) if str(kind).upper() == "EMA" else _sma(s, w)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _rolling_high(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).max()

def _rolling_low(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).min()

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    return float(dd.min()) if len(dd) else 0.0

def _daily_sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std(ddof=0) == 0 or rets.empty:
        return 0.0
    return float(rets.mean() / rets.std(ddof=0) * np.sqrt(252))

# ============== Datakoll =================

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.sort_index()
    need = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df[need].copy()
    df.index = pd.to_datetime(df.index)
    return df

# ============== Huvud-backtest (en aktie, long-only) =================

@dataclass
class Trade:
    EntryDate: pd.Timestamp
    EntryPrice: float
    ExitDate: Optional[pd.Timestamp]
    ExitPrice: Optional[float]
    Qty: int
    PnL: float
    ReturnPct: float
    Reason: str

def run_backtest(
    df: pd.DataFrame,
    **params: Any
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generisk backtest som stödjer:
      - strategy: 'breakout' | 'ma_cross' | 'ema_cross' | 'rsi'
      - breakout_lookback, exit_lookback
      - fast, slow, ma_type ('SMA'|'EMA') för ma/ema_cross
      - rsi_window, rsi_entry_min, rsi_exit_max
      - use_trend_filter, trend_ma_window, trend_ma_type ('SMA'|'EMA')
      - stop_loss_pct, take_profit_pct, trailing_pct, peak_floor_pct
      - atr_period, breakeven_atr_mult, cooldown_days_after_sl
      - initial_capital, risk_per_trade_pct  (procent, t.ex. 1.0 = 1%)
    Returnerar: (equity_df, trades_df, stats_dict)
    """
    # -------- defaults --------
    p = {
        "strategy": "breakout",

        "breakout_lookback": 20,
        "exit_lookback": 10,

        "fast": 20,
        "slow": 50,
        "ma_type": "SMA",

        "rsi_window": 14,
        "rsi_entry_min": 0,   # 0 = inaktiv
        "rsi_exit_max": 0,    # 0 = inaktiv

        "use_trend_filter": False,
        "trend_ma_window": 100,
        "trend_ma_type": "SMA",

        "stop_loss_pct": None,       # t.ex. 0.08
        "take_profit_pct": None,     # t.ex. 0.20
        "trailing_pct": None,        # t.ex. 0.10
        "peak_floor_pct": 0.0,       # t.ex. 0.10
        "atr_period": 14,
        "breakeven_atr_mult": 0.0,   # t.ex 1.0 -> flytta SL till breakeven efter +1*ATR

        "cooldown_days_after_sl": 0,

        "initial_capital": 100000.0,
        "risk_per_trade_pct": 1.0,   # procent (1.0 = 1%)
    }
    # skriv över med inkommande
    p.update({k: v for k, v in params.items() if v is not None})

    df = _normalize_ohlcv(df)
    if df.empty or len(df) < 30:
        # tomt resultat
        equity = pd.DataFrame({"Equity": [p["initial_capital"]]}, index=[pd.Timestamp.utcnow()])
        trades_df = pd.DataFrame(columns=["EntryDate","EntryPrice","ExitDate","ExitPrice","Qty","PnL","ReturnPct","Reason"])
        stats = {"Strategy": p["strategy"], "TotalReturn": 0.0, "MaxDD": 0.0, "SharpeD": 0.0, "Trades": 0, "WinRate": 0.0, "PF": 0.0, "AvgTrade": 0.0}
        return equity, trades_df, stats

    close = df["Close"]

    # indikatorer som ev. behövs
    atr = _atr(df, int(p["atr_period"])) if p["atr_period"] else pd.Series(index=df.index, dtype=float)

    trend_ok = pd.Series(True, index=df.index)
    if p["use_trend_filter"]:
        tma = _ma(close, int(p["trend_ma_window"]), p.get("trend_ma_type","SMA"))
        trend_ok = close > tma

    # strategisignaler (entry/exit bool)
    entry_sig = pd.Series(False, index=df.index)
    exit_sig  = pd.Series(False, index=df.index)

    strat = str(p["strategy"]).lower()

    if strat == "breakout":
        lb = int(p["breakout_lookback"])
        ex = int(p["exit_lookback"])
        # "klassisk" breakout: Close > high(lb) (exkl. dagens bar)
        prev_high = _rolling_high(df["High"].shift(1), lb)
        entry_sig = (close > prev_high) & prev_high.notna()
        # exit på low lookback
        exit_low = _rolling_low(df["Low"].shift(1), ex)
        exit_sig = (close < exit_low) & exit_low.notna()

    elif strat in ("ma_cross", "ema_cross"):
        fast = int(p["fast"])
        slow = int(p["slow"])
        ma_type = p.get("ma_type", "SMA")
        if strat == "ema_cross":
            ma_type = "EMA"  # tvinga EMA
        f = _ma(close, fast, ma_type)
        s = _ma(close, slow, ma_type)
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
        entry_sig = cross_up
        exit_sig  = cross_dn

    elif strat == "rsi":
        n = int(p["rsi_window"])
        rsi = _rsi(close, n)
        re = float(p.get("rsi_entry_min") or 0.0)
        rx = float(p.get("rsi_exit_max") or 0.0)
        if re > 0:
            entry_sig = (rsi >= re) & (rsi.shift(1) < re)
        if rx > 0:
            exit_sig  = (rsi <= rx) & (rsi.shift(1) > rx)

    # trendfilter appliceras på entry
    entry_sig = entry_sig & trend_ok

    # ========= simulering =========
    initial_cap = float(p["initial_capital"])
    risk_pct = float(p["risk_per_trade_pct"]) / 100.0  # till andel

    cash = initial_cap
    qty = 0
    entry_price = None
    entry_date: Optional[pd.Timestamp] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    trail_pct = p.get("trailing_pct")
    peak_floor = float(p.get("peak_floor_pct") or 0.0)

    highest_since_entry = None
    cooldown_counter = 0

    eq_series: List[float] = []
    idx_series: List[pd.Timestamp] = []
    trades: List[Trade] = []

    for i, (dt, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        this_atr = float(atr.get(dt, np.nan)) if not atr.empty else np.nan

        # mark-to-market equity
        equity_now = cash + (qty * price)
        idx_series.append(dt)
        eq_series.append(equity_now)

        # cooldown räknas ner
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # uppdatera trailing/peak vid öppen position
        if qty > 0:
            if highest_since_entry is None:
                highest_since_entry = price
            else:
                highest_since_entry = max(highest_since_entry, price)

            # trailing stop som % under högsta
            if trail_pct:
                tpct = float(trail_pct)
                tpct = max(0.0, min(tpct, 0.95))
                trail_stop = highest_since_entry * (1.0 - tpct)
                stop_price = max(stop_price or -np.inf, trail_stop)

            # breakeven med ATR
            be_mult = float(p.get("breakeven_atr_mult") or 0.0)
            if be_mult > 0 and not np.isnan(this_atr) and entry_price is not None:
                # flytta SL upp till entry när priset gått +N*ATR
                if price >= entry_price + be_mult * this_atr:
                    stop_price = max(stop_price or -np.inf, entry_price)

        # exitlogik
        exit_reason = None
        if qty > 0:
            # regelbaserad exit-signal
            if bool(exit_sig.iloc[i]):
                exit_reason = "RuleExit"

            # stop-loss
            sl = p.get("stop_loss_pct")
            if sl is not None and entry_price is not None:
                bound = entry_price * (1.0 - float(sl))
                if stop_price is None:
                    stop_price = bound
                else:
                    stop_price = max(stop_price, bound)
            # ta-profit
            tp = p.get("take_profit_pct")
            if tp is not None and entry_price is not None and price >= entry_price * (1.0 + float(tp)):
                exit_reason = exit_reason or "TakeProfit"

            # peak-floor: lämna om fall från topp > peak_floor_pct
            if peak_floor > 0 and highest_since_entry and price <= highest_since_entry * (1.0 - peak_floor):
                exit_reason = exit_reason or "PeakFloor"

            # stop-träff?
            if stop_price is not None and price <= stop_price:
                exit_reason = "StopLoss"

        # utför exit
        if qty > 0 and exit_reason:
            exit_px = price
            pnl = (exit_px - entry_price) * qty  # type: ignore
            ret_pct = (exit_px / entry_price - 1.0)  # type: ignore
            cash += qty * exit_px
            trades.append(Trade(entry_date, entry_price, dt, exit_px, qty, pnl, ret_pct, exit_reason))  # type: ignore
            qty = 0
            entry_price = None
            entry_date = None
            stop_price = None
            tp_price = None
            highest_since_entry = None
            if exit_reason == "StopLoss":
                cooldown_counter = int(p.get("cooldown_days_after_sl") or 0)
            # uppdatera equity efter exit
            equity_now = cash
            eq_series[-1] = equity_now  # ersätt senaste mark-to-market med verklig

        # entrylogik
        if qty == 0 and cooldown_counter == 0 and bool(entry_sig.iloc[i]):
            # positionsstorlek via risk-procent
            risk_cash = equity_now * risk_pct
            if p.get("stop_loss_pct"):
                risk_per_share = price * float(p["stop_loss_pct"])
                shares = int(np.floor(risk_cash / max(risk_per_share, 1e-9)))
            else:
                # om ingen SL: använt risk-procent som notional
                shares = int(np.floor(risk_cash / price))
            if shares <= 0:
                shares = int(max(1, np.floor(equity_now * 0.01 / max(price, 1e-9))))  # fallback

            cost = shares * price
            if shares > 0 and cost <= cash:
                qty = shares
                cash -= cost
                entry_price = price
                entry_date = dt
                highest_since_entry = price
                stop_price = None
                tp_price = None

    # stäng ev. öppen position vid slut
    if qty > 0 and entry_price is not None and entry_date is not None:
        last_dt = df.index[-1]
        last_px = float(df["Close"].iloc[-1])
        pnl = (last_px - entry_price) * qty
        ret_pct = (last_px / entry_price - 1.0)
        cash += qty * last_px
        trades.append(Trade(entry_date, entry_price, last_dt, last_px, qty, pnl, ret_pct, "EOD"))
        qty = 0

    equity = pd.Series(eq_series, index=pd.Index(idx_series, name="Date"), name="Equity").to_frame()

    # trades dataframe
    if trades:
        tdf = pd.DataFrame([t.__dict__ for t in trades])
    else:
        tdf = pd.DataFrame(columns=["EntryDate","EntryPrice","ExitDate","ExitPrice","Qty","PnL","ReturnPct","Reason"])

    # stats
    total_return = float(equity["Equity"].iloc[-1] / initial_cap - 1.0)
    maxdd = _max_drawdown(equity["Equity"])
    sharpe = _daily_sharpe(equity["Equity"])

    wins = tdf.loc[tdf["PnL"] > 0, "PnL"].sum() if not tdf.empty else 0.0
    losses = -tdf.loc[tdf["PnL"] < 0, "PnL"].sum() if not tdf.empty else 0.0
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    winrate = float((tdf["PnL"] > 0).mean()) if not tdf.empty else 0.0
    avg_trade = float(tdf["ReturnPct"].mean()) if not tdf.empty else 0.0

    stats = {
        "Strategy": p["strategy"],
        "TotalReturn": total_return,
        "MaxDD": maxdd,
        "SharpeD": sharpe,
        "Trades": int(len(tdf)),
        "WinRate": round(winrate, 4),
        "PF": 0.0 if np.isinf(pf) else round(pf, 5),
        "AvgTrade": round(avg_trade, 5),
    }

    return equity, tdf, stats

# ============== (valfritt) CLI för snabba tester =================

def load_ohlcv(source: str, target: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    En enkel loader så att andra moduler kan köra denna fil direkt:
      source: 'borsdata' eller 'csv'
    """
    if source == "csv":
        fp = Path(target)
        df = pd.read_csv(fp, sep=None, engine="python")
        # normalisera kolumnnamn
        norm = {}
        for c in df.columns:
            lc = c.lower()
            norm[c] = {"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","adj close":"Adj Close","volume":"Volume"}.get(lc, c)
        df = df.rename(columns=norm)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        need = [c for c in ["Open","High","Low","Close"] if c in df.columns]
        return df[need]
    elif source == "borsdata":
        from app.data_providers import get_ohlcv
        return get_ohlcv("borsdata", target, start, end)
    else:
        raise ValueError("Unknown source")

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv","borsdata"], required=True)
    ap.add_argument("--target", required=True, help="CSV path eller Börsdata-ticker")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    # strategi-val
    ap.add_argument("--strategy", default="breakout", choices=["breakout","ma_cross","ema_cross","rsi"])
    ap.add_argument("--bo", type=int, default=20, help="breakout_lookback")
    ap.add_argument("--ex", type=int, default=10, help="exit_lookback")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA","EMA"])
    ap.add_argument("--rsi-window", type=int, default=14)
    ap.add_argument("--rsi-entry-min", type=float, default=0.0)
    ap.add_argument("--rsi-exit-max", type=float, default=0.0)

    # trendfilter
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--tma", type=int, default=100)
    ap.add_argument("--tma-type", default="SMA", choices=["SMA","EMA"])

    # risk
    ap.add_argument("--sl", type=float, default=None, help="stop_loss_pct ex 0.08")
    ap.add_argument("--tp", type=float, default=None, help="take_profit_pct ex 0.20")
    ap.add_argument("--tr", type=float, default=None, help="trailing_pct ex 0.10")
    ap.add_argument("--pf", type=float, default=0.0, help="peak_floor_pct ex 0.10")
    ap.add_argument("--atr", type=int, default=14, help="atr_period")
    ap.add_argument("--be", type=float, default=0.0, help="breakeven_atr_mult")
    ap.add_argument("--cool", type=int, default=0)

    # kapital
    ap.add_argument("--cap", type=float, default=100000.0)
    ap.add_argument("--risk", type=float, default=1.0, help="risk_per_trade_pct")

    args = ap.parse_args()

    df = load_ohlcv(args.source, args.target, args.start, args.end)

    params = dict(
        strategy=args.strategy,
        breakout_lookback=args.bo,
        exit_lookback=args.ex,
        fast=args.fast,
        slow=args.slow,
        ma_type=args.ma_type,
        rsi_window=args.rsi_window,
        rsi_entry_min=args.rsi_entry_min,
        rsi_exit_max=args.rsi_exit_max,
        use_trend_filter=bool(args.tf),
        trend_ma_window=args.tma,
        trend_ma_type=args.tma_type,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        trailing_pct=args.tr,
        peak_floor_pct=args.pf,
        atr_period=args.atr,
        breakeven_atr_mult=args.be,
        cooldown_days_after_sl=args.cool,
        initial_capital=args.cap,
        risk_per_trade_pct=args.risk,
    )

    eq, trades, stats = run_backtest(df, **params)
    print("Stats:", stats)
    print("Trades:", len(trades))






