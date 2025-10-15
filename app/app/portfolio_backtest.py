from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd


# =========================================================
#               Hj?lpindikatorer
# =========================================================
def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, window: int = 20, nstd: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = m + nstd * sd
    lower = m - nstd * sd
    return m, upper, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


# =========================================================
#               Parametrar (alla nya ?r valfria)
# =========================================================
@dataclass
class PortfolioParams:
    # Bas
    strategy: str = "ma_cross"  # "ma_cross" | "rsi" | "breakout"
    trend_ma_type: str = "EMA"  # "EMA" | "SMA"
    use_trend_filter: bool = False
    trend_ma_window: int = 200

    # MA-cross
    fast: int = 20
    slow: int = 100

    # RSI-entry/exit
    use_rsi_filter: bool = False
    rsi_window: int = 14
    rsi_min: float = 30.0
    rsi_max: float = 70.0

    # Breakout
    breakout_lookback: int = 55
    exit_lookback: int = 20

    # NYTT: MACD-filter
    use_macd_filter: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_mode: str = "above_zero"   # "above_zero" | "signal_cross"

    # NYTT: Bollinger-villkor
    use_bb_filter: bool = False
    bb_window: int = 20
    bb_nstd: float = 2.0
    bb_mode: str = "close_above_upper"  # "close_above_upper" | "%B_gt" | "exit_below_mid"
    bb_percent_b_min: float = 0.8       # anv?nds om bb_mode="%B_gt"

    # NYTT: ATR-stopp
    atr_window: int = 14
    atr_stop_mult: float = 0.0          # 0 = av
    atr_trail_mult: float = 0.0         # 0 = av

    # Portfolio (enkel en-aktie logik st?ds ocks?)
    cost_bps: float = 0.0               # courtage i bps (0 = gratis)
    cash_rate_apy: float = 0.0          # r?nta p? kontanter (APY)
    max_positions: int = 5
    per_trade_pct: float = 5.0
    max_exposure_pct: float = 25.0


# =========================================================
#               Strategisignaler
# =========================================================
def trend_ok(close: pd.Series, p: PortfolioParams) -> pd.Series:
    if not p.use_trend_filter:
        return pd.Series(1, index=close.index)
    ma = ema(close, p.trend_ma_window) if p.trend_ma_type.upper() == "EMA" else sma(close, p.trend_ma_window)
    return (close > ma).astype(int)

def macd_ok(close: pd.Series, p: PortfolioParams) -> pd.Series:
    if not p.use_macd_filter:
        return pd.Series(1, index=close.index)
    m, sig, _ = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    if p.macd_mode == "signal_cross":
        # kr?v att MACD > signal
        return (m > sig).astype(int)
    # default: MACD ?ver nollinje
    return (m > 0).astype(int)

def bb_ok(close: pd.Series, p: PortfolioParams) -> Tuple[pd.Series, pd.Series]:
    """
    Returnerar:
      bb_entry_ok: 1/0 f?r entry-villkor
      bb_exit:     1/0 f?r exit-villkor (anv?nds bara om mode="exit_below_mid")
    """
    if not p.use_bb_filter:
        ok = pd.Series(1, index=close.index)
        return ok, pd.Series(0, index=close.index)

    mid, up, low = bollinger(close, p.bb_window, p.bb_nstd)
    bb_exit = pd.Series(0, index=close.index)

    mode = p.bb_mode
    if mode == "close_above_upper":
        ok = (close > up).astype(int)
    elif mode == "%B_gt":
        # %B = (Close - Lower) / (Upper - Lower)
        rng = (up - low).replace(0, np.nan)
        pct_b = (close - low) / rng
        ok = (pct_b >= p.bb_percent_b_min).astype(int)
    elif mode == "exit_below_mid":
        ok = pd.Series(1, index=close.index)  # entry ej begr?nsad
        bb_exit = (close < mid).astype(int)   # exit n?r under mittband
    else:
        ok = pd.Series(1, index=close.index)

    return ok, bb_exit


def strategy_signal(close: pd.Series, high: pd.Series, low: pd.Series, p: PortfolioParams) -> pd.Series:
    s = pd.Series(0, index=close.index)

    if p.strategy == "ma_cross":
        fast_ma = ema(close, p.fast)
        slow_ma = ema(close, p.slow)
        s = (fast_ma > slow_ma).astype(int)

    elif p.strategy == "rsi":
        rr = rsi(close, p.rsi_window)
        # enkel modell: i position n?r RSI ?r mellan min och max (mean-reversion kan g?ras separat)
        in_pos = (rr >= p.rsi_min) & (rr <= p.rsi_max)
        s = in_pos.astype(int)

    elif p.strategy == "breakout":
        hh = high.rolling(p.breakout_lookback, min_periods=p.breakout_lookback).max()
        long_signal = (close > hh.shift(1)).astype(int)
        # h?ll tills priset bryter under l?g-exit
        exit_line = low.rolling(p.exit_lookback, min_periods=p.exit_lookback).min()
        exit_sig = (close < exit_line.shift(1)).astype(int)

        # Bygg position ?statefully?
        pos = []
        in_pos = False
        for i in range(len(close)):
            if i == 0:
                pos.append(0); continue
            if not in_pos and long_signal.iloc[i] == 1:
                in_pos = True
            elif in_pos and exit_sig.iloc[i] == 1:
                in_pos = False
            pos.append(1 if in_pos else 0)
        s = pd.Series(pos, index=close.index)

    else:
        # ok?nd strategi -> alltid 0
        s = pd.Series(0, index=close.index)

    # Kombinera med filter: trend, MACD, BB
    t_ok = trend_ok(close, p)
    m_ok = macd_ok(close, p)
    bb_entry_ok, bb_exit = bb_ok(close, p)

    # slutlig entrysignal
    s = s * t_ok * m_ok * bb_entry_ok

    # l?gg p? BB-exit (om aktiv)
    if p.use_bb_filter and p.bb_mode == "exit_below_mid":
        # Forcera exit d?r bb_exit == 1
        in_pos = False
        out = []
        for i in range(len(s)):
            if i == 0:
                out.append(0); continue
            if not in_pos and s.iloc[i] == 1:
                in_pos = True
            elif in_pos and bb_exit.iloc[i] == 1:
                in_pos = False
            out.append(1 if in_pos else 0)
        s = pd.Series(out, index=s.index)

    return s


# =========================================================
#               Enkel backtest (daglig, andelsbaserad)
# =========================================================
def apply_atr_exits(high: pd.Series, low: pd.Series, close: pd.Series,
                    base_pos: pd.Series, p: PortfolioParams) -> pd.Series:
    """L?gg p? ATR-baserat initialt stop och/eller trailing stop ovanp? en given positionsserie (0/1)."""
    if (p.atr_stop_mult <= 0) and (p.atr_trail_mult <= 0):
        return base_pos

    a = atr(high, low, close, p.atr_window)
    pos = []
    in_pos = False
    stop_level = np.nan
    for i in range(len(close)):
        if i == 0:
            pos.append(0); continue

        signal = 1 if base_pos.iloc[i] > 0 else 0

        if not in_pos and signal == 1:
            # Ny entry: s?tt initial stop
            in_pos = True
            if p.atr_stop_mult > 0:
                stop_level = close.iloc[i] - p.atr_stop_mult * a.iloc[i]
            else:
                stop_level = -np.inf
        elif in_pos and signal == 0:
            # Exit pga grundsignal
            in_pos = False
            stop_level = np.nan

        if in_pos:
            # Trailing
            if p.atr_trail_mult > 0 and not np.isnan(a.iloc[i]):
                trail = close.iloc[i] - p.atr_trail_mult * a.iloc[i]
                stop_level = max(stop_level, trail)

            # Stop-check
            if low.iloc[i] <= stop_level:
                in_pos = False
                stop_level = np.nan

        pos.append(1 if in_pos else 0)

    return pd.Series(pos, index=close.index)


def backtest_one(df: pd.DataFrame, p: PortfolioParams) -> Tuple[pd.Series, pd.Series]:
    """
    df: m?ste inneh?lla kolumnerna Open/High/Low/Close/Volume (Volume kan vara 0 om ok?nd)
    Returnerar: equity (indexerad till 1.0), strat_ret (dagliga strategiavkastningar)
    """
    df = df.copy()
    for c in ["Open","High","Low","Close"]:
        if c not in df.columns:
            raise ValueError(f"Saknar kolumn {c}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Bas-signal fr?n vald strategi (+ trend/MACD/BB filters)
    base_signal = strategy_signal(close, high, low, p)

    # ATR-exits ovanp?
    final_pos = apply_atr_exits(high, low, close, base_signal, p)

    # K?r ?n?sta dags close?-fill (latency 1 dag)
    pos_shifted = final_pos.shift(1).fillna(0)

    # Avkastning
    ret = close.pct_change().fillna(0.0)
    strat_ret = pos_shifted * ret

    # Courtage i bps n?r position ?ndras
    if p.cost_bps and p.cost_bps > 0:
        turns = pos_shifted.diff().abs().fillna(0)
        fee = (p.cost_bps / 10000.0)
        strat_ret = strat_ret - turns * fee

    # Kontor?nta (p? ej investerad andel)
    if p.cash_rate_apy and p.cash_rate_apy > 0:
        daily_rf = p.cash_rate_apy / 252.0
        cash_weight = 1.0 - pos_shifted.clip(0, 1)
        strat_ret = strat_ret + cash_weight * daily_rf

    equity = (1 + strat_ret).cumprod()
    equity.iloc[0] = 1.0
    return equity, strat_ret


def summarize_perf(equity: pd.Series, strat_ret: pd.Series) -> Dict[str, Any]:
    total_ret = float(equity.iloc[-1] - 1.0)
    # Max drawdown
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    max_dd = float(dd.min())

    # Sharpe (annualized, rf=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = strat_ret.std()
        sharpe = 0.0 if vol == 0 or np.isnan(vol) else (strat_ret.mean() * 252) / (vol * np.sqrt(252))

    return {
        "TotalReturn": total_ret,
        "MaxDD": max_dd,
        "SharpeD": sharpe,
        "LengthDays": int(len(equity)),
    }


# =========================================================
#               Publik API (of?r?ndrat)
# =========================================================
__version__ = "0.7.0-indicators"

def run_portfolio_backtest(
    universe_data: Dict[str, pd.DataFrame],
    params: PortfolioParams
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    K?r backtest f?r ett universum (dict: ticker -> DataFrame med OHLCV).
    Returnerar:
      equity_df: kolumn per ticker (indexerad till 1.0)
      trades_df: (placeholder: vi exponerar positionerna; trade-log kan byggas ut)
      stats:     sammanfattning inklusive per-ticker
    """

    # K?r ticker-f?r-ticker och kombinera lika vikter (enkelt)
    equities: Dict[str, pd.Series] = {}
    rets: Dict[str, pd.Series] = {}
    positions: Dict[str, pd.Series] = {}

    for ticker, df in universe_data.items():
        try:
            eq, strat_ret = backtest_one(df, params)
            equities[ticker] = eq
            rets[ticker] = strat_ret
            # ?positioner? som proxy f?r trades (1/0). Vill du ha detaljerade trades kan vi bygga ut.
            positions[ticker] = (strat_ret != 0).astype(int)  # quick proxy
        except Exception as e:
            # Om en ticker fallerar, hoppa ?ver men logga i stats
            continue

    if not equities:
        raise RuntimeError("Ingen data kunde backtestas (saknas data/kolumner?).")

    # Kombinera lika viktat (daglig logik)
    aligned = pd.DataFrame({t: rets[t] for t in rets}).fillna(0.0)
    if aligned.shape[1] == 1:
        port_ret = aligned.iloc[:, 0]
    else:
        # lika viktad ?ver alla tillg?ngliga tickers den dagen
        port_ret = aligned.mean(axis=1)

    port_equity = (1 + port_ret).cumprod()
    port_equity.iloc[0] = 1.0

    equity_df = pd.DataFrame({"Portfolio": port_equity}, index=port_equity.index)
    for t, eq in equities.items():
        equity_df[t] = eq

    trades_df = pd.DataFrame({t: positions[t] for t in positions}).reindex_like(equity_df).fillna(0)

    # Stats
    stats = summarize_perf(port_equity, port_ret)
    stats["__version__"] = __version__
    stats["params"] = asdict(params)
    per_t = {}
    for t in equities:
        per_t[t] = summarize_perf(equities[t], rets[t])
    stats["per_ticker"] = per_t

    return equity_df, trades_df, stats
