# app/pages/1_Backtest.py
from __future__ import annotations

import os
from datetime import date
from typing import Optional, Dict, List, Tuple

import numpy as pd
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# VÃ¥ra moduler
from app.data_providers import get_ohlcv
from app.bd_modern_client import BDModernAdapter

PAGE_VERSION = "0.7.0-backtest (sim-trades + stop-loss)"


# -----------------------------
# Utility / Indicators
# -----------------------------
def normalize_ticker_for_bd(t: str) -> str:
    s = (t or "").strip().upper()
    if s.endswith(".ST"):
        s = s[:-3]
    return s.replace("-", " ")

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd - sig

def compute_indicators(df: pd.DataFrame, rsi_len: int, rsi_ma_n: int,
                       macd_fast: int, macd_slow: int, macd_signal: int) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = rsi(out["Close"], rsi_len)
    out["RSI_MA"] = out["RSI"].rolling(rsi_ma_n, min_periods=rsi_ma_n).mean()
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA200"] = sma(out["Close"], 200)
    out["MACD_hist"] = macd_hist(out["Close"], macd_fast, macd_slow, macd_signal)
    return out

def evaluate_signals(df: pd.DataFrame,
                     rsi_threshold: int,
                     require_rsi_above_ma: bool,
                     sma50_over_sma200: bool,
                     sma200_slope_days: int,
                     macd_pos_today: bool,
                     macd_pos_yesterday: bool,
                     sell_below_sma50: bool,
                     sell_macd_neg: bool,
                     sell_rsi_below: Optional[int]) -> pd.DataFrame:
    out = df.copy()

    # --- KÃ¶pregler ---
    conds = [out["RSI"] >= rsi_threshold]
    if require_rsi_above_ma: conds.append(out["RSI"] > out["RSI_MA"])
    if sma50_over_sma200: conds.append(out["SMA50"] > out["SMA200"])
    if sma200_slope_days > 0: conds.append(out["SMA200"] > out["SMA200"].shift(sma200_slope_days))
    if macd_pos_today:
        conds.append(out["MACD_hist"] > 0)
        if macd_pos_yesterday: conds.append(out["MACD_hist"].shift(1) > 0)
    buy_signal = pd.concat(conds, axis=1).all(axis=1) if conds else pd.Series(False, index=out.index)

    # --- SÃ¤ljregler (signaler) ---
    sell_conds = []
    if sell_below_sma50: sell_conds.append(out["Close"] < out["SMA50"])
    if sell_macd_neg: sell_conds.append(out["MACD_hist"] <= 0)
    if sell_rsi_below is not None: sell_conds.append(out["RSI"] < sell_rsi_below)
    sell_signal = pd.concat(sell_conds, axis=1).any(axis=1) if sell_conds else pd.Series(False, index=out.index)

    out["BUY"] = buy_signal
    out["SELL"] = sell_signal
    return out


# -----------------------------
# Simulation (trades + equity)
# -----------------------------
def simulate_trades(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,    # 5 bps = 0.05% per transaktion
    stop_loss_pct: float = 0.08,  # 8% stop
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """
    En enkel 1-positions backtest:
      - ENTER: nÃ¤r BUY Ã¤r True pÃ¥ dag t â†’ kÃ¶p pÃ¥ DAG t+1 OPEN
      - EXIT:  nÃ¤r SELL Ã¤r True pÃ¥ dag t â†’ sÃ¤lj pÃ¥ DAG t+1 OPEN (om vi inte stoppas tidigare)
      - STOP:  intradag â€“ om LÃ¥g <= stop, exit pÃ¥:
               * dagens OPEN om OPEN <= stop (gap ner)
               * annars pÃ¥ sjÃ¤lva stop-priset
      - Courtage/slippage: fee_bps tas bÃ¥de vid kÃ¶p och sÃ¤lj
    """
    df = df.copy()
    idx = df.index

    fee_pct = fee_bps / 10_000.0
    cash = float(initial_capital)
    shares = 0
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    pending_exit_next_open = False
    entry_idx = None  # typ: int

    # loggar
    trades: List[Dict[str, object]] = []
    equity_vals: List[float] = []

    def mark_to_market(i: int) -> float:
        if in_pos and shares > 0:
            return cash + shares * float(df["Close"].iloc[i])
        return cash

    for i in range(len(df)):
        # 0) Equity (mark-to-market pÃ¥ Close)
        equity_vals.append(mark_to_market(i))

        # 1) EXIT pÃ¥ nÃ¤sta dags OPEN som redan Ã¤r planerad
        if in_pos and pending_exit_next_open:
            # Exekveras pÃ¥ dagens OPEN
            open_px = float(df["Open"].iloc[i])
            proceeds = shares * open_px
            fee = proceeds * fee_pct
            cash += proceeds - fee

            trades.append({
                "entry_date": idx[entry_idx],
                "entry_price": entry_price,
                "exit_date": idx[i],
                "exit_price": open_px,
                "bars_held": i - entry_idx,
                "reason": "exit_signal_next_open",
                "pnl_pct": (open_px / entry_price - 1.0) if entry_price else 0.0,
            })

            # NollstÃ¤ll position
            shares = 0
            in_pos = False
            pending_exit_next_open = False
            entry_price = 0.0
            stop_price = 0.0
            entry_idx = None
            # uppdatera equity efter exekvering (mark-to-market nÃ¤sta loop)
            continue

        # 2) Om i position: kolla STOP intradag
        if in_pos:
            day_low = float(df["Low"].iloc[i])
            day_open = float(df["Open"].iloc[i])
            if day_low <= stop_price:
                # Gap-down? I sÃ¥ fall fyll pÃ¥ open. Annars pÃ¥ stop.
                exit_px = day_open if day_open <= stop_price else stop_price
                proceeds = shares * exit_px
                fee = proceeds * fee_pct
                cash += proceeds - fee

                trades.append({
                    "entry_date": idx[entry_idx],
                    "entry_price": entry_price,
                    "exit_date": idx[i],
                    "exit_price": exit_px,
                    "bars_held": i - entry_idx,
                    "reason": "stop_loss",
                    "pnl_pct": (exit_px / entry_price - 1.0) if entry_price else 0.0,
                })
                shares = 0
                in_pos = False
                pending_exit_next_open = False
                entry_price = 0.0
                stop_price = 0.0
                entry_idx = None
                continue

            # 3) Om SELL-signal idag â†’ planera exit pÃ¥ nÃ¤sta dags OPEN
            if bool(df["SELL"].iloc[i]) and i + 1 < len(df):
                pending_exit_next_open = True

        # 4) Om ej i position: kolla BUY-signal idag â†’ kÃ¶p i morgon OPEN
        if (not in_pos) and bool(df["BUY"].iloc[i]) and (i + 1 < len(df)):
            next_open = float(df["Open"].iloc[i + 1])
            if next_open > 0:
                # All-in position (helt kapital, heltal aktier)
                qty = int(cash // next_open)
                if qty > 0:
                    cost = qty * next_open
                    fee = cost * fee_pct
                    cash -= (cost + fee)
                    shares = qty
                    in_pos = True
                    entry_price = next_open
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                    entry_idx = i + 1
                    pending_exit_next_open = False
                # om qty=0 -> kÃ¶p ignoreras (fÃ¶r lite kapital)

    # Slutmarkering: om vi sitter med position vid sista baren â†’ stÃ¤ng pÃ¥ sista CLOSE
    if in_pos and shares > 0:
        last_close = float(df["Close"].iloc[-1])
        proceeds = shares * last_close
        fee = proceeds * fee_pct
        cash += proceeds - fee
        trades.append({
            "entry_date": idx[entry_idx],
            "entry_price": entry_price,
            "exit_date": idx[-1],
            "exit_price": last_close,
            "bars_held": len(df) - 1 - entry_idx,
            "reason": "end_close",
            "pnl_pct": (last_close / entry_price - 1.0) if entry_price else 0.0,
        })
        shares = 0
        in_pos = False
        entry_price = 0.0
        stop_price = 0.0
        entry_idx = None

        # uppdatera equity sista dagen (om du vill, men kurvan Ã¤r redan mark-to-market)

    equity = pd.Series(equity_vals, index=df.index, name="equity")

    # --- Nyckeltal ---
    if len(equity) >= 2:
        tot_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    else:
        tot_ret = 0.0

    # CAGR (approx)
    try:
        days = (df.index[-1] - df.index[0]).days
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365.25 / max(days, 1)) - 1.0
    except Exception:
        cagr = float("nan")

    # Max drawdown
    cummax = equity.cummax()
    dd = (equity / cummax - 1.0)
    max_dd = dd.min()

    # Trades DF & statistik
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = trades_df["pnl_pct"] > 0
        win_rate = wins.mean()
        avg_win = trades_df.loc[wins, "pnl_pct"].mean() if wins.any() else 0.0
        avg_loss = trades_df.loc[~wins, "pnl_pct"].mean() if (~wins).any() else 0.0
        gross_profit = trades_df.loc[wins, "pnl_pct"].clip(lower=0).sum()
        gross_loss = -trades_df.loc[~wins, "pnl_pct"].clip(upper=0).sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        bars_held_avg = trades_df["bars_held"].mean()
    else:
        win_rate = avg_win = avg_loss = profit_factor = bars_held_avg = 0.0

    stats = {
        "initial": initial_capital,
        "final": float(equity.iloc[-1]) if len(equity) else initial_capital,
        "total_return_pct": tot_ret,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "num_trades": int(len(trades_df)),
        "win_rate_pct": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "avg_bars_held": float(bars_held_avg),
    }
    return trades_df, equity, stats


def plot_price_with_signals(df: pd.DataFrame, title: str, last_n: int = 300,
                            entries: Optional[pd.DataFrame] = None,
                            exits: Optional[pd.DataFrame] = None):
    p = df.tail(last_n).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.2, 0.2])

    # Pris
    fig.add_trace(go.Candlestick(
        x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"],
        name="Pris", showlegend=False), row=1, col=1)

    # SMA50/SMA200
    if "SMA50" in p and "SMA200" in p:
        fig.add_trace(go.Scatter(x=p.index, y=p["SMA50"], mode="lines", name="SMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=p.index, y=p["SMA200"], mode="lines", name="SMA200"), row=1, col=1)

    # Volym
    fig.add_trace(go.Bar(x=p.index, y=p["Volume"], name="Volym", showlegend=False), row=2, col=1)

    # RSI
    if "RSI" in p:
        fig.add_trace(go.Scatter(x=p.index, y=p["RSI"], mode="lines", name="RSI"), row=3, col=1)

    # MarkÃ¶rer (entries/exits)
    if entries is not None and not entries.empty:
        e2 = entries[entries["entry_date"].isin(p.index)]
        fig.add_trace(go.Scatter(
            x=e2["entry_date"], y=[df.loc[d, "Low"] for d in e2["entry_date"]],
            mode="markers", name="KÃ–P",
            marker_symbol="triangle-up", marker_size=12
        ), row=1, col=1)
    if exits is not None and not exits.empty:
        x2 = exits[exits["exit_date"].isin(p.index)]
        fig.add_trace(go.Scatter(
            x=x2["exit_date"], y=[df.loc[d, "High"] for d in x2["exit_date"]],
            mode="markers", name="SÃ„LJ",
            marker_symbol="triangle-down", marker_size=12
        ), row=1, col=1)

    fig.update_layout(
        title=title,
        height=820,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, width='stretch')


def plot_equity(equity: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title="Kapital-/Equity-kurva", height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=f"Backtest (v{PAGE_VERSION})", layout="wide")
st.title("âš™ï¸ Backtest UI â€“ parametrar, SIM-handel & stop-loss")
st.caption(f"Version: {PAGE_VERSION}")

with st.sidebar:
    st.header("Parametrar â€“ Data")
    t_in = st.text_input("Ticker", value="HM B", help="Ex: ERIC B, HM B, VOLV B eller ERIC-B.STâ€¦")
    start_str = st.text_input("Startdatum", value="2020-01-01")

    st.header("KÃ–P-filter")
    rsi_threshold = st.slider("RSI-trÃ¶skel (>=)", 40, 70, 55)
    rsi_ma_n = st.slider("RSI ska vara Ã¶ver MA(n)", 1, 10, 5)
    sma50_gt_sma200 = st.checkbox("KrÃ¤v SMA50 > SMA200", value=True)
    sma200_up = st.checkbox("KrÃ¤v att SMA200 lutar upp (5 dagar)", value=True)
    macd_pos = st.checkbox("Krav: MACD-hist > 0 idag", value=True)
    macd_pos_yday = st.checkbox("â€¦ och igÃ¥r", value=False)

    st.header("SÃ„LJ-filter (signaler)")
    sell_below_sma50 = st.checkbox("SÃ¤lj om Close < SMA50", value=False)
    sell_macd_neg = st.checkbox("SÃ¤lj om MACD-hist â‰¤ 0", value=False)
    use_rsi_sell = st.checkbox("SÃ¤lj om RSI < X", value=False)
    sell_rsi_level = st.slider("RSI X (sÃ¤lj)", 30, 60, 45) if use_rsi_sell else None

    st.header("Indikator-parametrar")
    rsi_len = st.slider("RSI-period", 7, 21, 14)
    macd_fast = st.slider("MACD fast EMA", 8, 15, 12)
    macd_slow = st.slider("MACD slow EMA", 20, 30, 26)
    macd_signal = st.slider("MACD signal EMA", 5, 12, 9)

    st.header("Sim-handel")
    initial_capital = st.number_input("Startkapital (SEK)", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    fee_bps = st.slider("Courtage/Slippage (bps per transaktion)", 0, 100, 5,
                        help="1 bps = 0,01%. 5 bps = 0,05% per kÃ¶p/sÃ¤lj.")
    stop_loss_pct = st.slider("Stop-loss (%)", 1, 20, 8) / 100.0
    last_n = st.slider("Visa sista N dagar i graf", 100, 1000, 300, 50)
    table_rows = st.slider("Rader i transaktions-/datatabeller", 50, 5000, 500, 50)

    run_btn = st.button("ðŸš€ KÃ¶r backtest", type="primary", width='stretch')

# Diagnostik
with st.expander("ðŸ”Ž Diagnostik", expanded=False):
    key = os.getenv("BORSDATA_API_KEY") or os.getenv("BD_API_KEY") or os.getenv("BD_TOKEN") or os.getenv("BORSDATA_KEY")
    st.write({"API-key present": bool(key), "key_len": len(key or "")})
    try:
        meta = BDModernAdapter().choose(normalize_ticker_for_bd(t_in))
        st.write({"InsId": meta and meta.get("InsId"), "Ticker": meta and meta.get("Ticker"), "Name": meta and meta.get("Name")})
    except Exception as e:
        st.error(f"Instrumentlookup fel: {e}")

# KÃ¶rning
if run_btn:
    # 1) Data
    ticker = normalize_ticker_for_bd(t_in)
    try:
        df_raw = get_ohlcv("borsdata", ticker, start=start_str)
    except Exception as e:
        st.error(f"Fel vid dataladdning: {e}")
        st.stop()

    if df_raw is None or df_raw.empty:
        st.warning("Inga rader frÃ¥n datakÃ¤llan. Testa annat startdatum eller ticker.")
        st.stop()

    # 2) Indikatorer + signaler
    idf = compute_indicators(df_raw, rsi_len=rsi_len, rsi_ma_n=rsi_ma_n,
                             macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal)
    edf = evaluate_signals(
        idf,
        rsi_threshold=rsi_threshold,
        require_rsi_above_ma=True,
        sma50_over_sma200=sma50_gt_sma200,
        sma200_slope_days=(5 if sma200_up else 0),
        macd_pos_today=macd_pos,
        macd_pos_yesterday=macd_pos_yday,
        sell_below_sma50=sell_below_sma50,
        sell_macd_neg=sell_macd_neg,
        sell_rsi_below=(sell_rsi_level if use_rsi_sell else None),
    )

    # 3) Simulerade affÃ¤rer
    trades_df, equity, stats = simulate_trades(
        edf,
        initial_capital=initial_capital,
        fee_bps=float(fee_bps),
        stop_loss_pct=float(stop_loss_pct),
    )

    # 4) Nyckeltal
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SlutvÃ¤rde (SEK)", f"{stats['final']:,.0f}".replace(",", " "))
        st.metric("Total avkastning", f"{stats['total_return_pct']*100:,.2f}%")
    with c2:
        st.metric("CAGR", f"{stats['cagr_pct']*100:,.2f}%")
        st.metric("Max drawdown", f"{stats['max_drawdown_pct']*100:,.2f}%")
    with c3:
        st.metric("# AffÃ¤rer", f"{stats['num_trades']}")
        st.metric("Vinst% (win rate)", f"{stats['win_rate_pct']*100:,.1f}%")
    with c4:
        st.metric("Profit factor", f"{stats['profit_factor']:.2f}")
        st.metric("Genomsnittlig innehavstid", f"{stats['avg_bars_held']:.1f} dgr")

    st.divider()

    # 5) Grafer
    st.subheader("Pris + signaler")
    entries = trades_df[["entry_date", "entry_price"]] if not trades_df.empty else None
    exits = trades_df[["exit_date", "exit_price"]] if not trades_df.empty else None
    plot_price_with_signals(edf, f"{ticker} â€“ pris/ind. med kÃ¶p/sÃ¤ljmarkÃ¶rer", last_n=last_n,
                            entries=entries, exits=exits)

    st.subheader("Equity-kurva")
    plot_equity(equity)

    st.divider()

    # 6) Tabeller & export
    st.subheader("AffÃ¤rer")
    if trades_df.empty:
        st.info("Inga affÃ¤rer.")
    else:
        tview = trades_df.copy()
        tview["pnl_%"] = (tview["pnl_pct"] * 100).round(2)
        tview = tview[["entry_date","entry_price","exit_date","exit_price","bars_held","reason","pnl_%"]]
        st.dataframe(tview.tail(table_rows), width='stretch')
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("ðŸ’¾ Ladda ner affÃ¤rer (CSV)", data=csv, file_name=f"{ticker}_trades.csv", mime="text/csv", width='stretch')

    st.subheader("Data (med indikatorer & signaler)")
    view = edf[["Open","High","Low","Close","Volume","SMA50","SMA200","RSI","RSI_MA","MACD_hist","BUY","SELL"]].tail(table_rows)
    st.dataframe(view, width='stretch')

else:
    st.info("StÃ¤ll in parametrar och klicka **KÃ¶r backtest**.")

