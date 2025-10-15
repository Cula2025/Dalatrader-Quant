# app/ui.py
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import date
from typing import Optional, List, Dict, Tuple

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

APP_VERSION = "1.0.0-trades"  # <-- VISAS I UI

# --- Rot & .env ---
ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Våra moduler
from app.data_providers import get_ohlcv
from app.bd_modern_client import BDModernAdapter

# ---------- Hjälp ----------
def _fmt_dt(x: Optional[pd.Timestamp]) -> str:
    if x is None or pd.isna(x): return "—"
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def normalize_ticker_for_bd(t: str) -> str:
    s = (t or "").strip().upper()
    if s.endswith(".ST"): s = s[:-3]
    return s.replace("-", " ")

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    gain = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = dn.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n, min_periods=n).mean()
def ema(s: pd.Series, span: int) -> pd.Series: return s.ewm(span=span, adjust=False, min_periods=span).mean()

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    sig = m.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return m - sig

def compute_indicators(df: pd.DataFrame, rsi_len: int, rsi_ma_n: int,
                       macd_fast: int, macd_slow: int, macd_signal: int) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = rsi(out["Close"], rsi_len)
    out["RSI_MA"] = out["RSI"].rolling(rsi_ma_n, min_periods=rsi_ma_n).mean()
    out["SMA50"] = sma(out["Close"], 50)
    out["SMA200"] = sma(out["Close"], 200)
    out["MACD_hist"] = macd_hist(out["Close"], macd_fast, macd_slow, macd_signal)
    return out

def evaluate_signals(df: pd.DataFrame, rsi_threshold: int, require_rsi_above_ma: bool,
                     sma50_over_sma200: bool, sma200_slope_days: int,
                     macd_pos_today: bool, macd_pos_yday: bool,
                     sell_below_sma50: bool, sell_macd_neg: bool, sell_rsi_below: Optional[int]) -> pd.DataFrame:
    out = df.copy()
    buys = [out["RSI"] >= rsi_threshold]
    if require_rsi_above_ma: buys.append(out["RSI"] > out["RSI_MA"])
    if sma50_over_sma200:    buys.append(out["SMA50"] > out["SMA200"])
    if sma200_slope_days>0:  buys.append(out["SMA200"] > out["SMA200"].shift(sma200_slope_days))
    if macd_pos_today:
        buys.append(out["MACD_hist"] > 0)
        if macd_pos_yday: buys.append(out["MACD_hist"].shift(1) > 0)
    out["BUY"] = pd.concat(buys, axis=1).all(axis=1)

    sells: List[pd.Series] = []
    if sell_below_sma50: sells.append(out["Close"] < out["SMA50"])
    if sell_macd_neg:    sells.append(out["MACD_hist"] <= 0)
    if sell_rsi_below is not None: sells.append(out["RSI"] < sell_rsi_below)
    out["SELL"] = pd.concat(sells, axis=1).any(axis=1) if sells else pd.Series(False, index=out.index)
    return out

# ---------- Simulering ----------
def simulate_trades(df: pd.DataFrame, initial_capital: float=100_000.0, fee_bps: float=5.0,
                    stop_loss_pct: float=0.08, max_holding_days: int=60) -> Tuple[pd.DataFrame, pd.Series, dict]:
    df = df.copy(); idx = df.index
    fee_pct = fee_bps / 10_000.0
    cash = float(initial_capital); shares = 0; in_pos = False
    entry_price = 0.0; stop_price = 0.0; entry_i: Optional[int]=None; exit_next_open=False
    trades: List[dict] = []; equity_vals: List[float] = []

    def mtm(i:int)->float: return cash + shares*float(df["Close"].iloc[i]) if in_pos and shares>0 else cash

    for i in range(len(df)):
        equity_vals.append(mtm(i))

        # planerad exit i morgon open
        if in_pos and exit_next_open:
            open_px=float(df["Open"].iloc[i]); proceeds=shares*open_px; fee=proceeds*fee_pct; cash+=proceeds-fee
            trades.append({"entry_date":idx[entry_i],"entry_price":entry_price,"exit_date":idx[i],"exit_price":open_px,
                           "bars_held":i-(entry_i or i),"reason":"signal_exit_next_open",
                           "pnl_pct":(open_px/entry_price-1.0) if entry_price else 0.0})
            shares=0; in_pos=False; exit_next_open=False; entry_price=0.0; stop_price=0.0; entry_i=None
            continue

        if in_pos:
            day_low=float(df["Low"].iloc[i]); day_open=float(df["Open"].iloc[i])
            # intraday stop
            if day_low <= stop_price:
                exit_px=day_open if day_open<=stop_price else stop_price
                proceeds=shares*exit_px; fee=proceeds*fee_pct; cash+=proceeds-fee
                trades.append({"entry_date":idx[entry_i],"entry_price":entry_price,"exit_date":idx[i],
                               "exit_price":exit_px,"bars_held":i-(entry_i or i),"reason":"stop_loss",
                               "pnl_pct":(exit_px/entry_price-1.0) if entry_price else 0.0})
                shares=0; in_pos=False; exit_next_open=False; entry_price=0.0; stop_price=0.0; entry_i=None
                continue
            # max holding
            if max_holding_days>0 and entry_i is not None and (i-entry_i)>=max_holding_days and (i+1)<len(df):
                exit_next_open=True
            # säljsignal
            if bool(df["SELL"].iloc[i]) and (i+1)<len(df): exit_next_open=True

        # entry i morgon open
        if (not in_pos) and bool(df["BUY"].iloc[i]) and (i+1)<len(df):
            next_open=float(df["Open"].iloc[i+1]); qty=int(cash//next_open)
            if qty>0:
                cost=qty*next_open; fee=cost*fee_pct; cash-=(cost+fee); shares=qty; in_pos=True
                entry_price=next_open; stop_price=entry_price*(1.0-float(stop_loss_pct)); entry_i=i+1; exit_next_open=False

    # stäng på sista close
    if in_pos and shares>0:
        last_close=float(df["Close"].iloc[-1]); proceeds=shares*last_close; fee=proceeds*fee_pct; cash+=proceeds-fee
        trades.append({"entry_date":idx[entry_i],"entry_price":entry_price,"exit_date":idx[-1],"exit_price":last_close,
                       "bars_held":len(df)-1-(entry_i or len(df)-1),"reason":"end_close",
                       "pnl_pct":(last_close/entry_price-1.0) if entry_price else 0.0})
    equity=pd.Series(equity_vals,index=df.index,name="equity")

    # nyckeltal
    tot_ret = equity.iloc[-1]/equity.iloc[0]-1.0 if len(equity)>=2 else 0.0
    try:
        days=(df.index[-1]-df.index[0]).days; cagr=(equity.iloc[-1]/equity.iloc[0])**(365.25/max(days,1))-1.0
    except Exception: cagr=float("nan")
    max_dd=(equity/equity.cummax()-1.0).min()
    trades_df=pd.DataFrame(trades)
    if not trades_df.empty:
        wins=trades_df["pnl_pct"]>0; win_rate=wins.mean()
        avg_win=trades_df.loc[wins,"pnl_pct"].mean() if wins.any() else 0.0
        avg_loss=trades_df.loc[~wins,"pnl_pct"].mean() if (~wins).any() else 0.0
        gp=trades_df.loc[wins,"pnl_pct"].clip(lower=0).sum(); gl=-trades_df.loc[~wins,"pnl_pct"].clip(upper=0).sum()
        pf=(gp/gl) if gl>0 else float("inf"); held=trades_df["bars_held"].mean()
    else:
        win_rate=avg_win=avg_loss=pf=held=0.0
    stats={"initial":initial_capital,"final":float(equity.iloc[-1]) if len(equity) else initial_capital,
           "total_return_pct":tot_ret,"cagr_pct":cagr,"max_drawdown_pct":max_dd,
           "num_trades":int(len(trades_df)),"win_rate_pct":win_rate,"avg_win_pct":avg_win,
           "avg_loss_pct":avg_loss,"profit_factor":pf,"avg_bars_held":float(held)}
    return trades_df,equity,stats

# ---------- Plot ----------
def plot_price(df: pd.DataFrame, last_n:int=300, entries:Optional[pd.DataFrame]=None,
               exits:Optional[pd.DataFrame]=None, title:str="Pris"):
    p=df.tail(last_n).copy()
    fig=make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6,0.2,0.2])
    fig.add_trace(go.Candlestick(x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], showlegend=False), row=1,col=1)
    if "SMA50" in p and "SMA200" in p:
        fig.add_trace(go.Scatter(x=p.index,y=p["SMA50"],mode="lines",name="SMA50"),row=1,col=1)
        fig.add_trace(go.Scatter(x=p.index,y=p["SMA200"],mode="lines",name="SMA200"),row=1,col=1)
    fig.add_trace(go.Bar(x=p.index,y=p["Volume"],showlegend=False,name="Volym"),row=2,col=1)
    if "RSI" in p: fig.add_trace(go.Scatter(x=p.index,y=p["RSI"],mode="lines",name="RSI"),row=3,col=1)
    if entries is not None and not entries.empty:
        e2=entries[entries["entry_date"].isin(p.index)]
        fig.add_trace(go.Scatter(x=e2["entry_date"], y=[df.loc[d,"Low"] for d in e2["entry_date"]],
                                 mode="markers", name="KÖP", marker_symbol="triangle-up", marker_size=12), row=1,col=1)
    if exits is not None and not exits.empty:
        x2=exits[exits["exit_date"].isin(p.index)]
        fig.add_trace(go.Scatter(x=x2["exit_date"], y=[df.loc[d,"High"] for d in x2["exit_date"]],
                                 mode="markers", name="SÄLJ", marker_symbol="triangle-down", marker_size=12), row=1,col=1)
    fig.update_layout(title=title, height=820, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width='stretch')

def plot_equity(eq: pd.Series):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Equity"))
    fig.update_layout(title="Kapital-/Equity-kurva", height=300, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, width='stretch')

# ---------- UI ----------
st.set_page_config(page_title=f"Tradbot {APP_VERSION}", layout="wide")
st.markdown(f"<h2>Tradbot UI — <code>{APP_VERSION}</code></h2><small>ENTRY FILE: {__file__}</small>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Vy")
    view = st.radio("Välj vy", options=["OHLCV", "Backtest"], index=1)

with st.sidebar:
    st.divider(); st.header("Gemensamt")
    ticker_in = st.text_input("Ticker", value="HM B")
    start_dt = st.date_input("Startdatum", value=date(2020,1,1))
    if st.button("🧹 Rensa cache", width='stretch'):
        st.cache_data.clear(); st.success("Cache rensad.")

# ---- OHLCV ----
if view=="OHLCV":
    @st.cache_data(show_spinner=False, ttl=600)
    def load_ohlcv_ohlcv(t: str, s: Optional[str])->pd.DataFrame:
        return get_ohlcv("borsdata", normalize_ticker_for_bd(t), s)
    start_str = start_dt.strftime("%Y-%m-%d")
    with st.spinner("Hämtar OHLCV …"):
        df = load_ohlcv_ohlcv(ticker_in, start_str)
    if df.empty: st.warning("Inga rader."); 
    else:
        c1,c2,c3,c4=st.columns(4)
        with c1: st.metric("Antal rader", f"{len(df):,}".replace(","," "))
        with c2: st.metric("Första datum", _fmt_dt(df.index.min()))
        with c3: st.metric("Sista datum", _fmt_dt(df.index.max()))
        with c4: st.metric("Senaste Close", f"{df['Close'].dropna().iloc[-1]:,.2f}".replace(","," "))
        plot_price(df, last_n=300, title=f"{ticker_in} – pris/volym")
        st.dataframe(df.tail(500), width='stretch')

# ---- Backtest ----
else:
    with st.sidebar:
        st.header("KÖP-filter")
        rsi_thr = st.slider("RSI-tröskel (>=)", 40, 70, 50)
        rsi_len = st.slider("RSI-period", 7, 21, 14)
        rsi_ma_n = st.slider("RSI MA(n)", 1, 10, 5)
        rsi_above_ma = st.checkbox("Kräv RSI > RSI_MA", True)
        sma50_gt_200 = st.checkbox("Kräv SMA50 > SMA200", False)
        sma200_up = st.checkbox("SMA200 lutar upp (5 dagar)", False)
        macd_pos = st.checkbox("MACD-hist > 0 idag", True)
        macd_pos_yday = st.checkbox("… och igår", False)

        st.header("SÄLJ-filter")
        sell_below_sma50 = st.checkbox("Close < SMA50", True)
        sell_macd_neg = st.checkbox("MACD-hist ≤ 0", False)
        use_rsi_sell = st.checkbox("RSI < X", False)
        sell_rsi_level = st.slider("RSI X (sälj)", 30, 60, 45) if use_rsi_sell else None

        st.header("Sim-handel")
        init_cap = st.number_input("Startkapital (SEK)", 10_000.0, 10_000_000.0, 100_000.0, 10_000.0)
        fee_bps = st.slider("Courtage/Slippage (bps)", 0, 100, 5)
        stop_loss = st.slider("Stop-loss (%)", 1, 20, 8) / 100.0
        max_hold = st.slider("Max innehavsdagar (0=ingen)", 0, 250, 60)

        last_n = st.slider("Visa sista N dagar i graf", 100, 1000, 300, 50)
        table_rows = st.slider("Rader i tabeller", 50, 5000, 500, 50)

        if st.button("🔧 Demo-parametrar (skapa affärer)", width='stretch'):
            st.session_state.update({
                "RSI":50,"RSI_LEN":14,"RSI_MA":5,"RSI_ABOVE":True,"SMA50_200":False,"SMA200_UP":False,
                "MACD_POS":True,"MACD_POS_YDAY":False,"SELL_SMA50":True,"SELL_MACD":False,
                "USE_RSI_SELL":False,"STOP":8,"MAX_HOLD":60
            }); st.success("Demo-parametrar satta. Kör backtest ↘"); 

        run = st.button("🚀 Kör backtest", type="primary", width='stretch')

    if run:
        start_str = start_dt.strftime("%Y-%m-%d")
        with st.spinner("Hämtar data & kör …"):
            raw = get_ohlcv("borsdata", normalize_ticker_for_bd(ticker_in), start_str)
        if raw.empty:
            st.warning("Inga rader."); 
        else:
            ind = compute_indicators(raw, rsi_len=rsi_len, rsi_ma_n=rsi_ma_n, macd_fast=12, macd_slow=26, macd_signal=9)
            sig = evaluate_signals(ind, rsi_threshold=rsi_thr, require_rsi_above_ma=rsi_above_ma,
                                   sma50_over_sma200=sma50_gt_200, sma200_slope_days=(5 if sma200_up else 0),
                                   macd_pos_today=macd_pos, macd_pos_yday=macd_pos_yday,
                                   sell_below_sma50=sell_below_sma50, sell_macd_neg=sell_macd_neg,
                                   sell_rsi_below=(sell_rsi_level if use_rsi_sell else None))

            st.info(f"BUY-signaler: **{int(sig['BUY'].sum())}** · SELL-signaler: **{int(sig['SELL'].sum())}**")

            trades, equity, stats = simulate_trades(sig, initial_capital=float(init_cap), fee_bps=float(fee_bps),
                                                    stop_loss_pct=float(stop_loss), max_holding_days=int(max_hold))

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Slutvärde", f"{stats['final']:,.0f}".replace(","," ")); st.metric("Avkastning", f"{stats['total_return_pct']*100:,.2f}%")
            with c2: st.metric("CAGR", f"{stats['cagr_pct']*100:,.2f}%"); st.metric("Max DD", f"{stats['max_drawdown_pct']*100:,.2f}%")
            with c3: st.metric("# Affärer", f"{stats['num_trades']}"); st.metric("Vinst% ", f"{stats['win_rate_pct']*100:,.1f}%")
            with c4: st.metric("Profit factor", f"{stats['profit_factor']:.2f}"); st.metric("Snitt dagar", f"{stats['avg_bars_held']:.1f}")

            st.divider(); st.subheader("Pris + signaler")
            entries = trades[["entry_date","entry_price"]] if not trades.empty else None
            exits   = trades[["exit_date","exit_price"]] if not trades.empty else None
            plot_price(sig, last_n=last_n, entries=entries, exits=exits, title=f"{ticker_in} – pris/ind./signaler")

            st.subheader("Equity-kurva"); plot_equity(equity)

            st.divider(); st.subheader("Affärer")
            if trades.empty:
                st.warning("Inga affärer – lätta på filterna eller klicka **Demo-parametrar**.")
            else:
                v=trades.copy(); v["pnl_%"]=(v["pnl_pct"]*100).round(2)
                st.dataframe(v[["entry_date","entry_price","exit_date","exit_price","bars_held","reason","pnl_%"]].tail(table_rows), width='stretch')
                st.download_button("💾 Ladda ner affärer (CSV)", data=trades.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{normalize_ticker_for_bd(ticker_in)}_trades.csv", mime="text/csv",
                                   width='stretch')
    else:
        st.info("Ställ in parametrar och klicka **Kör backtest**.")


