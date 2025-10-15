# run_ui_050.py  (unik fil så vi vet att rätt UI körs)
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import date
from typing import Optional
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

APP_VERSION = "0.5.0-standalone"

# Sätt projektrot och säkerställ importväg
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ladda .env från projektroten
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

# Importera våra moduler
from app.data_providers import get_ohlcv
from app.bd_modern_client import BDModernAdapter

def _fmt_dt(x: Optional[pd.Timestamp]) -> str:
    if x is None or pd.isna(x): return "—"
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def _mask_key(k: Optional[str]) -> str:
    if not k: return "—"
    return f"{'*'*(max(len(k)-4,0))}{k[-4:]} (len={len(k)})"

@st.cache_data(show_spinner=False, ttl=600)
def load_ohlcv_cached(ticker: str, start: Optional[str]) -> pd.DataFrame:
    return get_ohlcv("borsdata", ticker, start)

st.set_page_config(page_title=f"Tradbot UI {APP_VERSION}", layout="wide")
st.title(f"🚀 Tradbot – OHLCV-Viewer (v{APP_VERSION})")
st.caption(f"ENTRY FILE: {__file__}")

with st.sidebar:
    st.header("⚙️ Inställningar")
    key = os.getenv("BORSDATA_API_KEY") or os.getenv("BD_API_KEY") or os.getenv("BD_TOKEN") or os.getenv("BORSDATA_KEY")
    st.write(f"API-nyckel: **{'✅' if key else '❌'}**  {_mask_key(key)}")

    try:
        import importlib.metadata as im
        st.caption(
            "Runtime → "
            f"python={sys.version.split()[0]}, "
            f"streamlit={im.version('streamlit')}, "
            f"plotly={im.version('plotly')}"
        )
    except Exception:
        pass

    ticker = st.text_input("Ticker", value="HM B")
    start_dt = st.date_input("Startdatum", value=date(2020,1,1))
    show_table_rows = st.slider("Rader i tabellen", 50, 5000, 500, 50)
    show_last_n = st.slider("Visa sista N rader", 50, 2000, 300, 50)

    run_btn = st.button("Hämta data", type="primary", width='stretch')

with st.expander("🧪 Diagnostik", expanded=False):
    st.write({"cwd": os.getcwd(), "sys.executable": sys.executable, "root": str(ROOT)})
    try:
        meta = BDModernAdapter().choose(ticker)
        st.write({"InsId": meta and meta.get("InsId"), "Ticker": meta and meta.get("Ticker"), "Name": meta and meta.get("Name")})
    except Exception as e:
        st.error(f"Adapter fel: {e}")

if run_btn:
    if not key:
        st.error("Ingen Börsdata-nyckel. Lägg BORSDATA_API_KEY i .env i projektroten.")
        st.stop()

    start_str = start_dt.strftime("%Y-%m-%d")
    with st.spinner("Hämtar OHLCV…"):
        df = load_ohlcv_cached(ticker.strip(), start_str)

    if df is None or df.empty:
        st.warning("Inga rader. Testa annat startdatum/ticker.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Antal rader", f"{len(df):,}".replace(","," "))
    with c2: st.metric("Första datum", _fmt_dt(df.index.min() if isinstance(df.index,pd.DatetimeIndex) else None))
    with c3: st.metric("Sista datum", _fmt_dt(df.index.max() if isinstance(df.index,pd.DatetimeIndex) else None))
    with c4:
        last_close = df["Close"].dropna().iloc[-1]
        st.metric("Senaste Close", f"{last_close:,.2f}".replace(","," "))

    st.subheader("Graf")
    plot_df = df.tail(show_last_n).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75,0.25])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                                 low=plot_df["Low"], close=plot_df["Close"], showlegend=False), row=1,col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], showlegend=False), row=2,col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Data")
    st.dataframe(df.tail(show_table_rows), width='stretch')
