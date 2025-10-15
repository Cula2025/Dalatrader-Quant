from __future__ import annotations
import json, traceback, os, glob, datetime as dt
from typing import Dict, Any, Optional
import pandas as pd, streamlit as st

from app.btwrap import run_backtest as RUN_BT
from app.data_providers import get_ohlcv as GET_OHLCV
try:
    from app.equity_extract import extract_equity as EXTRACT_EQ  # noqa: F401
except Exception:
    EXTRACT_EQ = None
try:
    from app.trade_extract import extract_trades as EXTRACT_TR  # noqa: F401
except Exception:
    EXTRACT_TR = None

st.set_page_config(page_title="Backtester V2", page_icon="🧪", layout="wide")

PRIMARY = "#1f6feb"; ACCENT = "#d29922"
st.markdown(f"""
<style>
:root {{ --primary:{PRIMARY}; --accent:{ACCENT}; }}
.block-container {{ padding-top: .75rem; }}
.stButton>button {{ background: var(--primary); color: white; border: 0; }}
</style>
""", unsafe_allow_html=True)

_PROFILES_DIR = os.path.join(os.getcwd(), "profiles")

def _fail(msg: str, exc: Optional[BaseException] = None) -> None:
    st.error(msg)
    if exc:
        with st.expander("Teknisk detalj"):
            st.code(traceback.format_exc())
    st.stop()

def _list_profiles() -> Dict[str, str]:
    if not os.path.isdir(_PROFILES_DIR): return {}
    out: Dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(_PROFILES_DIR, "*.json"))):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out

def _load_profile_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list) and data["profiles"]:
        prof = data["profiles"][0]
        return prof.get("params", {}) if isinstance(prof, dict) else {}
    if isinstance(data, dict): return data.get("params", data)
    return {}

def _safe_json_editor(title: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    pretty = json.dumps(obj, ensure_ascii=False, indent=2)
    txt = st.text_area(title, value=pretty, height=220, help="Redigera parametrar som JSON.")
    try:
        parsed = json.loads(txt) if txt.strip() else {}
        if not isinstance(parsed, dict):
            st.warning("JSON måste vara ett objekt (dict). Använder originalparametrar."); return obj
        return parsed
    except Exception as e:
        st.warning(f"Ogiltig JSON – använder original. ({e})"); return obj

def _compute_bh(ticker: str, equity_index: pd.DatetimeIndex) -> pd.Series:
    if equity_index is None or len(equity_index) == 0:
        return pd.Series(dtype="float64", name="BH")
    start = equity_index[0]
    px = GET_OHLCV(ticker, start=str(start.date()), end=None)
    if not isinstance(px, pd.DataFrame) or "Close" not in px.columns:
        return pd.Series(dtype="float64", name="BH")
    close = pd.to_numeric(px["Close"], errors="coerce").dropna()
    close = close[close.index >= start]
    if close.empty: return pd.Series(dtype="float64", name="BH")
    bh = close / float(close.iloc[0]); bh.name = "BH"; return bh

def _ratio(x: pd.Series) -> float:
    try:
        if x is None or len(x) < 2: return float("nan")
        return float(x.iloc[-1] / x.iloc[0])
    except Exception: return float("nan")

st.title("🧪 Backtester V2")
colL, colR = st.columns([1, 2], gap="large")

with colL:
    st.subheader("Inmatning")
    ticker = st.text_input("Ticker", value="AAK", help="Ange aktiesymbol (ex. AAK).")
    prof_map = _list_profiles()
    prof_name = st.selectbox("Profil (JSON i profiles/)", options=["(ingen)"] + list(prof_map.keys()), index=0)
    base_params: Dict[str, Any] = {}
    if prof_name != "(ingen)":
        try: base_params = _load_profile_params(prof_map[prof_name])
        except Exception as e: st.warning(f"Kunde inte läsa profilen: {e}"); base_params = {}
    params = _safe_json_editor("Parametrar (JSON)", base_params)
    start_date = st.date_input("Startdatum för backtest", value=dt.date(2020,10,5))
    do_run = st.button("Kör backtest", type="primary")

with colR:
    st.subheader("Resultat")
    if do_run:
        if not ticker.strip(): _fail("Ange en ticker.")
        try:
            res: Dict[str, Any] = RUN_BT(p={"ticker": ticker.strip(), "params": params})
        except Exception as e:
            _fail("Backtest misslyckades (RUN_BT).", e)
        eq_raw = res.get("equity")
        if eq_raw is None: _fail("Resultatet saknar 'equity'.")
        try:
            eq = pd.to_numeric(pd.Series(eq_raw), errors="coerce").dropna()
            eq.index = pd.to_datetime(eq.index); eq = eq.sort_index(); eq.name = "Strategy"
        except Exception as e:
            _fail("Kunde inte tolka 'equity' till tidsserie.", e)
        bh = _compute_bh(ticker, eq.index)
        df_plot = pd.DataFrame(index=eq.index); df_plot["Strategy"] = eq
        if not bh.empty: df_plot["BH"] = bh.reindex(df_plot.index).ffill()
        btx = _ratio(eq); bhx = _ratio(bh) if "BH" in df_plot else float("nan")
        m1,m2,m3 = st.columns(3)
        m1.metric("BT× (Strategy / start)", f"{btx:.3f}" if btx==btx else "–")
        m2.metric("BH× (Indexerat / start)", f"{bhx:.3f}" if bhx==bhx else "–")
        m3.metric("Antal datapunkter", f"{len(df_plot):,}")
        st.line_chart(df_plot, height=360)
        trades = res.get("trades")
        if isinstance(trades, (list, tuple)) and trades:
            st.subheader("Trades")
            try: st.dataframe(pd.DataFrame(trades), use_container_width=True, height=240)
            except Exception: st.write(trades)
        with st.expander("Rådata / felsök"):
            st.write("Res-nycklar:", list(res.keys()))
            st.dataframe(df_plot.tail(10), use_container_width=True)
    else:
        st.info("Välj ticker, profil/parametrar och klicka **Kör backtest**.")
