import json, traceback
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest MIN", page_icon="🧪", layout="wide")
st.title("🧪 Backtest MIN")
st.caption("Lättvikts-backtest som alltid försöker: get_ohlcv → btwrap.run_backtest → fallback backtest.run_backtest(df,p).")

# ---------- Safe imports ----------
GET = None
RUN_WRAP = None
RUN_RAW  = None
try:
    from app.data_providers import get_ohlcv as GET
except Exception as e:
    st.error(f"Kunde inte importera app.data_providers.get_ohlcv: {type(e).__name__}: {e}")

try:
    from app.btwrap import run_backtest as RUN_WRAP
except Exception:
    RUN_WRAP = None

try:
    from backtest import run_backtest as RUN_RAW
except Exception:
    RUN_RAW = None

# ---------- Profiler ----------
base = Path("/srv/trader/app/profiles") if Path("/srv/trader/app/profiles").exists() else Path("profiles")
pro_files = sorted([p for p in base.glob("*.json")], key=lambda x: x.stat().st_mtime, reverse=True)

with st.sidebar:
    st.header("Ladda profil")
    sel_file = st.selectbox("Profilfil", pro_files, format_func=lambda p: p.name if isinstance(p, Path) else str(p)) if pro_files else None
    profs = []
    if sel_file and sel_file.exists():
        try:
            data = json.loads(sel_file.read_text(encoding="utf-8"))
            profs = data.get("profiles", [])
        except Exception as e:
            st.warning(f"Kunde inte läsa {sel_file.name}: {e}")
    sel_prof = st.selectbox("Välj profil i filen", profs, format_func=lambda x: x.get("name","–")) if profs else None
    use_prof = st.checkbox("Använd vald profil", value=bool(sel_prof))

# ---------- Startvärden ----------
p0 = {}
default_ticker = ""
if use_prof and sel_prof:
    default_ticker = sel_prof.get("ticker","")
    p0 = dict(sel_prof.get("params", {}))

def _get(key, default):
    return p0.get(key, default)

# ---------- Topp: ticker + datum ----------
row = st.columns(3)
with row[0]:
    ticker = st.text_input("Ticker", value=default_ticker)
with row[1]:
    from_date = st.text_input("Från (YYYY-MM-DD)", value=str(_get("from_date", "2020-10-01")))
with row[2]:
    to_date   = st.text_input("Till (YYYY-MM-DD)",  value=str(_get("to_date",   date.today().isoformat())))

st.markdown("### Parametrar")

# ---------- FULL GRID (4 kolumner) ----------
c1, c2, c3, c4 = st.columns(4)

# RSI
with c1:
    use_rsi_filter = st.checkbox("RSI-filter", value=bool(_get("use_rsi_filter", True)))
    rsi_window     = st.number_input("RSI-fönster", 5, 200, int(_get("rsi_window", 14)))
    rsi_min        = st.number_input("RSI min", 0.0, 100.0, float(_get("rsi_min", 30.0)))
    rsi_max        = st.number_input("RSI max", 0.0, 100.0, float(_get("rsi_max", 70.0)))

# Breakout/exit + MACD
with c2:
    breakout_lookback = st.number_input("Breakout lookback", 2, 1000, int(_get("breakout_lookback", 55)))
    exit_lookback     = st.number_input("Exit lookback",     2, 1000, int(_get("exit_lookback", 20)))

    use_macd_filter = st.checkbox("MACD-filter", value=bool(_get("use_macd_filter", False)))
    macd_fast       = st.number_input("MACD fast", 1, 200, int(_get("macd_fast", 12)))
    macd_slow       = st.number_input("MACD slow", 1, 400, int(_get("macd_slow", 26)))
    macd_signal     = st.number_input("MACD signal", 1, 200, int(_get("macd_signal", 9)))

# Trend + BB
with c3:
    use_trend_filter = st.checkbox("Trendfilter", value=bool(_get("use_trend_filter", False)))
    trend_ma_type    = st.selectbox("MA-typ", ["SMA","EMA"], index=(["SMA","EMA"].index(str(_get("trend_ma_type", "SMA"))) if str(_get("trend_ma_type","SMA")) in ["SMA","EMA"] else 0))
    trend_ma_window  = st.number_input("Trend MA-fönster", 2, 1000, int(_get("trend_ma_window", 200)))

    use_bb_filter = st.checkbox("Bollinger-filter", value=bool(_get("use_bb_filter", False)))
    bb_window     = st.number_input("BB window", 2, 1000, int(_get("bb_window", 20)))
    bb_nstd       = st.number_input("BB n std", 0.1, 10.0, float(_get("bb_nstd", 2.0)))
    bb_min        = st.number_input("BB min", 0.0, 5.0, float(_get("bb_min", 0.0)))

# Stop-loss / ATR
with c4:
    use_stop_loss = st.checkbox("Stop-loss", value=bool(_get("use_stop_loss", False)))
    stop_mode     = st.selectbox("Stop mode", ["pct","atr"], index=(["pct","atr"].index(str(_get("stop_mode","pct"))) if str(_get("stop_mode","pct")) in ["pct","atr"] else 0))
    stop_loss_pct = st.number_input("Stop loss % (0–1)", 0.0, 1.0, float(_get("stop_loss_pct", 0.10)))

    atr_window      = st.number_input("ATR window", 1, 1000, int(_get("atr_window", 14)))
    atr_mult        = st.number_input("ATR mult", 0.0, 10.0, float(_get("atr_mult", 2.0)))
    use_atr_trailing= st.checkbox("ATR trailing", value=bool(_get("use_atr_trailing", False)))
    atr_trail_mult  = st.number_input("ATR trail mult", 0.0, 10.0, float(_get("atr_trail_mult", 2.0)))

# Samla ihop param-dict
params = {
    "from_date": from_date, "to_date": to_date,

    "use_rsi_filter": use_rsi_filter, "rsi_window": int(rsi_window),
    "rsi_min": float(rsi_min), "rsi_max": float(rsi_max),

    "use_trend_filter": bool(use_trend_filter),
    "trend_ma_type": str(trend_ma_type), "trend_ma_window": int(trend_ma_window),

    "breakout_lookback": int(breakout_lookback),
    "exit_lookback": int(exit_lookback),

    "use_macd_filter": bool(use_macd_filter),
    "macd_fast": int(macd_fast), "macd_slow": int(macd_slow), "macd_signal": int(macd_signal),

    "use_bb_filter": bool(use_bb_filter),
    "bb_window": int(bb_window), "bb_nstd": float(bb_nstd), "bb_min": float(bb_min),

    "use_stop_loss": bool(use_stop_loss),
    "stop_mode": str(stop_mode), "stop_loss_pct": float(stop_loss_pct),

    "atr_window": int(atr_window), "atr_mult": float(atr_mult),
    "use_atr_trailing": bool(use_atr_trailing), "atr_trail_mult": float(atr_trail_mult),
}

st.markdown("---")
go = st.button("🚀 Kör backtest", width='stretch')

def _run_bt_with_fallback(tick: str, p: dict):
    if GET is None:
        raise RuntimeError("Ingen dataload-funktion tillgänglig.")
    # hämta df
    try:
        df = GET(tick, start=p.get("from_date"), end=p.get("to_date"))
    except TypeError:
        df = GET(tick, p.get("from_date"), p.get("to_date"))
    if df is None or (hasattr(df, "__len__") and len(df)==0):
        raise ValueError("Ingen data returnerades (df=None/empty).")

    # btwrap först
    if RUN_WRAP is not None:
        try:
            res = RUN_WRAP(p={"ticker": tick, "params": dict(p)})
            if isinstance(res, dict) and ("summary" in res or "equity" in res):
                return res
        except Exception:
            try:
                res = RUN_WRAP(tick, dict(p))
                if isinstance(res, dict) and ("summary" in res or "equity" in res):
                    return res
            except Exception:
                pass

    # raw
    if RUN_RAW is not None:
        try:
            res = RUN_RAW(df=df, p=dict(p))
            if isinstance(res, dict):
                return res
        except TypeError:
            res = RUN_RAW(df, dict(p))
            if isinstance(res, dict):
                return res

    raise RuntimeError("Kunde inte köra backtest via btwrap eller raw-funktionen.")

if go:
    if not (ticker or "").strip():
        st.warning("Ange ticker.")
    else:
        try:
            res = _run_bt_with_fallback(ticker.strip(), params)
            st.subheader("Resultat")
            summ = res.get("summary", {})
            if summ:
                st.json(summ)
            else:
                st.info("Inget summary i resultatet.")
            eq = res.get("equity")
            if isinstance(eq, pd.DataFrame) and not eq.empty:
                st.markdown("**Equity (sista 10 rader)**")
                st.dataframe(eq.tail(10), width='stretch')
            else:
                st.info("Ingen equity-tabell i resultatet.")
        except Exception as e:
            st.error(f"Körning misslyckades: {type(e).__name__}: {e}")
            st.code(traceback.format_exc())
else:
    st.info("Inget resultat att visa.")
