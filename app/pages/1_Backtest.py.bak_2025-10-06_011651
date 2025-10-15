from __future__ import annotations
import inspect, traceback, os, datetime as dt
from typing import Any, Optional, Tuple
import streamlit as st

# --- trygg debug-import ---
try:
    from app.debuglog import log_info, log_warn, log_error, setup_debug_ui
except Exception:
    def log_info(*a, **k): pass
    def log_warn(*a, **k): pass
    def log_error(*a, **k): pass
    def setup_debug_ui(*a, **k): pass

st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
st.title("🧪 Backtest")
setup_debug_ui(st)

def _detect_engine() -> Tuple[Optional[Any], str]:
    candidates = [
        ("app.backtest", "run_backtest"),
        ("app.backtest", "run"),
        ("app.engine", "run_backtest"),
        ("app.backtester", "run_single"),
    ]
    for mod, attr in candidates:
        try:
            m = __import__(mod, fromlist=[attr])
            fn = getattr(m, attr, None)
            if callable(fn):
                return fn, f"{mod}.{attr}"
        except Exception:
            continue
    return None, ""

ENGINE_FN, ENGINE_NAME = _detect_engine()
st.caption(f"Motor: {'—' if ENGINE_FN is None else ENGINE_NAME}")

# --- ticker-widget utan varningsspam ---
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "GETI B"
ticker = st.text_input("Ticker", key="ticker")

run_btn = st.button("Kör backtest")

def _call_engine(fn, ticker: str):
    """Anropa motorn tolerant: stöd ticker som kw eller positionellt."""
    try:
        sig = inspect.signature(fn)
        params = {p.name for p in sig.parameters.values()}
        if "ticker" in params:
            return fn(ticker=ticker)
        elif "symbol" in params:
            return fn(symbol=ticker)
        elif "code" in params:
            return fn(code=ticker)
        else:
            return fn(ticker)  # positionellt
    except Exception as e:
        log_error(f"Engine call failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "trace": traceback.format_exc()}

if run_btn:
    if ENGINE_FN is None:
        st.error("Ingen backtest-motor hittad i appen.")
        st.stop()

    with st.spinner(f"Kör backtest för {ticker}…"):
        res = _call_engine(ENGINE_FN, ticker)

    # Normalisera None → {}
    if res is None:
        st.warning("Backtest gav inget resultat (res=None). Kontrollera datakälla/nyckel.")
        res = {}

    summary = (res or {}).get("summary", {})
    if isinstance(summary, dict) and summary:
        st.subheader("Sammanfattning")
        for k, v in summary.items():
            st.write(f"- **{k}**: {v}")

    st.subheader("Rått resultat")
    try:
        out_dir = "trader/outputs"
        os.makedirs(out_dir, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"backtest_{ticker}_{ts}.json")
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        st.caption(f"Sparat: `{out_path}`")
    except Exception:
        pass
    st.json(res, expanded=False)
