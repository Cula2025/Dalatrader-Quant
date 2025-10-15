import json, pathlib
import pandas as pd
import streamlit as st
from app.data_providers import get_ohlcv
from app.equity_extract import extract_equity

try:
    from app.backtracker import run_backtest
    ENGINE = "backtracker"
except Exception:
    from app import btwrap as W
    run_backtest = W.run_backtest  # type: ignore
    ENGINE = "btwrap"

st.set_page_config(page_title="Portfolio V2", page_icon="📊", layout="wide")
st.title("📊 Portfolio V2")
st.caption(f"Engine: **{ENGINE}**  ·  Jämför tre profiler i en fil")

def pick_equity_payload(res: dict):
    for k in ("equity","summary"):
        if k in res and res[k] is not None:
            return res[k]
    return res

profs = sorted(pathlib.Path("profiles").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not profs:
    st.warning("Hittar inga filer i profiles/*.json")
    st.stop()

fname = st.selectbox("Profilfil", [p.name for p in profs])
data  = json.loads((pathlib.Path("profiles")/fname).read_text(encoding="utf-8"))
plist = data.get("profiles") or []
if not plist:
    st.warning("Inga profiler i filen.")
    st.stop()

rows = []
series = {}
for p in plist:
    name   = p.get("name","<namn saknas>")
    params = dict(p.get("params") or {})
    ticker = p.get("ticker") or params.get("ticker")

    with st.spinner(f"Kör {name} / {ticker}…"):
        res = run_backtest(p={"ticker": ticker, "params": params})
    eq = extract_equity(pick_equity_payload(res))
    tr_calc = (float(eq.iloc[-1]) - 1.0) if len(eq) else float("nan")
    facit   = (p.get("metrics") or {}).get("TotalReturn")

    # BH mot samma fönster
    df = get_ohlcv(ticker=ticker, start=params.get("from_date"), end=None)
    if df is not None and len(df) >= 2:
        bh_calc = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0)
    else:
        bh_calc = float("nan")

    rows.append({
        "Name": name,
        "Ticker": ticker,
        "TR(calc)": tr_calc,
        "TR(facit)": float(facit) if facit is not None else float("nan"),
        "BH(calc)": bh_calc,
        "Bars": int(len(eq)),
    })
    # normaliserad kurva för jämförelse
    if len(eq):
        series[name] = pd.Series(eq.values, index=eq.index, name=name)

df_out = pd.DataFrame(rows).set_index("Name")
st.dataframe(df_out, use_container_width=True)

# Kombinerad graf (om vi har serier)
if series:
    # Normalisera till 1.0 i start för jämförelse
    norm = {k: (s / float(s.iloc[0])) for k,s in series.items() if len(s)}
    st.line_chart(pd.DataFrame(norm))
