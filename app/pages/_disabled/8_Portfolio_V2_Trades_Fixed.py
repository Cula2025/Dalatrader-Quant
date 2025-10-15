from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd

try:
    from app.backtracker import run_backtest as RUN
    motor = "backtracker"
except Exception:
    from app.btwrap import run_backtest as RUN
    motor = "btwrap"

from app.trades_util import trades_df_from_result

st.title("🧾 Trades – Portfolio V2 (läs-endast, robust)")
all_files = sorted(Path("profiles").glob("*.json"))
defaults  = [str(p) for p in all_files[:3]]
files = st.multiselect("Profilfiler (bästa profilen per fil)", [str(p) for p in all_files], default=defaults)

if not files:
    st.info("Välj en eller flera profilfiler ovan.")
    st.stop()

rows, errs = [], []
for f in files:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        profiles = d.get("profiles") or []
        if not profiles:
            errs.append((f, "Ingen profil i filen")); continue
        best = max(profiles, key=lambda p: (p.get("metrics") or {}).get("TotalReturn", float("-inf")))
        params = dict(best.get("params") or {})
        ticker = best.get("ticker") or params.get("ticker")
        name   = best.get("name") or f"{ticker} – best"
        res    = RUN(p={"ticker": ticker, "params": params})
        df     = trades_df_from_result(res, ticker=ticker, profile_name=name)
        if df.empty:
            errs.append((f, "Inga trades eller kunde inte tolka datum")); continue
        df["File"] = Path(f).name
        rows.append(df)
    except Exception as e:
        errs.append((f, f"{type(e).__name__}: {e}"))

for f,msg in errs:
    st.error(f"{Path(f).name}: {msg}")

if not rows:
    st.info("Inga trades att visa – kontrollera fel ovan.")
    st.stop()

trades = pd.concat(rows).sort_index()
st.caption(f"Motor: {motor} • Rader: {len(trades)}")
st.dataframe(trades.rename_axis("Date").reset_index().set_index("Date"), use_container_width=True)

with st.expander("Summering per Ticker / Profile"):
    agg = trades.groupby(["Ticker","Profile"]).agg(
        trades=("cash_flow","size"),
        cash_in=("cash_flow", lambda s: s[s<0].sum()),
        cash_out=("cash_flow", lambda s: s[s>0].sum()),
        net_cash=("cash_flow","sum"),
        volume=("value_abs","sum"),
    )
    st.dataframe(agg, use_container_width=True)
