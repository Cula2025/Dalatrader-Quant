from __future__ import annotations
import pandas as pd
import streamlit as st
from app_v3.opt_runner_v3 import optimize

st.set_page_config(page_title="Optimizer V3", page_icon="🧪", layout="wide")
st.title("🧪 Optimizer V3")

tkr  = st.text_input("Ticker", "GETI B")
sims = st.number_input("Simuleringar", 1000, step=500, value=5000)
seed = st.number_input("Seed", 1234, step=1)
start = st.text_input("Start (YYYY-MM-DD)", "2020-10-14")
end   = st.text_input("End (YYYY-MM-DD, tom = idag)", "2025-10-14") or None
procs = st.number_input("Kärnor (processes)", 1, 32, value=4)

run = st.button("Kör optimering")
if run:
    pb = st.progress(0, text="Förbereder …")

    def on_tick(i, n):
        # i är 1-baserad
        frac = min(max(i / float(n), 0.0), 1.0)
        pb.progress(frac, text=f"Kör sim {i}/{n} …")

    with st.spinner("Optimerar…"):
        out = optimize(
            ticker=tkr, sims=int(sims), seed=int(seed),
            start=start, end=end, processes=int(procs),
            on_tick=on_tick,
        )

    pb.progress(1.0, text="Klar!")
    st.success(f"Sparade {len(out.get('profiles', []))} profiler → {out.get('profiles_path')}")

    # ----- Flattena scoreboard till riktiga kolumner -----
    sc = out.get("scoreboard") or []
    if sc:
        df = pd.json_normalize(sc, sep='.')
        # Välj vanliga kolumner (finns inte alla → fyll NaN)
        want = [
            "metrics.SharpeD","metrics.CAGR","metrics.TotalReturn","metrics.BuyHold",
            "metrics.MaxDD","metrics.Trades",
            "params.fast","params.slow","params.entry_mode","params.breakout_lookback","params.exit_lookback",
            "name","ticker"
        ]
        for c in want:
            if c not in df.columns:
                df[c] = pd.NA

        view = (df[want]
                .rename(columns={
                    "metrics.SharpeD":"SharpeD",
                    "metrics.CAGR":"CAGR",
                    "metrics.TotalReturn":"TotalReturn",
                    "metrics.BuyHold":"BuyHold",
                    "metrics.MaxDD":"MaxDD",
                    "metrics.Trades":"Trades",
                    "params.fast":"fast",
                    "params.slow":"slow",
                    "params.entry_mode":"entry",
                    "params.breakout_lookback":"bo_lookback",
                    "params.exit_lookback":"ex_lookback",
                })
                .sort_values("SharpeD", ascending=False)
                .reset_index(drop=True)
        )

        # runda några siffror för visning
        for c in ["SharpeD","CAGR","TotalReturn","BuyHold","MaxDD"]:
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(4)

        st.dataframe(view.head(25), use_container_width=True)

        with st.expander("Visa hela scoreboard (flattenad)"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Inget scoreboard att visa.")
