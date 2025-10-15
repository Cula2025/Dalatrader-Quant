# app/save_prices_ui.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from app.backtest_simple import load_ohlcv

st.set_page_config(page_title="Spara Börsdata", layout="centered")
st.title("Spara Börsdata → CSV")

with st.sidebar:
    st.markdown("**Inställningar**")
    default_ticker = st.text_input("Ticker (Börsdata-stil)", value="VOLV B")
    colA, colB = st.columns(2)
    with colA:
        start = st.date_input("Start", value=dt.date(2020,1,1))
    with colB:
        end = st.date_input("Slut", value=dt.date.today())

    out_dir = st.text_input("Ut-katalog", value="outputs/prices")
    save_name = st.text_input("Filnamn (valfritt, tomt = auto)", value="")

st.info("Välj ticker och datum i sidopanelen, klicka sedan **Spara**.")

load_dotenv()

btn = st.button("💾 Spara Börsdata till CSV", type="primary")
if btn:
    try:
        df = load_ohlcv("borsdata", default_ticker, start.isoformat(), end.isoformat())
        if df is None or df.empty:
            st.error("Inga priser hämtades – kontrollera ticker/datum.")
        else:
            df2 = df.copy()
            df2.index.name = "Date"
            cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df2.columns]
            df2 = df2[cols]

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            if save_name.strip():
                out_fp = Path(out_dir) / save_name.strip()
            else:
                safe = default_ticker.replace(" ", "_").replace("/", "-")
                out_fp = Path(out_dir) / f"{safe}_{start}_{end}.csv"

            df2.to_csv(out_fp)
            st.success(f"Sparat: {out_fp}")
            st.dataframe(df2.tail(10))
    except Exception as e:
        st.exception(e)



