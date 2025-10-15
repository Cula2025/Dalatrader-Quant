import streamlit as st
st.set_page_config(page_title="Trader", page_icon="📊")
st.title("Trader – startsida")

st.subheader("V2")
st.page_link("pages/7_Portfolio_V2_Lines.py", label="Portfolio V2 – linjer", icon="📈")

st.subheader("Classic")
st.page_link("pages/0_Optimizer.py", label="Optimizer", icon="🧪")
st.page_link("pages/1_Backtest.py", label="Backtest", icon="🔁")
