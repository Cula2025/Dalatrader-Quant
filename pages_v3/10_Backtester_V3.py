import os, sys, json, glob, time, traceback
from datetime import date
import pandas as pd
import streamlit as st

from app_v3.data_provider_v3 import get_ohlcv
from app_v3.bt_core_v3 import run_backtest

st.set_page_config(page_title="Backtester V3", layout="wide")
st.title("🧪 Backtester V3 — Profilval, indikatorpanel & körning")

def load_profiles_for_ticker(ticker: str):
    rows = []
    for pf in sorted(glob.glob("profiles_v3/*.json")):
        try:
            mtime = os.path.getmtime(pf)
            data = json.load(open(pf, "r", encoding="utf-8"))
            for idx, pr in enumerate(data.get("profiles", [])):
                if pr.get("ticker") == ticker:
                    rows.append({
                        "file": pf,
                        "file_mtime": mtime,
                        "file_mtime_h": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                        "profile_idx": idx,
                        "name": pr.get("name", "(no name)"),
                        "metrics": pr.get("metrics", {}),
                        "params": pr.get("params", {}),
                    })
        except Exception:
            pass
    rows.sort(key=lambda r: r["file_mtime"], reverse=True)
    return rows

def parse_date_str(d):
    y,m,dd = [int(x) for x in str(d).split("-")]
    return date(y,m,dd)

# --- UI topp ---
ticker = st.text_input("Ticker", value="ATCO B")
rows = load_profiles_for_ticker(ticker)
if not rows:
    st.info(f"Inga profiler hittades för “{ticker}” i profiles_v3/.")
    st.stop()

labels = [f"{i}. {r['name']}  ·  {os.path.basename(r['file'])}  ·  {r['file_mtime_h']}  ·  idx={r['profile_idx']}" for i, r in enumerate(rows)]
sel = st.selectbox("Välj profil", options=list(range(len(rows))), format_func=lambda i: labels[i], index=0)
picked = rows[sel]
st.caption(f"Vald: {picked['name']}  (fil: {os.path.basename(picked['file'])}, idx={picked['profile_idx']})")

m = picked.get("metrics", {})
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("BT× (profil)", f"{m.get('TotalReturn','NA')}")
with col2: st.metric("BH×", f"{m.get('BuyHold','NA')}")
with col3: st.metric("SharpeD", f"{m.get('SharpeD','NA')}")
with col4: st.metric("Trades", f"{m.get('Trades','NA')}")

# ------- Indikatorpanel (widgets) -------
st.subheader("Indikatorpanel")
p = dict(picked.get("params", {}))  # kopia

# Datum
c1, c2 = st.columns(2)
dfrom = c1.date_input("from_date", value=parse_date_str(p.get("from_date", "2020-01-01")))
dto   = c2.date_input("to_date",   value=parse_date_str(p.get("to_date", "2025-10-14")))

# Entry/Trend
c1, c2, c3 = st.columns(3)
entry_mode = c1.selectbox("entry_mode", ["ma_cross","donchian"], index=0 if p.get("entry_mode","ma_cross")=="ma_cross" else 1)
trend_ma_type = c2.selectbox("trend_ma_type", ["SMA","EMA"], index=0 if p.get("trend_ma_type","EMA")=="SMA" else 1)
fast = c1.number_input("fast", min_value=1, max_value=300, value=int(p.get("fast", 10)))
slow = c2.number_input("slow", min_value=2, max_value=400, value=int(p.get("slow", 50)))
breakout_lookback = c3.number_input("breakout_lookback", 1, 400, int(p.get("breakout_lookback", 55)))
exit_lookback     = c3.number_input("exit_lookback",     1, 400, int(p.get("exit_lookback", 20)))

# RSI
st.markdown("**RSI-filter**")
c1,c2,c3,c4 = st.columns(4)
use_rsi_filter = c1.checkbox("use_rsi_filter", value=bool(p.get("use_rsi_filter", False)))
rsi_window = c2.number_input("rsi_window", 2, 200, int(p.get("rsi_window", 14)))
rsi_min    = c3.number_input("rsi_min", 0.0, 100.0, float(p.get("rsi_min", 40.0)))
rsi_max    = c4.number_input("rsi_max", 0.0, 100.0, float(p.get("rsi_max", 60.0)))

# ADX / CHOP / OBV
st.markdown("**ADX / CHOP / OBV**")
c1,c2,c3,c4 = st.columns(4)
use_adx_filter = c1.checkbox("use_adx_filter", value=bool(p.get("use_adx_filter", False)))
adx_min        = c2.number_input("adx_min", 0.0, 100.0, float(p.get("adx_min", 20.0)))
use_chop_filter= c3.checkbox("use_chop_filter", value=bool(p.get("use_chop_filter", False)))
chop_max       = c4.number_input("chop_max", 0.0, 100.0, float(p.get("chop_max", 60.0)))

c1,c2 = st.columns(2)
use_obv_filter  = c1.checkbox("use_obv_filter", value=bool(p.get("use_obv_filter", False)))
obv_slope_window= c2.number_input("obv_slope_window", 1, 400, int(p.get("obv_slope_window", 20)))

# ATR / Stop
st.markdown("**ATR / Stop**")
c1,c2,c3 = st.columns(3)
atr_window      = c1.number_input("atr_window", 1, 400, int(p.get("atr_window", 14)))
use_atr_trailing= c2.checkbox("use_atr_trailing", value=bool(p.get("use_atr_trailing", False)))
atr_trail_mult  = c3.number_input("atr_trail_mult", 0.1, 10.0, float(p.get("atr_trail_mult", 2.0)))

c1,c2,c3 = st.columns(3)
use_stop_loss = c1.checkbox("use_stop_loss", value=bool(p.get("use_stop_loss", False)))
stop_mode     = c2.selectbox("stop_mode", ["pct","atr"], index=0 if p.get("stop_mode","pct")=="pct" else 1)
stop_loss_pct = c3.number_input("stop_loss_pct", 0.0, 1.0, float(p.get("stop_loss_pct", 0.08)))
atr_mult      = c3.number_input("atr_mult", 0.1, 10.0, float(p.get("atr_mult", 2.0)))

max_bars_in_trade = st.number_input("max_bars_in_trade", 0, 1000, int(p.get("max_bars_in_trade", 0)))

# JSON spegling av panelen
st.subheader("Parametrar (JSON) — speglar panelen")
panel_params = {
    "from_date": str(dfrom),
    "to_date":   str(dto),
    "entry_mode": entry_mode,
    "trend_ma_type": trend_ma_type,
    "fast": int(fast),
    "slow": int(slow),
    "breakout_lookback": int(breakout_lookback),
    "exit_lookback": int(exit_lookback),
    "use_rsi_filter": bool(use_rsi_filter),
    "rsi_window": int(rsi_window),
    "rsi_min": float(rsi_min),
    "rsi_max": float(rsi_max),
    "use_adx_filter": bool(use_adx_filter),
    "adx_min": float(adx_min),
    "use_chop_filter": bool(use_chop_filter),
    "chop_max": float(chop_max),
    "use_obv_filter": bool(use_obv_filter),
    "obv_slope_window": int(obv_slope_window),
    "atr_window": int(atr_window),
    "use_atr_trailing": bool(use_atr_trailing),
    "atr_trail_mult": float(atr_trail_mult),
    "use_stop_loss": bool(use_stop_loss),
    "stop_mode": stop_mode,
    "stop_loss_pct": float(stop_loss_pct),
    "atr_mult": float(atr_mult),
    "max_bars_in_trade": int(max_bars_in_trade),
}
params_text = st.text_area("Parametrar", value=json.dumps(panel_params, ensure_ascii=False, indent=2), height=360, label_visibility="collapsed")

# --- Kör & spara (med session_state) ---
run_col, save_col = st.columns([1,1])
run_clicked = run_col.button("🚀 Kör backtest")

# initiera state
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

res = None
df = None

if run_clicked:
    try:
        params = json.loads(params_text) if params_text.strip() else {}
        if not isinstance(params, dict):
            raise ValueError("Parametrar måste vara ett JSON-objekt.")
        if not params.get("from_date") or not params.get("to_date"):
            st.error("Parametrar saknar from_date/to_date.")
            st.stop()

        start = parse_date_str(params["from_date"])
        end   = parse_date_str(params["to_date"])

        with st.status("Hämtar data och kör backtest...", expanded=False) as s:
            df = get_ohlcv(ticker=ticker, start=start, end=end)
            if df is None or len(df)==0:
                st.error("Ingen data hämtades (kontrollera ticker/datum)")
                st.stop()
            res = run_backtest(df, params=params)
            s.update(label="Klar", state="complete")

        # Visa resultat
        st.subheader("Resultat")
        eq = res.get("equity")
        chart_df = None
        if isinstance(eq, pd.Series) and len(eq) > 0:
            chart_df = pd.DataFrame({"Strategy": eq})
            if "Close" in df.columns and len(df["Close"]) > 0:
                START_EQ = 100_000.0
                bh = START_EQ * (df["Close"] / df["Close"].iloc[0])
                try:
                    bh = bh.reindex(eq.index).ffill()
                except Exception:
                    pass
                chart_df["Buy&Hold"] = bh
        if chart_df is not None and len(chart_df) > 0:
            st.line_chart(chart_df.dropna())
        else:
            st.info("Ingen equity/Buy&Hold att visa.")

        st.subheader("Metrics")
        st.json(res.get("metrics", {}))

        # ✅ Spara undan senaste körning i session_state
        st.session_state["last_run"] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "params": params,
            "metrics": res.get("metrics", {}),
        }
        st.success("Senaste körning sparad i sessionen. Du kan nu trycka '💾 Spara profil' när som helst.")

    except Exception as e:
        st.error(f"Backtest misslyckades: {e}")
        st.caption("Traceback:")
        st.code(traceback.format_exc())

# 💾 Spara-knapp – alltid synlig om vi har en körning i session_state
if st.session_state.get("last_run"):
    with save_col:
        if st.button("💾 Spara profil till portfolio_v3/active/"):
            try:
                os.makedirs("portfolio_v3/active", exist_ok=True)
                safe_ticker = st.session_state["last_run"]["ticker"].replace(" ", "_").replace("/", "_")
                out_path = f"portfolio_v3/active/V3_Final_{safe_ticker}.json"
                final_prof = {
                    "name": f"{st.session_state['last_run']['ticker']} – final",
                    "ticker": st.session_state["last_run"]["ticker"],
                    "params": st.session_state["last_run"]["params"],
                    "metrics": st.session_state["last_run"]["metrics"],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"profiles": [final_prof]}, f, ensure_ascii=False, indent=2)
                st.success(f"Sparad: {out_path}")
            except Exception as ex:
                st.error(f"Kunde inte spara: {ex}")
else:
    save_col.caption("Kör ett backtest först för att kunna spara.")
