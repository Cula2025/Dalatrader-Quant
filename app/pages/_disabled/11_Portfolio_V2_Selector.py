import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Läs/spara settings via vår modul
from app.portfolio_settings import SETTINGS, get, set_and_save, save_all

st.set_page_config(page_title="Portfolio V2 – Selector", layout="wide")
st.title("🎛️ Portfolio V2 – Selector (urvals-motor)")
st.caption("Välj slots per dag, vikter (med normalisering), och filter. Spara uppdaterade settings på slutet.")

selector = get("selector", {}) or {}
weights  = selector.get("weights", {}) or {}
filters  = selector.get("filters", {}) or {}
caps     = selector.get("caps", {}) or {}

# Keys vi visar som sliders (matchar nuvarande struktur)
WEIGHT_KEYS = [
    ("total_return",       "Total return"),
    ("win_rate",           "Win rate"),
    ("recent_momentum",    "Recent momentum"),
    ("breakout_z",         "Breakout-Z"),
    ("trend_over_atr",     "Trend / ATR"),
    ("rel_strength_20d",   "Rel. strength 20d"),
    ("freshness",          "Freshness"),
    ("liquidity",          "Liquidity"),
    ("corr_penalty",       "Korrelation (penalty)"),
    ("sector_penalty",     "Sektor (penalty)"),
]

# =============== Layout ===============
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Slots / dag")
    slots = st.number_input("Antal köp-slots per dag", min_value=0, max_value=50, value=int(selector.get("slots_per_day", 2)), step=1)

with c2:
    st.subheader("Vikter")
    st.caption("Dra i reglagen. Du kan välja 'Auto-normalisera' så att summan blir 1.0 automatiskt.")
    auto_norm = st.toggle("Auto-normalisera vikter till 1.0", value=True)

    new_weights = {}
    # Sliders i 2 kolumner för bättre översikt
    left, right = st.columns(2)
    for i, (k, label) in enumerate(WEIGHT_KEYS):
        col = left if i % 2 == 0 else right
        with col:
            val = float(weights.get(k, 0.0))
            new_weights[k] = st.slider(label, 0.0, 1.0, val, 0.01, key=f"w_{k}")

    # Summering + ev. normalisering
    w_sum = sum(new_weights.values())
    st.write(f"**Summa vikter:** {w_sum:.3f}")
    if w_sum == 0:
        st.warning("Summan är 0. Urvalet ger då ingen effekt.")
        weights_final = new_weights
    else:
        if auto_norm:
            weights_final = {k: (v / w_sum) for k, v in new_weights.items()}
            st.info("Vikter auto-normaliseras till 1.0 inför sparning.")
        else:
            weights_final = new_weights
            if abs(w_sum - 1.0) > 1e-6:
                st.warning("Summan ≠ 1.0 och auto-normalisering är av. Det kan ge svårtolkade poäng.")

    # Liten vikt-visualisering (för att slippa Altair-varningar gör vi en enkel tabell)
    df_weights = pd.DataFrame(
        [{"key": lbl, "weight": weights_final[k]} for k, lbl in WEIGHT_KEYS]
    ).sort_values("weight", ascending=False).reset_index(drop=True)
    st.dataframe(df_weights, hide_index=True, use_container_width=True)

st.divider()

# =============== Filter ===============
st.subheader("Filter")
fc1, fc2 = st.columns([1, 2])
with fc1:
    min_days = int(filters.get("min_days_since_last_exit", 0))
    min_days = st.number_input("Min dagar sedan senaste exit", min_value=0, max_value=365, value=min_days, step=1)

with fc2:
    # För "exclude": försök hjälp med tickers från profiler, annars fritext
    try:
        profiles = list(Path("profiles").glob("*.json"))
        opt = []
        for p in profiles:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                profs = d.get("profiles") or []
                if profs:
                    t = profs[0].get("ticker") or (profs[0].get("params") or {}).get("ticker")
                    if t: opt.append(str(t))
            except Exception:
                pass
        opt = sorted(set(opt))
    except Exception:
        opt = []

    exclude_default = filters.get("exclude", []) or []
    if opt:
        exclude = st.multiselect("Exkludera tickers", options=opt, default=[x for x in exclude_default if x in opt])
    else:
        st.caption("Inga profil-tickers hittade – skriv kommaseparerad lista.")
        exclude_str = st.text_input("Exkludera tickers (t.ex. 'EQT,BOL')", ",".join(map(str, exclude_default)))
        exclude = [x.strip() for x in exclude_str.split(",") if x.strip()]

# =============== Caps (om någon struktur redan finns) ===============
st.subheader("Caps (valfritt)")
caps_ui = {}
if isinstance(caps, dict) and caps:
    for k, v in caps.items():
        if isinstance(v, (int, float)):
            caps_ui[k] = st.number_input(f"{k}", value=float(v))
        else:
            caps_ui[k] = v
else:
    st.caption("Inga caps definierade i settings ännu.")

st.divider()

# =============== Spara ===============
b1, b2, b3 = st.columns([1,1,3])
with b1:
    if st.button("💾 Spara"):
        new_selector = {
            "slots_per_day": int(slots),
            "weights": weights_final,
            "filters": {
                "min_days_since_last_exit": int(min_days),
                "exclude": exclude,
            },
            "caps": caps_ui if caps_ui else caps,
        }
        # uppdatera SETTINGS och spara
        SETTINGS["selector"] = new_selector
        save_all(SETTINGS)
        st.success("Selector sparad.")

with b2:
    if st.button("↩️ Återställ (ladda om)"):
        st.rerun()

# Visa rå JSON (för kontroll)
with st.expander("Rå JSON för selector (läs-endast)", expanded=False):
    st.code(json.dumps(SETTINGS.get("selector", {}), ensure_ascii=False, indent=2), language="json")
