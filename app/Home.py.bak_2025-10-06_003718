# -*- coding: utf-8 -*-
from datetime import date
from pathlib import Path
import streamlit as st
from app.debug_banner import show as debug_banner
debug_banner('Debug 0.1')

# Branding (logo, fonter, CSS, page config)
try:
    from app.branding import apply as brand
except Exception:
    def brand(*a, **k): pass
brand(page_title="Dala Trader – Home", page_icon="🐎")

# --- header med logohäst istället för emoji ---
from pathlib import Path as _P
try:
    from PIL import Image as _Image  # för ev. cropp
except Exception:
    _Image = None

_BASE = _P(__file__).resolve().parent         # /srv/trader/app
_ASSETS = _BASE / "assets"
_HORSE  = _ASSETS / "horse.png"
_LOGO   = _ASSETS / "logodaladrader.png"

# layout: liten ikon + titeltext
col1, col2 = st.columns([0.10, 1])
with col1:
    _img = _HORSE if _HORSE.exists() else _LOGO
    if _img.exists():
        if _Image is not None and _img == _LOGO:
            # croppa vänstra kvadraten ur fulla loggan (heuristik)
            try:
                _im = _Image.open(_img)
                w, h = _im.size
                m = min(w, h)
                _crop = _im.crop((0, 0, m, m))
                st.image(_crop, width=48)
            except Exception:
                st.image(str(_img), width=48)
        else:
            st.image(str(_img), width=48)
with col2:
    st.markdown('<h1 style="margin:0">Dala Trader</h1>', unsafe_allow_html=True)
# --- /header ---
st.caption("Kvantitativa verktyg för portföljer, optimering och backtesting.")

# Snabblänkar (fungerar i nya Streamlit; annars informativ fallback)
st.markdown("### Snabbnavigering")
try:
    st.page_link("pages/2_Portfolio.py", label="📈 Öppna Portfolio", icon="💼")
    st.page_link("pages/0_Optimizer.py", label="⚙️ Öppna Optimizer", icon="⚙️")
    st.page_link("pages/1_Backtrack.py", label="⏪ Öppna Backtrack", icon="⏪")
except Exception:
    st.info("Använd sidomenyn för att navigera till Portfolio/Optimizer/Backtrack.")

st.markdown("---")
st.subheader("Kom igång")
st.markdown(
    "- Gå till **Portfolio** för att köra profiler mot universum och jämföra mot Buy&Hold/OMXS30.\n"
    "- Gå till **Optimizer** för att hitta parametrar (RSI, breakout, stoploss m.m.).\n"
    "- Gå till **Backtrack** för att testa sparade profiler mot andra perioder."
)
