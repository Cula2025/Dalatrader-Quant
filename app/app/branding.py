# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import streamlit as st

def apply(page_title: str | None = None, page_icon: str = "💹"):
    """
    Ladda logga, fonter och CSS. Kan även sätta page_config om du skickar in page_title.
    Säker att anropa flera gånger (no-op om redan gjort).
    """
    # set_page_config (ignorera om redan satt)
    if page_title:
        try:
            st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
        except Exception:
            pass

    base = Path(__file__).resolve().parents[1]  # /srv/trader/app
    logo = base / "assets" / "logodaladrader.png"
    css  = base / "assets" / "theme.css"

    # Google Fonts
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
h1,h2,h3 { font-family: 'Space Grotesk', system-ui, sans-serif; letter-spacing:.2px; }
html, body, [class*="st-"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
</style>
""", unsafe_allow_html=True)

    # Theme CSS om den finns
    if css.exists():
        try:
            st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)
        except Exception:
            pass

    # Logga i sidomenyn
    try:
        with st.sidebar:
            if logo.exists():
                st.image(str(logo), width='stretch')
                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    except Exception:
        pass
