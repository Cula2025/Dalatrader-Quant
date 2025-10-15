from __future__ import annotations
from pathlib import Path
import streamlit as st

# ---- Fixed 4-color palette ----
BLACK = "#000000"
YELLOW = "#E6A626"   # tweak if your logo uses a different hex
WHITE = "#FFFFFF"
DARK = "#111111"     # secondary surface

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

def find_logo() -> Path | None:
    for patt in ["dalatrader_logo.png", "dalatrader-logo.png", "dalatrader.png"]:
        p = ASSETS / patt
        if p.exists():
            return p
    for p in ASSETS.glob("dalat*logo.*"):
        return p
    for p in ASSETS.glob("dalar*logo.*"):
        return p
    return None

def inject_base_styles() -> None:
    st.markdown(
        f"""
        <style>
          html, body, [data-testid="stAppViewContainer"] {{
            background: {BLACK};
            color: {WHITE};
          }}
          .block-container {{ max-width: 1200px; padding-top: 16px; }}
          .stButton>button {{
            background: {YELLOW};
            color: {BLACK};
            font-weight: 800;
            font-size: 18px;
            padding: 14px 22px;
            border: none; border-radius: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,.35);
          }}
          .stButton>button:hover {{ filter: brightness(1.06); }}
          .card {{
            background: {DARK};
            border: 1px solid #222;
            border-radius: 14px; padding: 16px;
            box-shadow: 0 2px 16px rgba(0,0,0,.5);
          }}
          .kpi {{ display:inline-flex; gap:.4rem; background:#1a1a1a; color:{WHITE};
                 padding:.25rem .55rem; border-radius:9999px; font-size:.8rem; }}
          h1,h2,h3,h4,h5,h6 {{ color: {WHITE}; }}
          .accent {{ color: {YELLOW}; }}
          .muted {{ color: #cccccc; }}
          .dl-btn button {{ background:{DARK} !important; color:{WHITE} !important; border:1px solid #333 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def header(title: str = "Dalatrader", subtitle: str = "Quant trading specialists", logo_width: int = 110) -> None:
    logo = find_logo()
    col_logo, col_title = st.columns([1, 8], vertical_alignment="center")
    with col_logo:
        if logo:
            st.image(str(logo), width=logo_width)
        else:
            st.write(" ")
    with col_title:
        st.markdown(f"<h1 style='margin:0'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='accent' style='font-weight:700;margin-top:-6px'>{subtitle}</div>", unsafe_allow_html=True)
    st.divider()

def setup_page(page_title: str, page_icon: str = "📈") -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    inject_base_styles()
