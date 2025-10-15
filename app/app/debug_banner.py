# -*- coding: utf-8 -*-
def show(text="Debug 0.1"):
    # Liten klistrig banner högst upp på sidan
    import streamlit as st
    html = f"""
    <div style="position:sticky;top:0;z-index:1000;
                background:#0f172a;color:#facc15;
                padding:6px 10px;margin:-16px -16px 8px -16px;
                border-bottom:1px solid rgba(255,255,255,.15);
                font-weight:600;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
        {text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
