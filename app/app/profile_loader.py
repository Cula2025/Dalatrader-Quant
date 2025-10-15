# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

PROFILES_DIR = Path("/srv/trader/app/profiles")

# vilka params i filen mappar till vilka widget-keys i Backtest-sidan
KEYMAP = {
    "use_rsi_filter": "use_rsi_filter",
    "rsi_window": "rsi_window",
    "rsi_min": "rsi_min",
    "rsi_max": "rsi_max",

    "use_trend_filter": "use_trend_filter",
    "trend_ma_type": "trend_ma_type",
    "trend_ma_window": "trend_ma_window",

    "breakout_lookback": "breakout_lookback",
    "exit_lookback": "exit_lookback",

    "use_macd_filter": "use_macd_filter",
    "macd_fast": "macd_fast",
    "macd_slow": "macd_slow",
    "macd_signal": "macd_signal",

    "use_bb_filter": "use_bb_filter",
    "bb_window": "bb_window",
    "bb_nstd": "bb_nstd",
    "bb_min": "bb_min",

    "use_stop_loss": "use_stop_loss",
    "stop_mode": "stop_mode",
    "stop_loss_pct": "stop_loss_pct",

    "atr_window": "atr_window",
    "atr_mult": "atr_mult",
    "use_atr_trailing": "use_atr_trailing",
    "atr_trail_mult": "atr_trail_mult",
}

def _list_profile_files() -> List[Path]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p for p in PROFILES_DIR.glob("*.json") if p.is_file())

def _load_profiles(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = data.get("profiles") or []
        return profiles
    except Exception as e:
        st.sidebar.error(f"Kunde inte läsa profilfilen: {type(e).__name__}: {e}")
        return []

def _apply_to_session(profile: Dict[str, Any]) -> None:
    # ticker
    t = (profile.get("ticker") or "").strip()
    if t:
        st.session_state["ticker"] = t

    # datum (om finns)
    p = profile.get("params") or {}
    from_date = p.get("from_date") or profile.get("from_date")
    to_date   = p.get("to_date")   or profile.get("to_date")
    if from_date:
        st.session_state["from_date"] = str(from_date)
    if to_date:
        st.session_state["to_date"] = str(to_date)

    # parametrar
    for src, dst in KEYMAP.items():
        if src in p:
            st.session_state[dst] = p[src]

def render_sidebar():
    with st.sidebar.expander("📂 Ladda profil", expanded=True):
        files = _list_profile_files()
        if not files:
            st.info("Inga .json-profiler i /srv/trader/app/profiles")
            return

        # välj fil
        file_labels = [f.name for f in files]
        file_idx = st.selectbox("Profilfil", options=list(range(len(files))),
                                format_func=lambda i: file_labels[i], key="bf_file_idx")
        fp = files[file_idx]

        # välj profil i filen
        profiles = _load_profiles(fp)
        if not profiles:
            return

        def _lab(i: int) -> str:
            p = profiles[i]
            n = p.get("name") or p.get("profile") or f"Profil {i+1}"
            t = (p.get("ticker") or "").strip()
            return f"{t} – {n}" if t else n

        prof_idx = st.selectbox("Välj profil", options=list(range(len(profiles))),
                                format_func=_lab, key="bf_prof_idx")
        chosen = profiles[prof_idx]

        # Enda knappen
        if st.button("✅ Använd i Backtest", key="bf_apply"):
            _apply_to_session(chosen)
            st.success("Profil applicerad i Backtest-parametrarna.")
            st.rerun()
