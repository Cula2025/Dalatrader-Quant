# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os

try:
    import streamlit as st
    HAS_ST = True
except Exception:
    st = None
    HAS_ST = False

PROFILE_QUEUE_KEY = "__pending_profile__"

PARAM_KEYS = [
    "use_rsi_filter","rsi_window","rsi_min","rsi_max",
    "use_trend_filter","trend_ma_type","trend_ma_window",
    "breakout_lookback","exit_lookback",
    "use_macd_filter","macd_fast","macd_slow","macd_signal",
    "use_bb_filter","bb_window","bb_nstd","bb_min",
    "use_stop_loss","stop_mode","stop_loss_pct",
    "atr_window","atr_mult","use_atr_trailing","atr_trail_mult",
]
DATE_KEYS = ["from_date","to_date"]

def _ui_error(msg: str) -> None:
    if HAS_ST:
        try:
            st.sidebar.error(msg)
        except Exception:
            try:
                st.error(msg)
            except Exception:
                print(msg)
    else:
        print(msg)

def _apply_profile_to_state(prof: dict) -> None:
    """Applicera profilens ticker/datum/params till session_state."""
    if not isinstance(prof, dict) or not HAS_ST:
        return
    # Ticker
    t = (prof.get("ticker") or "").strip()
    if t:
        st.session_state["ticker"] = t
    # Datum
    for k in DATE_KEYS:
        v = prof.get(k) or ""
        if v:
            st.session_state[k] = str(v)
    # Parametrar
    params = prof.get("params") or {}
    for k in PARAM_KEYS:
        if k in params:
            st.session_state[k] = params[k]

def pre_apply() -> None:
    """Kallas TIDIGT i skriptet, innan widgets skapas."""
    if not HAS_ST:
        return
    if PROFILE_QUEUE_KEY in st.session_state:
        prof = st.session_state.get(PROFILE_QUEUE_KEY)
        try:
            _apply_profile_to_state(prof or {})
        finally:
            del st.session_state[PROFILE_QUEUE_KEY]

def queue_profile_apply(json_path: str, profile_name: str) -> None:
    """Lägg profil i kö och trigga rerun."""
    try:
        if not json_path:
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles", [])
        chosen = None
        for p in profiles:
            if p.get("name") == profile_name:
                chosen = p
                break
        if chosen is None and profiles:
            chosen = profiles[0]
        if chosen and HAS_ST:
            st.session_state[PROFILE_QUEUE_KEY] = chosen
            st.rerun()
        elif not chosen:
            _ui_error("Hittade ingen profil i filen.")
    except Exception as e:
        _ui_error(f"Kunde inte läsa profilfilen: {type(e).__name__}: {e}")
