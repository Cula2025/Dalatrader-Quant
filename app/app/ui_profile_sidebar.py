from pathlib import Path
from datetime import date as _date
import json
import streamlit as st

PROFILES_DIR = (Path("outputs") / "opt_results").resolve()

def render_profiles_sidebar():
    st.sidebar.header("Optimeringsprofil")
    st.sidebar.caption(f"Profilkatalog: `{PROFILES_DIR}`")

    try:
        files = sorted(PROFILES_DIR.glob("*.json"))
    except Exception as e:
        st.sidebar.error(f"Kunde inte läsa profiler: {e}")
        return
    if not files:
        st.sidebar.info("Inga *.json-profiler hittades.")
        return

    file_idx = st.sidebar.selectbox(
        "Välj optimeringsfil",
        options=list(range(len(files))),
        format_func=lambda i: files[i].name,
        key="srv_profile_file_idx",
    )
    fp = files[file_idx]

    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        profiles = data.get("profiles") or []
        if isinstance(data, dict) and not profiles:
            profiles = [data]
    except Exception as e:
        st.sidebar.error(f"Profilmport misslyckades: {e}")
        return
    if not profiles:
        st.sidebar.warning("Filen innehåller inga profiler.")
        return

    def _label(i: int) -> str:
        p = profiles[i]
        t = (p.get("ticker") or "").strip()
        n = p.get("name") or p.get("profile") or f"Profil {i+1}"
        tr = p.get("metrics", {}).get("TotalReturn")
        suf = f" • TR={float(tr):.2f}×" if tr is not None else ""
        return (f"{t} – {n}{suf}") if t else (f"{n}{suf}")

    prof_idx = st.sidebar.selectbox(
        "Välj profil i filen",
        options=list(range(len(profiles))),
        format_func=_label,
        key="srv_profile_idx",
    )
    sel = profiles[prof_idx]

    # Mappa profilnycklar -> dina widget-keys
    keymap = {
        "use_trend_filter": "use_trend_filter",
        "trend_ma_window": "trend_ma_window",
        "rsi_window": "rsi_window",
        "rsi_min": "rsi_min",
        "rsi_max": "rsi_max",
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
        "atr_trail_mult": "atr_trail_mult",
        "sims": "sims",
        "seed": "seed",
    }

    if st.sidebar.checkbox("☑️ Använd vald profil", key="apply_server_profile"):
        # Meta
        st.session_state["loaded_profile_file"] = fp.name
        st.session_state["loaded_profile_name"] = sel.get("name") or sel.get("profile") or fp.stem
        # Ticker
        t = (sel.get("ticker") or "").strip()
        if t:
            st.session_state["ticker"] = t
        # Parametrar
        params = sel.get("params") or {}
        for src, dst in keymap.items():
            if src in params:
                st.session_state[dst] = params[src]
        # Basdatum om saknas
        today = _date.today()
        st.session_state.setdefault("from_date", today.replace(year=today.year-5).isoformat())
        st.session_state.setdefault("to_date", today.isoformat())
        st.sidebar.success("Profil applicerad på parametrarna.")
