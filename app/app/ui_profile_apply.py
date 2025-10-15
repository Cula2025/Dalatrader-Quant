# -*- coding: utf-8 -*-
from pathlib import Path
import json
import streamlit as st

PROFILES_DIR = Path("outputs/opt_results").resolve()

def _read_profiles(fp: Path) -> list[dict]:
    """Läs profiler ur json (stöder både {profiles:[...]} och single-profil)."""
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        st.sidebar.error(f"Profilmport misslyckades: {e}")
        return []
    profs = data.get("profiles") or []
    if isinstance(data, dict) and not profs:
        profs = [data]
    out = []
    for p in profs:
        out.append({
            "name": p.get("name") or p.get("profile"),
            "ticker": (p.get("ticker") or "").strip(),
            "params": p.get("params") or {},
            "metrics": p.get("metrics") or {},
        })
    return out

def _keymap() -> dict:
    """Mappar profilens nycklar -> dina widget-keys i sidan."""
    return {
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

        # extra som vi sett i dina filer
        "use_atr_trailing": "use_atr_trailing",
        "atr_trail_mult": "atr_trail_mult",

        # meta
        "sims": "sims",
        "seed": "seed",
        "from_date": "from_date",
        "to_date": "to_date",
    }

def apply_profile_to_state(prof: dict, *, selected_file: Path | None):
    """Skriv profilens värden till st.session_state så widgets får dem."""
    if not prof:
        return
    # Ticker
    t = (prof.get("ticker") or "").strip()
    if t:
        st.session_state["ticker"] = t
    # Parametrar
    params = prof.get("params") or {}
    for src, dst in _keymap().items():
        if src in params:
            st.session_state[dst] = params[src]
    # Metadata för header
    if selected_file is not None:
        st.session_state["loaded_profile_file"] = selected_file.name
    st.session_state["loaded_profile_name"] = prof.get("name") or prof.get("profile") or (selected_file.stem if selected_file else "-")

def render_profile_header():
    st.markdown(
        "**Vald fil:** `{}` | **Profil:** `{}` | **Ticker:** `{}`".format(
            st.session_state.get("loaded_profile_file", "-"),
            st.session_state.get("loaded_profile_name", "-"),
            st.session_state.get("ticker", "-"),
        )
    )

def render_profiles_sidebar():
    """Sidopanel: välj fil -> välj profil -> ☑️ Använd vald profil."""
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

    profiles = _read_profiles(fp)
    if not profiles:
        st.sidebar.warning("Filen innehåller inga profiler.")
        return

    def _label(i: int) -> str:
        p = profiles[i]
        t = p.get("ticker") or ""
        n = p.get("name") or f"Profil {i+1}"
        tr = p.get("metrics", {}).get("TotalReturn")
        suf = f" • TR={float(tr):.2f}×" if tr is not None else ""
        return (f"{t} – {n}{suf}") if t else (f"{n}{suf}")

    def_idx = 0
    prof_idx = st.sidebar.selectbox(
        "Välj profil i filen",
        options=list(range(len(profiles))),
        index=def_idx,
        format_func=_label,
        key="srv_profile_idx",
    )

    # Bara applicera när användaren väljer det:
    if st.sidebar.checkbox("☑️ Använd vald profil", value=False, key="srv_profile_apply"):
        apply_profile_to_state(profiles[prof_idx], selected_file=fp)
