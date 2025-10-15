import json, math
from pathlib import Path
import streamlit as st

try:
    import pandas as pd
except Exception:  # Streamlit kör sin egen env
    pd = None

PROFILES_DIR = Path("profiles")

def _load_profile_params(ticker: str) -> dict:
    fp = PROFILES_DIR / f"{ticker}.json"
    if not fp.exists():
        return {}
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        p0 = (d.get("profiles") or [None])[0] or {}
        return dict(p0.get("params") or {})
    except Exception:
        return {}

def _save_profile_params(ticker: str, params: dict) -> bool:
    fp = PROFILES_DIR / f"{ticker}.json"
    if not fp.exists():
        payload = {"profiles": [{"name": f"default:{ticker}",
                                 "ticker": ticker,
                                 "params": params}]}
    else:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {"profiles": []}
        except Exception:
            payload = {"profiles": []}
        profs = payload.setdefault("profiles", [])
        if not profs:
            profs.append({"name": f"default:{ticker}", "ticker": ticker, "params": {}})
        profs[0]["ticker"] = ticker
        profs[0]["params"] = params

    try:
        bkp = fp.with_suffix(fp.suffix + ".bak")
        if fp.exists():
            bkp.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

# Heuristiska ranges (kan byggas ut)
RANGES = {
    "atr_window": (2, 200, 1),
    "bb_window": (2, 200, 1),
    "breakout_lookback": (2, 400, 1),
    "exit_lookback": (1, 200, 1),
    "rsi_window": (2, 100, 1),
    "sma_window": (2, 400, 1),
    "atr_mult": (0.0, 10.0, 0.05),
    "atr_trail_mult": (0.0, 10.0, 0.05),
    "bb_min": (0.0, 1.0, 0.01),
    "bb_nstd": (0.1, 5.0, 0.05),
    "breakout_z": (0.0, 5.0, 0.05),
    "trend_over_atr": (0.0, 10.0, 0.05),
    "rsi_buy": (0.0, 100.0, 1.0),
    "rsi_sell": (0.0, 100.0, 1.0),
}

def _float_step(v: float) -> float:
    if v == 0:
        return 0.01
    try:
        return max(0.01, 10 ** (math.floor(math.log10(abs(v))) - 2))
    except Exception:
        return 0.01

def _render_control(k, v):
    rng = RANGES.get(k)
    if isinstance(v, bool):
        return st.checkbox(k, value=bool(v))
    if isinstance(v, int):
        if rng:
            lo, hi, step = rng
            return st.number_input(k, min_value=int(lo), max_value=int(hi), step=int(step), value=int(v))
        return st.number_input(k, step=1, value=int(v))
    if isinstance(v, float) or isinstance(v, int):
        fval = float(v)
        if rng:
            lo, hi, step = rng
            return st.slider(k, min_value=float(lo), max_value=float(hi), step=float(step), value=float(fval))
        return st.number_input(k, value=fval, step=_float_step(fval), format="%.6f")
    # fallback: str
    return st.text_input(k, value=str(v))

def render_params_panel(ticker: str):
    """
    Renderar en kompakt parameterpanel.
    Returnerar (params_final:dict|None, df_override:DataFrame|None)
    """
    if not ticker:
        return None, None

    st.markdown("### Parametrar")
    pc = _load_profile_params(ticker)
    params = dict(pc)

    c1, c2 = st.columns(2)
    ui = {}
    keys = sorted(params.keys())
    # Om profilen är tom – visa bara info + rå-json editor under expander
    if not keys:
        st.info("Inga kända parametrar i profilen. Du kan spara via rå-JSON längre ned.")
    for i, k in enumerate(keys):
        with (c1 if i % 2 == 0 else c2):
            ui[k] = _render_control(k, params[k])

    # Startdatum
    st.divider()
    st.caption("Startdatum (klipper datan för backtestet):")
    start_date = st.date_input("Startdatum", value=None, key="btv2_start_date")

    # Knappar
    b1, b2, b3 = st.columns(3)
    params_final = None
    df_override = None
    with b1:
        if st.button("Använd dessa parametrar", type="primary"):
            params_final = dict(ui) if ui else dict(pc)
            if start_date is not None:
                try:
                    from app.data_providers import get_ohlcv as GET_OHLCV
                    df_override = GET_OHLCV(ticker=ticker, start=str(start_date), end=None)
                except Exception:
                    df_override = None
            st.success("Parametrar applicerade för nästa körning.")

    with b2:
        if st.button("Återställ från profil"):
            st.experimental_rerun()

    with b3:
        if st.button("Spara till profil"):
            to_save = dict(ui) if ui else dict(pc)
            _save_profile_params(ticker, to_save)
            st.success(f"Sparat till profiles/{ticker}.json")

    with st.expander("Parametrar (rå JSON)", expanded=False):
        raw = dict(ui) if ui else dict(pc)
        st.code(json.dumps(raw, ensure_ascii=False, indent=2), language="json")

    return params_final, df_override
