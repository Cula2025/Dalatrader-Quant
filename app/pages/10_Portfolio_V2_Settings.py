from __future__ import annotations
import json
from pathlib import Path
from datetime import date
import streamlit as st
import pandas as pd

CONFIG = Path("config/portfolio_settings.json")

WEIGHT_FIELDS = [
    ("total_return",      "Total avkastning"),
    ("win_rate",          "Träffsäkerhet"),
    ("recent_momentum",   "Senaste momentum"),
    ("breakout_z",        "Breakout-Z"),
    ("trend_over_atr",    "Trend / ATR"),
    ("rel_strength_20d",  "Relativ styrka (20d)"),
    ("freshness",         "Färskhet"),
    ("liquidity",         "Likviditet"),
    ("corr_penalty",      "Korrelation (straff)"),
    ("sector_penalty",    "Sektor (straff)"),
]

DEFAULTS = {
    "selector": {
        "slots_per_day": 2,
        "weights": {
            "total_return":     0.50,
            "win_rate":         0.30,
            "recent_momentum":  0.20,
            "breakout_z":       0.35,
            "trend_over_atr":   0.25,
            "rel_strength_20d": 0.20,
            "freshness":        0.10,
            "liquidity":        0.10,
            "corr_penalty":     0.20,
            "sector_penalty":   0.10,
        },
        "filters": {
            "min_days_since_last_exit": 0,
            "exclude": [],
        },
        "tiebreaker": "liquidity_then_cap",
        "mode": "simple",
    },
    "caps": {
        "correlation_cap": 0.85,     # 85% (lagras som 0.85)
        "sector_cap": 0.40,          # 40% (lagras som 0.40)
        "allow_pyramiding": False,
        "backlog_days": 3,
    },
    "panic_rule": {
        "enabled": True,
        "index_ticker": "OMXS30GI",
        "trigger": "daily_change",   # (förberett: "drawdown")
        "threshold": -0.05,          # -5%  → sälja allt
        "cooloff_days": 1,           # vilodagar efter panic
        "reentry_relief": -0.03,     # extra marginal innan återinträde
    },
    "portfolio": {
        "start_date": "2020-10-05",
    },
}

def _deep_merge(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

def load_cfg() -> dict:
    data = {}
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        pass
    out = json.loads(json.dumps(DEFAULTS))  # deep copy
    _deep_merge(out, data)

    # se till att weight-nycklar finns
    w = out.setdefault("selector", {}).setdefault("weights", {})
    for k, _ in WEIGHT_FIELDS:
        w.setdefault(k, DEFAULTS["selector"]["weights"][k])

    # skydd för nya block
    out.setdefault("caps", DEFAULTS["caps"].copy())
    out.setdefault("panic_rule", DEFAULTS["panic_rule"].copy())
    out.setdefault("portfolio", DEFAULTS["portfolio"].copy())
    return out

def save_cfg(cfg: dict):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

def normalize_weights_inplace(cfg: dict):
    w = (cfg.get("selector") or {}).get("weights") or {}
    s = float(sum(w.values())) if w else 0.0
    if s > 0:
        cfg["selector"]["weights"] = {k: float(v)/s for k, v in w.items()}

# -------------------- UI-block --------------------

def render_weights(cfg: dict) -> dict:
    st.subheader("Signalkraft (Weights)")
    st.caption("Dra i reglagen. Summan normaliseras automatiskt när du sparar.")
    w = (cfg.get("selector") or {}).get("weights") or {}
    w = {k: float(w.get(k, DEFAULTS["selector"]["weights"][k])) for k, _ in WEIGHT_FIELDS}

    cols = st.columns(3)
    for i, (key, label) in enumerate(WEIGHT_FIELDS):
        with cols[i % 3]:
            w[key] = st.slider(label, 0.0, 1.0, w[key], 0.01, key=f"w_{key}")

    raw_sum = sum(w.values())
    st.write(f"**Summa (rå): {raw_sum:.2f}** – normaliseras till 100% vid **Spara**.")
    if raw_sum > 0:
        norm = {k: v/raw_sum for k, v in w.items()}
        df_prev = (
            pd.Series(norm).mul(100).round(1).rename("Vikt %")
            .sort_values(ascending=False).to_frame()
        )
        st.dataframe(df_prev, use_container_width=True, height=260)
    else:
        st.info("Välj några vikter > 0 så visas förhandsvisningen här.")

    cfg.setdefault("selector", {})["weights"] = w
    return cfg

def render_selector(cfg: dict) -> dict:
    st.subheader("Urval – slots per dag")
    sel = cfg.setdefault("selector", {})
    slots = int(sel.get("slots_per_day", DEFAULTS["selector"]["slots_per_day"]))
    sel["slots_per_day"] = st.number_input("Max nya köp / dag", 0, 20, slots, 1, key="slots_per_day")
    return cfg

def render_caps(cfg: dict) -> dict:
    st.subheader("Caps & begränsningar")
    caps = cfg.setdefault("caps", {}).copy()

    col1, col2 = st.columns(2)
    with col1:
        cap_corr_pct = int(round(100 * float(caps.get("correlation_cap", DEFAULTS["caps"]["correlation_cap"]))))
        cap_corr_pct = st.slider("Korrelationstak (%)", 0, 100, cap_corr_pct, 1, key="cap_corr")
        caps["correlation_cap"] = cap_corr_pct / 100.0
    with col2:
        cap_sector_pct = int(round(100 * float(caps.get("sector_cap", DEFAULTS["caps"]["sector_cap"]))))
        cap_sector_pct = st.slider("Sektortak (%)", 0, 100, cap_sector_pct, 1, key="cap_sector")
        caps["sector_cap"] = cap_sector_pct / 100.0

    col3, col4 = st.columns(2)
    with col3:
        caps["allow_pyramiding"] = st.toggle(
            "Tillåt pyramidisering (öka i vinnare)", bool(caps.get("allow_pyramiding", False)), key="cap_pyr"
        )
    with col4:
        caps["backlog_days"] = st.number_input(
            "Backlog-dagar (köer som får rulla)", 0, 30, int(caps.get("backlog_days", 3)), 1, key="cap_backlog"
        )

    cfg["caps"] = caps
    return cfg

def render_panic_rule(cfg: dict) -> dict:
    st.subheader("Panic rule (sälj allt)")
    pr = cfg.setdefault("panic_rule", {}).copy()

    pr["enabled"] = st.toggle("Aktiverad", bool(pr.get("enabled", True)), key="pr_on")
    pr["index_ticker"] = st.text_input("Index-ticker", pr.get("index_ticker", "OMXS30GI"), key="pr_tkr")

    pr["trigger"] = st.selectbox(
        "Trigger",
        options=["daily_change"],  # förberett: "drawdown"
        index=0 if pr.get("trigger","daily_change") == "daily_change" else 0,
        key="pr_trig",
    )

    # procentslidrar som lagras som fraktion (negativa)
    thr_pct = int(round(100 * float(pr.get("threshold", -0.05))))
    thr_pct = st.slider("Tröskel (% daglig förändring)", -15, 0, thr_pct, 1, help="Ex. -5% ⇒ sälj allt.", key="pr_thr")
    pr["threshold"] = thr_pct / 100.0

    pr["cooloff_days"] = st.number_input("Vilodagar efter panic", 0, 10, int(pr.get("cooloff_days", 1)), 1, key="pr_cool")

    relief_pct = int(round(100 * float(pr.get("reentry_relief", -0.03))))
    relief_pct = st.slider("Återinträdes-marginal (%)", -10, 0, relief_pct, 1,
                           help="Extra buffert innan nya köp tillåts", key="pr_relief")
    pr["reentry_relief"] = relief_pct / 100.0

    cfg["panic_rule"] = pr
    return cfg

def render_portfolio(cfg: dict) -> dict:
    st.subheader("Portfölj – startdatum")
    sd = (cfg.setdefault("portfolio", {})).get("start_date")
    try:
        dflt = date.fromisoformat(sd) if sd else date.today()
    except Exception:
        dflt = date.today()
    new = st.date_input("Starta portföljen från", value=dflt, key="p_start_date")
    cfg["portfolio"]["start_date"] = new.isoformat()
    st.caption("Används av backtest/portföljsidor för att klippa historik och möjliggöra ny start.")
    return cfg

# -------------------- MAIN --------------------

def main():
    st.title("Portfolio V2 – Settings")
    st.markdown("<style>div.stSlider label{font-weight:600}</style>", unsafe_allow_html=True)

    cfg = load_cfg()

    with st.expander("Signalkraft (Weights)", expanded=True):
        cfg = render_weights(cfg)

    colA,colB = st.columns([1,1])
    with colA:
        with st.expander("Urval (Slots per day)", expanded=True):
            cfg = render_selector(cfg)
    with colB:
        with st.expander("Caps", expanded=True):
            cfg = render_caps(cfg)

    with st.expander("Panic rule (sälj allt)", expanded=False):
        cfg = render_panic_rule(cfg)

    with st.expander("Portfölj – startdatum", expanded=False):
        cfg = render_portfolio(cfg)

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Spara", type="primary"):
            normalize_weights_inplace(cfg)
            save_cfg(cfg)
            st.success("Inställningar sparade.")
    with c2:
        if st.button("Reset till standard"):
            save_cfg(DEFAULTS)
            st.warning("Återställd till standard.")
    with c3:
        if st.button("Ångra ändringar"):
            st.info("Läser om från fil.")
            st.experimental_rerun()

    with st.expander("Rå JSON", expanded=False):
        st.code(json.dumps(cfg, indent=2, ensure_ascii=False), language="json")

if __name__ == "__main__":
    main()
