# ==== IMPORT SHIM (run_backtest) =========================================
try:
    # Försök legacy-modulen om den finns lokalt
    from backtest import run_backtest as RUN_BT
except Exception:
    # Fallback till projektmodulen
    from app.btwrap import run_backtest as RUN_BT
# ==========================================================================

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

# === Motor & datakälla — samma som Optimizer ===
from app.data_providers import get_ohlcv as GET_OHLCV

from app.equity_extract import extract_equity
from app.trade_extract import to_trades_df

try:
# legacy/local name
except Exception:
    from app.btwrap import run_backtest as RUN_BT  # project module

# ==== IMPORTS (helpers) ==================================================
try:
    from app.data_providers import get_ohlcv as GET_OHLCV
except Exception:
    GET_OHLCV = None

try:
    from app.equity_extract import extract_equity
except Exception:
    def extract_equity(x): return x  # no-op fallback

try:
    from app.trade_extract import to_trades_df

try:
    from backtest import run_backtest as RUN_BT    # legacy/local
except Exception:
    from app.btwrap import run_backtest as RUN_BT   # project module
# ==========================================================================


except Exception:
    def to_trades_df(x): return x     # no-op fallback
# ==========================================================================
# ==========================================================================

st.set_page_config(page_title="Backtracker V2 (Beta)", layout="wide")
st.title("🧪 Backtracker V2 (BETA) – motor: backtest.run_backtest")

# ---------------------------------------------------------
# Hjälpare
# ---------------------------------------------------------
def normalize_profiles_dict(d: dict) -> dict:
    d = dict(d or {})
    profs = d.get("profiles")
    if not isinstance(profs, list):
        p = {}
        if "ticker" in d: p["ticker"] = d["ticker"]
        if "params" in d: p["params"] = d["params"]
        d["profiles"] = [p] if p else []
    d["profiles"] = [p for p in d["profiles"] if isinstance(p, dict)]
    return d

def pretty_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)

def parse_json_or_err(txt: str) -> tuple[dict|None, str|None]:
    try:
        return json.loads(txt) if txt.strip() else {}, None
    except Exception as e:
        return None, str(e)

def make_backup(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = Path("backup") / f"{path.name}.{pd.Timestamp.now():%Y-%m-%d_%H%M%S}"
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup
    except Exception:
        return None

def _to_dt(s: pd.Series) -> pd.Series:
    if s is None or not len(s):
        return pd.Series(dtype="float64")
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()]
    return s

# -------------------- PARAM-FORM --------------------------
# Heuristik: gruppera och välj input-typ per nyckel
GROUP_ORDER = ["breakout", "exit", "atr", "bb", "rsi", "risk", "misc"]

def group_of(key: str) -> str:
    k = key.lower()
    if k.startswith("breakout"): return "breakout"
    if k.startswith("exit"):     return "exit"
    if k.startswith("atr"):      return "atr"
    if k.startswith("bb"):       return "bb"
    if k.startswith("rsi"):      return "rsi"
    if "risk" in k:              return "risk"
    return "misc"

def render_param_field(key: str, val):
    k = key.lower()
    help_txt = None

    # Bool → checkbox
    if isinstance(val, bool):
        return st.checkbox(key, value=val)

    # Int-heuristik
    if any(x in k for x in ["window", "lookback", "period", "bars"]):
        v = int(val) if isinstance(val, (int, float)) else 20
        return st.number_input(key, min_value=1, max_value=2000, step=1, value=v)

    # Multiplikatorer
    if "mult" in k or "coef" in k:
        v = float(val) if isinstance(val, (int,float)) else 1.0
        return st.number_input(key, min_value=0.0, max_value=20.0, step=0.05, value=float(v), format="%.4f")

    # Bollinger
    if k.startswith("bb_"):
        if "nstd" in k:
            v = float(val) if isinstance(val, (int,float)) else 2.0
            return st.number_input(key, min_value=0.0, max_value=6.0, step=0.1, value=float(v), format="%.2f")
        if "window" in k:
            v = int(val) if isinstance(val, (int,float)) else 20
            return st.number_input(key, min_value=2, max_value=400, step=1, value=int(v))
        if "min" in k:
            v = float(val) if isinstance(val, (int,float)) else 0.05
            return st.number_input(key, min_value=0.0, max_value=1.0, step=0.01, value=float(v), format="%.2f")

    # ATR
    if k.startswith("atr_") or k == "atr":
        # multipliers och window hanteras ovan; övriga som float
        v = float(val) if isinstance(val,(int,float)) else 1.0
        return st.number_input(key, min_value=0.0, max_value=50.0, step=0.05, value=float(v), format="%.4f")

    # RSI
    if k.startswith("rsi"):
        if "window" in k or "period" in k:
            v = int(val) if isinstance(val,(int,float)) else 14
            return st.number_input(key, min_value=2, max_value=200, step=1, value=int(v))
        else:
            # typ nivåer 0..100
            v = float(val) if isinstance(val,(int,float)) else 50.0
            return st.number_input(key, min_value=0.0, max_value=100.0, step=0.5, value=float(v), format="%.1f")

    # Fallback: numeric → number_input, annars text
    if isinstance(val, (int,float)):
        v = float(val)
        return st.number_input(key, value=v, step=0.1, format="%.6g")
    # list/dict → litet JSON-fält
    if isinstance(val, (list, dict)):
        raw = st.text_area(key, value=pretty_json(val), height=120)
        parsed, err = parse_json_or_err(raw)
        if err:
            st.warning(f"{key}: ogiltig JSON, behåller ursprungsvärde.")
            return val
        return parsed
    # str → text_input
    return st.text_input(key, value=str(val))

def render_params_form(params: dict) -> dict:
    """Bygg ett trevligt UI och returnera redigerad dict."""
    params = dict(params or {})
    if not params:
        st.info("Den här profilen saknar parametrar.")
        return {}

    # sortera nycklar och gruppera
    keys = sorted(params.keys())
    groups = {g: [] for g in GROUP_ORDER}
    for k in keys:
        g = group_of(k)
        groups.setdefault(g, []).append(k)

    edited = dict(params)
    st.subheader("Parametrar")
    tab_labels = [g.capitalize() for g in GROUP_ORDER if groups.get(g)]
    tabs = st.tabs(tab_labels or ["Params"])

    tab_map = {lab.lower(): t for lab, t in zip([g for g in GROUP_ORDER if groups.get(g)], tabs)}

    # rendera per grupp
    for g in GROUP_ORDER:
        ks = groups.get(g)
        if not ks: 
            continue
        with tab_map[g]:
            cols = st.columns(2, vertical_alignment="center")
            half = (len(ks)+1)//2
            left_keys, right_keys = ks[:half], ks[half:]
            with cols[0]:
                for k in left_keys:
                    edited[k] = render_param_field(k, edited.get(k))
            with cols[1]:
                for k in right_keys:
                    edited[k] = render_param_field(k, edited.get(k))

    with st.expander("Avancerat: rå JSON (synkas med formuläret)"):
        raw = st.text_area("",
                           value=pretty_json(edited),
                           height=220,
                           label_visibility="collapsed")
        parsed, err = parse_json_or_err(raw)
        if err:
            st.warning(f"Ogiltig JSON i rutan ovan: {err} — formuläret gäller tills JSON är korrekt.")
        else:
            edited = parsed or {}

    return edited

# ---------------------------------------------------------
# UI – Källa & profilval
# ---------------------------------------------------------
files = sorted(Path("profiles").glob("*.json"))
left, right = st.columns([2,1], vertical_alignment="bottom")
with left:
    file_opt = st.selectbox("Profilfil", options=[str(p) for p in files], index=0 if files else None, placeholder="profiles/*.json")
with right:
    st.caption("🛈 Välj fil → profil → redigera params → Spara → Kör.")

ticker: str | None = None
params: dict = {}
profiles: list[dict] = []
selected_idx = 0

if file_opt:
    pf = Path(file_opt)
    try:
        raw = json.loads(pf.read_text(encoding="utf-8"))
        D = normalize_profiles_dict(raw)
        profiles = D.get("profiles", [])
        labels = []
        for i,p in enumerate(profiles):
            t = p.get("ticker") or (p.get("params") or {}).get("ticker") or "?"
            name = p.get("name") or f"profil {i+1}"
            labels.append(f"{i+1}: {t} — {name}")
        if not labels:
            labels = ["(tom)"]
        selected_idx = st.selectbox("Välj profil i filen", options=range(len(labels)), format_func=lambda i: labels[i] if i < len(labels) else "(tom)")
        current = profiles[selected_idx] if profiles else {}
        ticker = current.get("ticker") or (current.get("params") or {}).get("ticker")
        params = dict(current.get("params") or {})
    except Exception as e:
        st.error(f"Kunde inte läsa {file_opt}: {e}")

st.divider()

# ---------------------------------------------------------
# Redigera profilens metadata & params
# ---------------------------------------------------------
meta_col, edit_col = st.columns([1,2])

with meta_col:
    st.subheader("Profil")
    prof_name = st.text_input("Namn (valfritt)", value=(profiles[selected_idx].get("name") if profiles else "") or "")
    ticker = st.text_input("Ticker", value=ticker or "")

with edit_col:
    # 🔧 Här: användarvänligt formulär i stället för rå JSON
    params = render_params_form(params)

# ---------------------------------------------------------
# Spara tillbaka i filen
# ---------------------------------------------------------
save_col1, save_col2, save_col3 = st.columns([1,1,2])
with save_col1:
    btn_save = st.button("💾 Spara till fil", type="primary", disabled=not file_opt)
with save_col2:
    btn_add = st.button("➕ Spara som ny profil", disabled=not file_opt)

if (btn_save or btn_add) and file_opt:
    pf = Path(file_opt)
    try:
        raw = json.loads(pf.read_text(encoding="utf-8"))
        D = normalize_profiles_dict(raw)
        P = D.get("profiles", [])
        new_prof = {"ticker": ticker, "name": prof_name, "params": params}

        make_backup(pf)  # best-effort

        if btn_save and P:
            P[selected_idx] = new_prof
        else:
            P.append(new_prof)

        D["profiles"] = P
        pf.write_text(pretty_json(D) + "\n", encoding="utf-8")
        st.success(f"Sparat till {pf.name} ({'uppdaterad profil' if btn_save else 'ny profil tillagd'}).")
    except Exception as e:
        st.error(f"Kunde inte spara: {e}")

st.divider()

# ---------------------------------------------------------
# Datumval + Kör
# ---------------------------------------------------------
start_date = st.date_input("Startdatum (frivilligt)", value=None)
end_date   = st.date_input("Slutdatum (frivilligt)", value=None)
start_str = str(start_date) if start_date else None
end_str   = str(end_date) if end_date else None

# Lägg även i params om strategin använder dem
if start_str: params["from_date"] = start_str
if end_str:   params["to_date"]   = end_str

run = st.button("▶︎ Kör backtest", type="primary", use_container_width=True, disabled=not ticker)

if run:
    if not ticker:
        st.error("Ange en ticker.")
        st.stop()

    try:
        df = GET_OHLCV(ticker=ticker, start=start_str, end=end_str)
        if df is None or df.empty:
            st.error("Inga prisdata hämtades för den här kombinationen av ticker/datum.")
            st.stop()

        res = RUN_BT(df=df, p={"ticker": ticker, "params": params})

        eq_src = res["equity"] if isinstance(res, dict) and "equity" in res else res
        tr_src = res["trades"] if isinstance(res, dict) and "trades" in res else res

        equity = _to_dt(extract_equity(eq_src))
        trades = to_trades_df(tr_src)
        if len(trades) and "ticker" not in trades.columns:
            trades = trades.copy()
            trades["ticker"] = str(ticker)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Equity (sista)", f"{float(equity.iloc[-1]) if len(equity) else float('nan'):.2f}")
        with m2: st.metric("Observations", f"{len(equity)}")
        with m3: st.metric("Trades", f"{len(trades)}")

        st.subheader("Equity (tabell)")
        st.dataframe(pd.DataFrame({"equity": equity}).tail(30), use_container_width=True, height=300)
        st.download_button("⬇️ equity.csv", data=pd.DataFrame({"equity": equity}).to_csv(index=True).encode("utf-8"),
                           file_name="equity.csv", mime="text/csv")

        st.subheader("Trades (tabell)")
        st.dataframe(trades.tail(50), use_container_width=True, height=350)
        st.download_button("⬇️ trades.csv", data=trades.to_csv(index=True).encode("utf-8"),
                           file_name="trades.csv", mime="text/csv")

    except Exception as e:
        st.exception(e)
        st.stop()
else:
    st.info("Välj fil, profil och redigera i formuläret. Spara om du vill. Kör sedan ▶︎.")
