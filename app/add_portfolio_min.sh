set -euo pipefail
cd /srv/trader/app

ts=$(date +%F_%H%M%S)
mkdir -p backups

target="pages/2_Portfolio_MIN.py"
[ -f "$target" ] && cp -v "$target" "backups/2_Portfolio_MIN.py.${ts}"

cat >"$target" <<'PY'
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Branding (om du har den) ---
try:
    from app.branding import apply as brand
except Exception:
    def brand(*a, **k): pass

brand(page_title="Dala Trader – Portfolio MIN", page_icon="🧺")
st.title("🧺 Portfolio MIN")
st.caption("Minimal portfölj-sida med förvaltningslogik: hämtar OHLCV → bygger buy&hold eller strategi-equity per ticker → ombalanserar.")

# --- Inläsare: priser och (ev.) strategi ---
try:
    from app.data_providers import get_ohlcv as GET_OHLCV
except Exception as e:
    GET_OHLCV = None
    st.error(f"Kunde inte importera dataprovider: {type(e).__name__}: {e}")

def _run_strategy_equity(ticker:str, start:str, end:str, params:dict|None=None) -> pd.Series:
    """Försök ta fram strategi-equity via btwrap.run_backtest, annars returnera tom serie."""
    try:
        from app.btwrap import run_backtest as RUNBT
    except Exception:
        return pd.Series(dtype="float64")

    p = {"ticker": ticker, "params": dict(params or {})}
    p["params"]["from_date"] = start
    p["params"]["to_date"]   = end

    try:
        out = RUNBT(p=p)
    except TypeError:
        # äldre signaturer
        try:
            out = RUNBT(ticker, p["params"])
        except Exception:
            return pd.Series(dtype="float64")
    except Exception:
        return pd.Series(dtype="float64")

    # Plocka equity ur olika returformat
    if isinstance(out, dict):
        eq = out.get("equity") or out.get("Equity")
        if isinstance(eq, pd.DataFrame):
            df = eq.copy()
            # normalisera kolumner
            low = {c:str(c).strip().lower() for c in df.columns}
            df = df.rename(columns=low)
            if "equity" in df.columns:
                s = pd.Series(df["equity"].values, index=pd.to_datetime(df.index)).sort_index()
                return s.astype(float)
            if "date" in df.columns and "equity" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"]))
                s = df["equity"].astype(float).sort_index()
                return s
        # ibland skickas list/np
        try:
            s = pd.Series(eq).astype(float)
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            return pd.Series(dtype="float64")

    # tuple: (summary, equity_df)
    if isinstance(out, tuple) and len(out)>=2 and hasattr(out[1], "iloc"):
        df = out[1]
        low = {c:str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=low)
        if "equity" in df.columns:
            s = pd.Series(df["equity"].values, index=pd.to_datetime(df.index)).sort_index()
            return s.astype(float)

    return pd.Series(dtype="float64")

def _buyhold_equity_from_close(df: pd.DataFrame) -> pd.Series:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype="float64")
    cols = {c:str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    # gissa close/price
    for c in ("close","adj close","adjclose","price","last"):
        if c in df.columns:
            s = pd.Series(pd.to_numeric(df[c], errors="coerce"), index=pd.to_datetime(df.index)).dropna().sort_index()
            if not s.empty and s.iloc[0] != 0:
                return (s / s.iloc[0]).rename("BH")
    # fallback om df är 1-kolumn
    if df.shape[1] == 1:
        s = pd.Series(pd.to_numeric(df.iloc[:,0], errors="coerce"), index=pd.to_datetime(df.index)).dropna().sort_index()
        if not s.empty and s.iloc[0] != 0:
            return (s / s.iloc[0]).rename("BH")
    return pd.Series(dtype="float64")

# --- UI ---
colA,colB,colC = st.columns([1.2,1,1])
with colA:
    tickers_raw = st.text_input("Tickers (komma-separerat)", value="GETI B, VOLV B, HM B").strip()
with colB:
    start = st.date_input("Från", value=date.today()-timedelta(days=365*5)).isoformat()
with colC:
    end   = st.date_input("Till", value=date.today()).isoformat()

colw1,colw2 = st.columns([1,1])
with colw1:
    weights_raw = st.text_input("Vikter (komma-separerat, valfritt – annars likavikt)", value="")
with colw2:
    rebalance = st.selectbox("Ombalansera", options=["Ingen (buy&hold vikter)","Månadsvis","Kvartalsvis","Årsvis","Daglig"], index=1)

use_strategy = st.checkbox("Använd strategi-equity (btwrap) om tillgänglig, annars buy&hold", value=True)

st.markdown("---")
go = st.button("🚀 Kör portfölj")

def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _rebalance_dates(idx: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    if mode == "Daglig": 
        return idx
    if mode == "Ingen (buy&hold vikter)":
        return pd.DatetimeIndex([idx[0]])
    if mode == "Månadsvis":
        return pd.to_datetime(pd.Series(idx).dt.to_period("M").drop_duplicates().dt.start_time).intersection(idx)
    if mode == "Kvartalsvis":
        return pd.to_datetime(pd.Series(idx).dt.to_period("Q").drop_duplicates().dt.start_time).intersection(idx)
    if mode == "Årsvis":
        return pd.to_datetime(pd.Series(idx).dt.to_period("Y").drop_duplicates().dt.start_time).intersection(idx)
    return pd.DatetimeIndex([idx[0]])

if go:
    if GET_OHLCV is None:
        st.error("Dataprovider saknas.")
        st.stop()

    tickers = _parse_list(tickers_raw)
    if not tickers:
        st.warning("Ange minst en ticker.")
        st.stop()

    # vikter
    if weights_raw.strip():
        w = [float(x.replace(",",".").strip()) for x in weights_raw.split(",") if x.strip()]
        if len(w) != len(tickers):
            st.error("Antalet vikter måste matcha antal tickers.")
            st.stop()
        w = np.array(w, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            w = w / w.sum()
    else:
        w = np.ones(len(tickers), dtype=float) / len(tickers)

    # Bygg en tabell med komponent-equity
    series = {}
    info_rows = []
    for t in tickers:
        # 1) hämta pris
        try:
            px = GET_OHLCV(t, start=start, end=end)
        except TypeError:
            px = GET_OHLCV(t, start, end)
        except Exception as e:
            st.warning(f"{t}: kunde inte hämta data: {type(e).__name__}: {e}")
            continue

        # 2) strategi eller buy&hold
        s = pd.Series(dtype="float64")
        if use_strategy:
            s = _run_strategy_equity(t, start, end, params={})
        if s.empty:
            s = _buyhold_equity_from_close(px)
        s.name = t
        if not s.empty:
            # normalisera till 1.0 i start
            s = s / (s.iloc[0] if s.iloc[0] else 1.0)
            series[t] = s
            info_rows.append({"Ticker": t, "N": int(s.shape[0])})
        else:
            info_rows.append({"Ticker": t, "N": 0})

    if not series:
        st.error("Ingen komponent fick giltig equity.")
        st.stop()

    df = pd.concat(series.values(), axis=1).dropna(how="all").sort_index()
    st.subheader("Komponent-equity (normaliserad)")
    st.dataframe(df.tail(10))

    # Avkastningar (dagliga)
    rets = df.pct_change().fillna(0.0)

    # Ombalanserad portfölj
    rb_dates = _rebalance_dates(rets.index, rebalance)
    weights = pd.DataFrame(index=rets.index, columns=df.columns, data=0.0)

    last_w = pd.Series(w, index=df.columns, dtype=float)
    for dt in rets.index:
        if dt in rb_dates:
            last_w = pd.Series(w, index=df.columns, dtype=float)  # sätt lika/angivna vikter
        weights.loc[dt] = last_w.values

    # portfölj-retur = summa(vikt * retur)
    port_ret = (weights * rets).sum(axis=1)
    port_eq  = (1.0 + port_ret).cumprod().rename("Portfolio")

    # Nyckeltal
    def _stats(s: pd.Series) -> dict:
        if s.empty: return {}
        r = s.pct_change().dropna()
        mean = r.mean()
        std  = r.std(ddof=0)
        cagr = (s.iloc[-1]) ** (252.0/len(s)) - 1.0 if len(s)>0 else np.nan
        sharpeD = (mean/std) * np.sqrt(252) if std and std==std and std>0 else np.nan
        roll = s.cummax()
        dd   = (s/roll - 1.0).min()
        return {"CAGR": cagr, "SharpeD": sharpeD, "MaxDD": dd}
    kp = _stats(port_eq)

    st.subheader("Portfölj – equity & nyckeltal")
    st.line_chart(port_eq)
    st.write({
        "CAGR": None if kp.get("CAGR") is None else round(float(kp["CAGR"]), 4),
        "SharpeD": None if kp.get("SharpeD") is None else round(float(kp["SharpeD"]), 3),
        "MaxDD": None if kp.get("MaxDD") is None else round(float(kp["MaxDD"]), 3),
        "Period": f"{start} → {end}",
        "Tickers": tickers,
        "Rebalance": rebalance,
    })

    st.markdown("### Komponentinfo")
    st.dataframe(pd.DataFrame(info_rows))
else:
    st.info("Fyll i tickers/period och klicka **Kör portfölj**.")
PY

# enkel syntaktisk koll (om venv finns kör den)
if [ -x .venv/bin/python ]; then
  .venv/bin/python -m py_compile "$target" || true
fi

echo "[ok] Portfolio MIN skapad: $target"
