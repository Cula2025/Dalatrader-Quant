import json
from pathlib import Path
import pandas as pd
import streamlit as st


# --- NO-CHART MONKEYPATCH BEGIN ---
try:
    import streamlit as st
except Exception:
    pass
else:
    def __nochart(*args, **kwargs):
        # grafer temporärt avstängda under felsökning
        return None
    try:
        st.line_chart      = __nochart
        st.area_chart      = __nochart
        st.bar_chart       = __nochart
        st.altair_chart    = __nochart
        st.pyplot          = __nochart
        st.plotly_chart    = __nochart
        st.vega_lite_chart = __nochart
        st.caption("🔧 Chart-rendering avstängd temporärt.")
    except Exception:
        pass
# --- NO-CHART MONKEYPATCH END ---

from app import btwrap as W
from app.trade_extract import to_trades_df

from app.portfolio_math import pick_first
st.set_page_config(page_title="Portfolio V2 – Trades", layout="wide")
st.title("📒 Portfolio V2 – Trades")

# Välj profiler
all_files = sorted([str(p) for p in Path("profiles").glob("*.json")])
sel_files = st.multiselect("Profilfiler (profiles/*.json)", all_files, default=all_files)

if not sel_files:
    st.info("Välj minst en profilfil för att visa trades.")
    st.stop()

frames: list[pd.DataFrame] = []

def _is_df(x) -> bool:
    return isinstance(x, pd.DataFrame)

def _non_empty_df(x) -> bool:
    return _is_df(x) and not x.empty

for f in sel_files:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        p = (d.get("profiles") or [])[0] if isinstance(d, dict) else {}
    except Exception as e:
        st.warning(f"Kan inte läsa {f}: {e}")
        continue

    ticker = (
        (p.get("ticker") if isinstance(p, dict) else None)
        or (p.get("params", {}).get("ticker") if isinstance(p, dict) else None)
        or Path(f).stem
    )

    try:
        params = dict(p.get("params") or {}) if isinstance(p, dict) else {}
        res = W.run_backtest(p={"ticker": ticker, "params": params})
        raw = res.get("trades") if isinstance(res, dict) else res
        df  = to_trades_df(raw)
    except Exception as e:
        st.warning(f"to_trades_df-fel för {ticker} ({f}): {e}")
        continue

    if not _non_empty_df(df):
        continue

    # Normalisering
    df = df.copy()

    # Säkerställ datumindex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            # Saknar datum → hoppa över
            continue

    # Sätt ticker, ta bort profile
    df["ticker"] = str(ticker)
    if "profile" in df.columns:
        df.drop(columns=["profile"], inplace=True)

    # Notional: value eller qty*price
    if "notional" not in df.columns:
        if "value" in df.columns:
            df["notional"] = df["value"].abs()
        else:
            df["notional"] = (df.get("qty", 0) * df["price"]).abs()

    # Begränsa kolumner och appenda
    keep = [c for c in ["side","price","qty","notional","pnl","ticker"] if c in df.columns]
    if keep:
        frames.append(df[keep])

if not frames:
    st.info("Inga trades att visa ännu.")
    st.stop()

tdf = pd.concat(frames, axis=0).sort_index()

# Datumfilter i sidopanelen
start_default = tdf.index.min().date()
end_default   = tdf.index.max().date()
date_range = st.sidebar.date_input(
    "Datumintervall",
    (start_default, end_default)
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
    if d0 is not None and d1 is not None:
        m = (tdf.index >= pd.Timestamp(d0)) & (tdf.index <= pd.Timestamp(d1))
        tdf = tdf.loc[m]

# KPI: BUY/SELL, omsättning, realiserad PnL (SELL)
buy_n  = int((tdf["side"] == "BUY").sum())  if "side" in tdf.columns else 0
sell_n = int((tdf["side"] == "SELL").sum()) if "side" in tdf.columns else 0
turnover = float(tdf.get("notional", pd.Series(dtype="float64")).sum())
realized = float(tdf.loc[tdf["side"] == "SELL", "pnl"].sum()) if "side" in tdf.columns and "pnl" in tdf.columns else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("BUY", f"{buy_n}")
c2.metric("SELL", f"{sell_n}")
c3.metric("Omsättning (sum notional)", f"{int(round(turnover)):,.0f} kr".replace(",", " "))
c4.metric("Realiserad PnL (SELL-rader)", f"{int(round(realized)):,.0f} kr".replace(",", " "))

st.caption(f"Rader: {len(tdf):,} | Period: {tdf.index.min().date()} → {tdf.index.max().date()}".replace(",", " "))

# Visa tabell
cols = [c for c in ["date","ticker","side","price","qty","notional","pnl"] if c in (["date"] + list(tdf.columns))]
out = tdf.copy()
out["date"] = out.index
out = out[cols]
st.dataframe(out, use_container_width=True)

# Nedladdning
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Ladda ner CSV", data=csv, file_name="portfolio_trades.csv", mime="text/csv")
