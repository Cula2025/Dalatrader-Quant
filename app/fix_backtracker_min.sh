#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/srv/trader/app"
PAGES_DIR="$APP_DIR/pages"
TARGET="$PAGES_DIR/1_Backtest.py"
VENV_PY="$APP_DIR/.venv/bin/python"

echo "== Backtest MIN installer =="
cd "$APP_DIR"

# 1) Backup ev. befintlig fil
mkdir -p "$PAGES_DIR" backups
if [ -f "$TARGET" ]; then
  cp -v "$TARGET" "backups/1_Backtest.py.bak_$(date +%F_%H%M%S)"
fi

# 2) Skriv minimal, robust Backtest MIN
cat > "$TARGET" <<'PY'
import streamlit as st
import pandas as pd

# Samma provider som Optimizer
from app.data_providers import get_ohlcv as GET_OHLCV
# Kör motorn via stabil signatur: run_backtest(df=<DataFrame>, p=<dict>)
from backtest import run_backtest as RUN_BT

st.set_page_config(page_title="Backtest MIN", page_icon="🧪", layout="wide")
st.title("🧪 Backtest MIN")
st.caption("Minimal ren version som använder samma datakälla som Optimizer.")

# --- Inputs (unika keys) ---
s = st.session_state
colA, colB, colC = st.columns(3)

with colA:
    ticker = st.text_input("Ticker", value=s.get("ticker", ""), key="min_ticker")
with colB:
    from_date = st.text_input("Från (YYYY-MM-DD)", value=s.get("from_date", "2020-10-01"), key="min_from")
with colC:
    to_date = st.text_input("Till (YYYY-MM-DD)", value=s.get("to_date", "2025-10-01"), key="min_to")

st.markdown("#### Parametrar (enkla defaults – kan läsas från profil senare)")
pcol1, pcol2, pcol3, pcol4 = st.columns(4)
with pcol1:
    breakout = st.number_input("Breakout lookback", 5, 300, 55, key="min_breakout")
with pcol2:
    exit_lb = st.number_input("Exit lookback", 2, 200, 20, key="min_exitlb")
with pcol3:
    rsi_w = st.number_input("RSI-fönster", 2, 100, 14, key="min_rsiw")
with pcol4:
    macd_f = st.number_input("MACD fast", 2, 50, 12, key="min_macdf")

params = {
    "use_rsi_filter": True,
    "rsi_window": int(rsi_w),
    "rsi_min": 30.0,
    "rsi_max": 70.0,
    "use_trend_filter": False,
    "trend_ma_type": "SMA",
    "trend_ma_window": 200,
    "breakout_lookback": int(breakout),
    "exit_lookback": int(exit_lb),
    "use_macd_filter": True,
    "macd_fast": int(macd_f),
    "macd_slow": 26,
    "macd_signal": 9,
    "use_bb_filter": False,
    "bb_window": 20,
    "bb_nstd": 2.0,
    "bb_min": 0.0,
    "use_stop_loss": False,
    "stop_mode": "pct",
    "stop_loss_pct": 0.10,
    "atr_window": 14,
    "atr_mult": 2.0,
    "use_atr_trailing": False,
    "atr_trail_mult": 2.0,
    "from_date": from_date,
    "to_date": to_date,
}

run = st.button("🚀 Kör backtest", type="primary")

st.markdown("---")
placeholder = st.container()

if run:
    if not ticker.strip():
        st.error("Ange en ticker.")
    else:
        with st.spinner("Hämtar data…"):
            try:
                df = GET_OHLCV(ticker.strip(), start=from_date, end=to_date)
            except TypeError:
                df = GET_OHLCV(ticker.strip(), from_date, to_date)
            except Exception as e:
                st.error(f"Kunde inte hämta data: {type(e).__name__}: {e}")
                df = None

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.warning("Ingen data hittades för vald period/ticker.")
        else:
            with st.spinner("Kör motor…"):
                try:
                    out = RUN_BT(df=df, p=params)
                except TypeError:
                    out = RUN_BT(df, params)
                except Exception as e:
                    st.error(f"Körning misslyckades: {type(e).__name__}: {e}")
                    out = None

            if not out or not isinstance(out, dict):
                st.error("Inget resultat att visa (tomt svar från motor).")
            else:
                summ = out.get("summary", {})
                eq = out.get("equity")
                trades = out.get("trades")

                st.subheader("Resultat")
                st.json(summ or {"info": "tom summary"})

                if isinstance(eq, pd.DataFrame) and not eq.empty:
                    st.markdown("**Equity (sista 10 rader)**")
                    st.dataframe(eq.tail(10), width='stretch')
                else:
                    st.info("Ingen equity-tabell i svaret.")

                if isinstance(trades, pd.DataFrame) and not trades.empty:
                    with st.expander("Trades (förhandsvisning)"):
                        st.dataframe(trades.head(20), width='stretch')
PY

# 3) Syntax-kontroll (venv om den finns, annars systemets python)
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" -m py_compile "$TARGET"
else
  python3 -m py_compile "$TARGET"
fi

# 4) Starta om Streamlit-tjänsten
if command -v sudo >/dev/null 2>&1; then
  sudo systemctl restart trader-ui.service || true
else
  systemctl restart trader-ui.service || true
fi

echo "[done] Backtest MIN installerad. Ladda om sidan och testa."
