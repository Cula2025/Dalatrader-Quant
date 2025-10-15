from app.data_providers import get_ohlcv
from app.backtest import run_backtest
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns and hasattr(df.index,"inferred_type") and "date" in str(df.index.inferred_type):
        df = df.reset_index().rename(columns={"index":"Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # behåll bara giltiga rader
    return df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)

def get_summary(res):
    if isinstance(res, dict):
        return res.get("summary", {})
    if isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], dict):
        return res[0]
    return {}

def run_case(name, df, params):
    print(f"=== {name} ===")
    res = run_backtest(df.copy(), **params)
    s   = get_summary(res)
    print("Params:", params)
    print("Summary:", s, "\n")
    return s

# ----- Ladda data -----
df = clean(get_ohlcv("VOLV B", start="2020-01-01", end="2025-09-16", source="borsdata"))
print(f"Rows: {len(df)}  Period: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}\n")

# ----- Basparametrar (A) -----
A = dict(
    use_rsi_filter=True, rsi_window=14, rsi_min=25, rsi_max=60,
    breakout_lookback=55, exit_lookback=20,
    use_trend_filter=True, trend_ma_type="EMA", trend_ma_window=100,
    cost_bps=0.0, slip_bps=0.0
)

# B = bredare RSI
B = dict(A, rsi_min=15, rsi_max=80)

# C = alla filter av (borde alltid ge trades vid nån punkt)
C = dict(
    use_rsi_filter=True, rsi_window=14, rsi_min=15, rsi_max=80,
    breakout_lookback=0, exit_lookback=0,
    use_trend_filter=False,
    cost_bps=0.0, slip_bps=0.0
)

# D = stäng AV EMA-gate, behåll breakout/exit
D = dict(A, use_trend_filter=False)

# E = behåll EMA-gate, stäng AV breakout/exit
E = dict(A, breakout_lookback=0, exit_lookback=0)

# F = RSI-only, tight
F = dict(
    use_rsi_filter=True, rsi_window=14, rsi_min=25, rsi_max=60,
    breakout_lookback=0, exit_lookback=0,
    use_trend_filter=False,
    cost_bps=0.0, slip_bps=0.0
)
G = dict(
    use_rsi_filter=True, rsi_window=9, rsi_min=22, rsi_max=68,
    use_trend_filter=True, trend_ma_type="EMA", trend_ma_window=100,
    breakout_lookback=30, exit_lookback=15,
    breakout_mode="confirm", breakout_recent_window=10,
    cost_bps=0.0, slip_bps=0.0
)
print("=== G Tuned confirm breakout ===")
sG = run_case("G Tuned", df, G)

# ----- Kör alla fall -----
sA = run_case("A BASELINE", df, A)
sB = run_case("B Wider RSI", df, B)
sC = run_case("C All filters OFF", df, C)
sD = run_case("D No EMA gate", df, D)
sE = run_case("E No breakout/exit", df, E)
sF = run_case("F RSI-only", df, F)