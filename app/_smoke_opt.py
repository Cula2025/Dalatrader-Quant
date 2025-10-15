from app.data_providers import get_ohlcv
from app.backtest import run_backtest
import pandas as pd, datetime as dt

def norm(df):
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index":"Date"})
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    for c in ("Open","High","Low","Close","Volume"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna().sort_values("Date").reset_index(drop=True)

df = norm(get_ohlcv("VOLV B", start="2020/01/01", end=dt.date.today().strftime("%Y/%m/%d"), source="borsdata"))
params = dict(strategy="rsi", use_rsi_filter=True, rsi_window=14, rsi_min=25, rsi_max=60,
              breakout_lookback=55, exit_lookback=20, use_trend_filter=False, trend_ma_type="EMA", trend_ma_window=0)
res = run_backtest(df.copy(), **params)
# unify
def adapt(res):
    if isinstance(res, dict): return res
    if isinstance(res,(tuple,list)):
        parts=list(res)+[None,None,None,None]
        return {"summary":parts[0] or {}, "trades":parts[1], "equity_buy":parts[2], "equity_keep":parts[3]}
    return {"summary":{}, "trades":None}
out = adapt(res)
sm = out.get("summary",{})
print("Bars:",sm.get("Bars"),"Trades:",sm.get("Trades"),"TR:",sm.get("TotalReturn"))