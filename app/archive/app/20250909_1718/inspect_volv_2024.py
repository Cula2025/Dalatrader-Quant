# app/inspect_volv_2024.py
import os, numpy as np, pandas as pd
from app.bd_legacy_client import BDClient

START = "2024-01-10"

c = BDClient()
rows = c.prices_by_ticker("VOLV B", days=5000)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("Inga rader från Börsdata för VOLV B")

df["Date"] = pd.to_datetime(df["d"])
df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
df = df[["Date","Open","High","Low","Close","Volume"]].sort_values("Date")
df = df[df["Date"] >= pd.to_datetime(START)].copy()

os.makedirs("./outputs", exist_ok=True)
outfp = os.path.abspath("./outputs/volv_b_since_2024-01-10.csv")
df.to_csv(outfp, index=False)

# Snabb diagnostik
ret = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1) if len(df)>1 else float("nan")
dr = df.set_index("Date")["Close"].pct_change().dropna()
vol = float(dr.std()*np.sqrt(252)) if len(dr)>2 else float("nan")
zeros = int((df["Volume"]<=0).sum())
gap_up = int((df["Open"] > df["Close"].shift(1)*1.03).sum())
gap_dn = int((df["Open"] < df["Close"].shift(1)*0.97).sum())

print(f"Saved: {outfp}  rows={len(df)}  {df['Date'].min().date()}..{df['Date'].max().date()}")
print(f"Total return: {ret:.2%}   Ann.vol: {vol:.2%}   Zero-volume days: {zeros}")
print(f"Gap up >3%: {gap_up}   Gap down >3%: {gap_dn}")
print("Last 5 rows:")
print(df.tail(5).to_string(index=False))
