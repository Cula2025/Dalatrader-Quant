import sys, os
sys.path.insert(0, r"C:\trader")

from app.env_bootstrap import load_env
load_env()

print("BORS_API_KEY:", os.getenv("BORS_API_KEY"))
print("BORSDATA_KEY:", os.getenv("BORSDATA_KEY"))

from app.bd_modern_client import get_ohlcv

for sym in ["VOLV B", "VOLV-B", "ABB", "ERIC B"]:
    try:
        df = get_ohlcv(sym, "2019-01-01", None)
        print(sym, "->", len(df), "rader")
        print(df.head(2))
    except Exception as e:
        print(sym, "ERROR:", e)
