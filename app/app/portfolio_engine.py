from __future__ import annotations
from typing import Dict, Iterable, Tuple
import pandas as pd
from app.data_providers import get_ohlcv

def _to_ts(x): return pd.to_datetime(x).normalize()

def signals_from_trades_df(trades: pd.DataFrame, ticker: str, profile: str) -> pd.DataFrame:
    df = trades.copy()
    cols = set(df.columns)
    # Fall A: backtest-resumé med Entry/Exit-kolumner
    if {"EntryTime","EntryPrice","ExitTime","ExitPrice"}.issubset(cols):
        rows=[]
        for _,r in df.iterrows():
            rows.append({"date":_to_ts(r["EntryTime"]), "ticker":ticker, "side":"SELL" if pd.isna(r["EntryPrice"]) else "BUY",  "price":float(r["EntryPrice"])})
            rows.append({"date":_to_ts(r["ExitTime"]),  "ticker":ticker, "side":"SELL", "price":float(r["ExitPrice"])})
        sig = pd.DataFrame(rows)
    # Fall B: redan BUY/SELL-ledger
    elif {"side","price"}.issubset(cols):
        sig = df.reset_index().rename(columns={"index":"date"})
        if "date" not in sig.columns: sig["date"] = pd.to_datetime(sig.pop("time"), errors="coerce")
        if "ticker" not in sig.columns: sig["ticker"] = ticker
        sig = sig[["date","ticker","side","price"]]
        sig["date"] = sig["date"].apply(_to_ts)
    else:
        raise ValueError("Okänt trades-format (saknar Entry/Exit eller side/price).")
    sig["profile"] = profile
    sig = sig.dropna(subset=["date","side","price"]).sort_values(["date","side","ticker"])
    # SÄKERSTÄLL att SELL ligger före BUY samma dag (så kassa frigörs)
    sig["side_sort"] = sig["side"].str.upper().map({"SELL":0,"BUY":1}).fillna(1)
    sig = sig.sort_values(["date","side_sort","ticker"]).drop(columns=["side_sort"])
    return sig

def fetch_prices(tickers: Iterable[str], start: str) -> pd.DataFrame:
    frames=[]
    for t in tickers:
        df = get_ohlcv(ticker=t, start=start, end=None)[["Close"]].rename(columns={"Close":t}).dropna()
        df.index = pd.to_datetime(df.index).normalize()
        frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1, join="outer").sort_index().ffill()

def simulate_constant_fraction(signals: pd.DataFrame, prices: pd.DataFrame,
                               start_cash: float=100_000.0, fraction: float=1/3) -> Tuple[pd.Series, pd.DataFrame]:
    if prices.empty: return pd.Series(dtype="float64", name="Portfolio"), pd.DataFrame()
    dates = prices.index

    # Mappa signal-datum till nästa handelsdag i prices
    def next_day(ts):
        i = dates.searchsorted(ts, side="left")
        return pd.NaT if i>=len(dates) else dates[i]
    sig = signals.copy()
    sig["date"] = sig["date"].apply(next_day)
    sig = sig.dropna(subset=["date"]).sort_values(["date","ticker","side"])

    holdings: Dict[str,float] = {}
    cash = float(start_cash)
    ledger_rows=[]
    equity=[]

    by_date = {d:g for d,g in sig.groupby("date")}
    for d in dates:
        row_prices = prices.loc[d]

        # 1) SÄLJ först (frigör kassa)
        if d in by_date:
            sells = by_date[d][by_date[d]["side"].str.upper()=="SELL"]
            for _,r in sells.iterrows():
                t = r["ticker"]; px = float(r["price"])
                qty = holdings.get(t, 0.0)
                if qty>0:
                    val = qty*px
                    cash += val
                    ledger_rows.append({"date":d, "ticker":t, "side":"SELL", "price":px, "qty":qty, "value":val})
                    holdings[t]=0.0

        # 2) KÖP sedan, dela kassan enligt 1/3-regeln (skala ned om kassa inte räcker)
        if d in by_date:
            buys = by_date[d][by_date[d]["side"].str.upper()=="BUY"]
            if len(buys):
                invested = sum(holdings.get(t,0.0)*float(row_prices.get(t, float("nan"))) for t in prices.columns)
                E = cash + invested
                target_each = fraction * E
                total_needed = target_each * len(buys)
                scale = min(1.0, cash/total_needed) if total_needed>0 else 0.0
                for _,r in buys.iterrows():
                    t = r["ticker"]; px = float(r["price"])
                    if px<=0: continue
                    alloc = target_each * scale
                    if alloc<=0: continue
                    qty = alloc/px  # fractional shares för enkelhet
                    cash -= alloc
                    holdings[t] = holdings.get(t,0.0) + qty
                    ledger_rows.append({"date":d, "ticker":t, "side":"BUY", "price":px, "qty":qty, "value":alloc})

        # 3) Mark-to-market (daglig portföljvärdering)
        invested = sum(holdings.get(t,0.0)*float(row_prices.get(t, float("nan"))) for t in prices.columns)
        E = cash + invested
        equity.append((d, E))

    eq = pd.Series([v for _,v in equity], index=[d for d,_ in equity], name="Portfolio")
    ledger = pd.DataFrame(ledger_rows).set_index("date").sort_index()
    return eq, ledger

def buyhold_equal_weight(prices: pd.DataFrame, start_cash: float=100_000.0) -> pd.Series:
    if prices.empty: return pd.Series(dtype="float64", name="Buy&Hold")
    cols = list(prices.columns)
    start = prices.iloc[0]
    alloc = start_cash/len(cols)
    qty = {t: alloc/max(1e-12, float(start[t])) for t in cols}
    vals = (prices * pd.Series(qty)).sum(axis=1)
    vals.name = "Buy&Hold"
    return vals
