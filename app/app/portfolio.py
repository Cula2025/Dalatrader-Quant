import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def _ensure_close_col(df: pd.DataFrame) -> pd.Series:
    for c in ("Close","close","Adj Close","adj_close","adj_close"):
        if c in df.columns:
            return df[c].astype(float)
    # If single-column series was passed
    if isinstance(df, pd.Series):
        return df.astype(float)
    # fallback: try first numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[0]].astype(float)
    raise ValueError("Could not find a price column (e.g., Close)")

def to_price_matrix(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dict[ticker -> OHLCV df] -> DataFrame of Close prices (aligned, ffilled)."""
    close = {}
    for t, df in ohlcv.items():
        s = _ensure_close_col(df).rename(t)
        close[t] = s
    P = pd.concat(close.values(), axis=1)
    P = P.sort_index().ffill().dropna(how="all")
    return P

def simulate_constant_mix(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    start_cash: float = 1.0,
    rebalance: str = "M",           # "M" monthly, "W" weekly, "N" never
    fee_bps: float = 0.0,           # round-trip cost in basis points applied on rebalances
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Vector-lite simulation with optional periodic rebalancing."""
    w = pd.Series(weights, dtype=float)
    w = w[w != 0.0]
    w = (w / w.abs().sum()).reindex(prices.columns).fillna(0.0)

    # Restrict to assets in weights and drop columns w=0
    P = prices[w.index[w.values != 0.0]]
    if P.empty:
        raise ValueError("No overlapping tickers between prices and weights.")

    # Returns matrix
    R = P.pct_change().fillna(0.0)

    # Rebalance mask
    if rebalance == "M":
        rebalance_days = R.index.to_series().dt.to_period("M").ne(
            R.index.to_series().shift(1).dt.to_period("M")
        )
    elif rebalance == "W":
        rebalance_days = R.index.to_series().dt.to_period("W").ne(
            R.index.to_series().shift(1).dt.to_period("W")
        )
    else:
        rebalance_days = pd.Series(False, index=R.index)
        rebalance_days.iloc[0] = True  # set initial alloc

    # Simulate
    V = start_cash
    value = []
    cur_w = w.values.copy()
    fee = fee_bps / 1e4

    for i, dt in enumerate(R.index):
        if i == 0 or rebalance_days.iloc[i]:
            # trade to target weights -> trading cost proportional to turnover
            if i > 0 and fee > 0:
                # turnover = sum |target - current| / 2 ; apply fee on traded notional
                turnover = np.abs(cur_w - w.values).sum() / 2.0
                V *= (1.0 - fee * turnover)
            cur_w = w.values.copy()
        # portfolio return for the day is w_t- times asset returns
        port_ret = float(np.dot(cur_w, R.iloc[i].values))
        V *= (1.0 + port_ret)
        # weights drift
        if i+1 < len(R):
            drift = (1.0 + R.iloc[i].values)
            cur_w = cur_w * drift
            s = cur_w.sum()
            if s != 0:
                cur_w = cur_w / s
        value.append(V)

    equity = pd.DataFrame(
        {"value": pd.Series(value, index=R.index)}
    )
    equity["ret"] = equity["value"].pct_change().fillna(0.0)

    # Metrics
    n = len(equity)
    if n > 1:
        cagr = equity["value"].iloc[-1] ** (252.0 / n) - 1.0
        vol = equity["ret"].std() * math.sqrt(252.0)
    else:
        cagr = 0.0; vol = 0.0
    sharpe = (cagr / vol) if vol > 0 else np.nan
    roll_max = equity["value"].cummax()
    dd = equity["value"] / roll_max - 1.0
    max_dd = dd.min()
    metrics = {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "MaxDD": float(max_dd),
        "Last": float(equity["value"].iloc[-1]) if n else float(start_cash),
    }
    equity["drawdown"] = dd
    return equity, metrics
