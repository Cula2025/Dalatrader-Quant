import numpy as np
import pandas as pd

def total_return_pct(initial_equity: float, final_equity: float) -> float:
    """Total avkastning i procent från initial till final equity."""
    if initial_equity <= 0:
        return 0.0
    return (final_equity / initial_equity - 1.0) * 100.0

def cagr_pct(initial_equity: float, final_equity: float, years: float) -> float:
    """CAGR i procent givet antal år."""
    if initial_equity <= 0 or years <= 0:
        return 0.0
    return ( (final_equity / initial_equity) ** (1.0 / years) - 1.0 ) * 100.0

def max_drawdown_pct(equity_curve: pd.Series | list | np.ndarray) -> float:
    """Maximal nedgång i procent från equity‐kurvan."""
    eq = pd.Series(equity_curve).astype(float)
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    dd = (eq / running_max - 1.0) * 100.0
    return float(dd.min())  # negativt tal, ex -26.8
