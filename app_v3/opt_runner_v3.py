from __future__ import annotations
import os, json, math, random, time, multiprocessing as mp
from typing import Dict, Any, Callable, List, Tuple
import pandas as pd

from app_v3.data_provider_v3 import get_ohlcv
from app_v3.bt_core_v3 import run_backtest

# --- param-källa: försök återanvända din befintliga generator om den finns ---
def _get_param_sampler():
    try:
        # Om du har en egen param-generator i din nuvarande modul
        from app_v3.opt_core_v3 import make_params as _mp  # type: ignore
        return _mp
    except Exception:
        pass

    # Fallback: enkel parameter-yta som ändå brukar ge vettiga signaler
    def _fallback(rng: random.Random) -> Dict[str, Any]:
        fast = rng.randint(5, 25)
        slow = rng.randint(max(fast+5, 20), 120)
        return {
            "fast": fast,
            "slow": slow,
            # filters (slå av/på slumpat men rimligt)
            "use_rsi_filter": rng.random() < 0.5,
            "rsi_window": rng.randint(8, 20),
            "rsi_min": rng.uniform(30, 45),
            "rsi_max": rng.uniform(55, 70),
            "use_trend_filter": rng.random() < 0.5,
            "trend_ma_type": rng.choice(["SMA","EMA","WMA"]),
            "trend_ma_window": rng.randint(50, 120),
            "breakout_lookback": rng.randint(20, 80),
            "exit_lookback": rng.randint(10, 60),
            "use_macd_filter": rng.random() < 0.4,
            "macd_fast": rng.randint(8, 14),
            "macd_slow": rng.randint(18, 30),
            "macd_signal": rng.randint(8, 14),
            "use_bb_filter": rng.random() < 0.3,
            "bb_window": rng.randint(15, 30),
            "bb_nstd": rng.uniform(1.5, 2.5),
            "bb_min": rng.uniform(0.15, 0.35),
            "use_stop_loss": rng.random() < 0.35,
            "stop_mode": rng.choice(["pct","atr"]),
            "stop_loss_pct": rng.uniform(0.08, 0.2),
            "atr_window": rng.randint(10, 20),
            "atr_mult": rng.uniform(1.5, 3.2),
            "use_atr_trailing": rng.random() < 0.35,
            "atr_trail_mult": rng.uniform(1.5, 3.0),
        }
    return _fallback

def _score(metrics: Dict[str, Any]) -> Tuple[float,float,float]:
    # Sorteringsnyckel: SharpeD först, sedan CAGR, sedan TotalReturn
    sd = float(metrics.get("SharpeD", 0.0) or 0.0)
    cg = float(metrics.get("CAGR", 0.0) or 0.0)
    tr = float(metrics.get("TotalReturn", 0.0) or 0.0)
    return (sd, cg, tr)

def _profile_from_result(ticker: str, params: Dict[str, Any], metrics: Dict[str, Any], start: str, end: str|None) -> Dict[str, Any]:
    # Rensa numpy-typer
    clean_m = {k: (float(v) if isinstance(v, (int,float)) else (float(v) if hasattr(v, "__float__") else v))
               for k, v in metrics.items()}
    p = dict(params)
    p["from_date"] = start
    if end: p["to_date"] = end
    return {
        "name": f"{ticker} – candidate",
        "ticker": ticker,
        "params": p,
        "metrics": clean_m,
    }

def _eval_one(args) -> Dict[str, Any] | None:
    # Worker-process: får allt den behöver
    i, seed, base_params, df_bt, ticker, start, end = args
    rng = random.Random(seed + i)
    # merge base_params (om du vill låsa vissa), annars bara generatorn
    sampler = _get_param_sampler()
    p = sampler(rng)
    if base_params:
        p.update(base_params)
    # kör motor
    try:
        res = run_backtest(df_bt, p)
    except Exception:
        return None
    eq = pd.to_numeric(pd.Series(res.get("equity")), errors="coerce").dropna()
    if len(eq) < 2:
        return None
    m = dict(res.get("metrics") or {})
    if not m:
        # Beräkna ett minimisätt metrics om motorn inte lämnade något
        try:
            rets = eq.pct_change().dropna()
            sd = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets)>50 and rets.std()>0 else 0.0
            tr = float(eq.iloc[-1]/eq.iloc[0] - 1.0)
            # CAGR på samma indexfönster
            days = (eq.index[-1] - eq.index[0]).days or 1
            yrs = days/365.25
            cagr = float((eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1.0) if yrs>0 else tr
            m = {"SharpeD": sd, "TotalReturn": tr, "CAGR": cagr}
        except Exception:
            return None
    return _profile_from_result(ticker, p, m, start, end)

def optimize(ticker: str, sims: int, seed: int, start: str, end: str|None=None,
             on_tick: Callable[[int,int], None] | None = None,
             processes: int | None = None,
             base_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Kör sims st optimeringar, parallellt. Returnerar:
      {
        'profiles': [top3...],
        'profiles_path': 'profiles_v3/<TICKER>.json',
        'scoreboard': [top20...]
      }
    """
    # 1) Hämta pris EN gång
    df_bt = get_ohlcv(ticker, start=start, end=end).sort_index()
    if df_bt is None or df_bt.empty:
        raise RuntimeError("Ingen prisdata")

    # 2) Multiprocess
    n = int(max(1, sims))
    procs = int(processes or max(1, min(mp.cpu_count(), 4)))
    work = [(i+1, seed, base_params or {}, df_bt, ticker, start, end) for i in range(n)]

    results: List[Dict[str, Any]] = []
    done = 0

    def _tick():
        nonlocal done
        done += 1
        if on_tick:
            try: on_tick(done, n)
            except Exception: pass

    if procs == 1:
        for w in work:
            out = _eval_one(w)
            if out: results.append(out)
            _tick()
    else:
        with mp.get_context("spawn").Pool(processes=procs) as pool:
            for out in pool.imap_unordered(_eval_one, work, chunksize=max(1, n // (procs*4))):
                if out: results.append(out)
                _tick()

    if not results:
        return {"profiles": [], "profiles_path": None, "scoreboard": []}

    # 3) Sortera & välj toppar
    results.sort(key=lambda r: _score(r["metrics"]), reverse=True)
    top3 = results[:3]
    scoreboard = results[:min(20, len(results))]

    # 4) Spara JSON
    os.makedirs("profiles_v3", exist_ok=True)
    out_path = os.path.join("profiles_v3", f"{ticker}.json")
    out_obj = {"profiles": top3}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    return {"profiles": top3, "profiles_path": out_path, "scoreboard": scoreboard}
