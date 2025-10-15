from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Tuple
import json
from datetime import date, datetime

CFG = Path("config/portfolio_settings.json")

_DEFAULTS: Dict[str, Any] = {
  "portfolio": {
    "start_date_backtest": "2020-10-05",  # ISO (YYYY-MM-DD) eller "today"
    "start_date_live": "today"            # när du går live; används senare
  },
  "panic_rule": {
    "enabled": True,
    "index_ticker": "OMXS30GI",
    "trigger": "daily_change",   # "daily_change" | "drawdown" (förberett)
    "threshold": -0.05,          # -5% (<=0)
    "cooldown_days": 0
  },
  "selection": {
    "slots_per_day": 2,
    "weights": {
      "total_return": 0.5,
      "win_rate": 0.3,
      "recent_momentum": 0.2,
      "breakout_z": 0.35,
      "trend_over_atr": 0.25,
      "rel_strength_20d": 0.2,
      "freshness": 0.1,
      "liquidity": 0.1,
      "corr_penalty": 0.2,
      "sector_penalty": 0.1
    },
    "filters": {
      "min_days_since_last_exit": 0,
      "exclude": []
    },
    "caps": {
      "max_pct_per_asset": 0.33,   # 33% per aktie (övre gräns)
      "correlation_cap": 0.85,
      "sector_cap": 0.4,
      "allow_pyramiding": False,
      "backlog_days": 3
    },
    "tiebreaker": "liquidity_then_cap",
    "mode": "simple"
  },
  "leverage": {
    "allow": False,
    "default": 1.0   # 1.0 | 2.0 | 4.0 (om instrumentet stödjer)
  }
}

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _to_date(dval: Any) -> date | None:
    if dval is None:
        return None
    if isinstance(dval, date):
        return dval
    if isinstance(dval, datetime):
        return dval.date()
    if isinstance(dval, str):
        if dval.strip().lower() == "today":
            return date.today()
        try:
            return datetime.fromisoformat(dval.strip()).date()
        except Exception:
            return None
    return None

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _validate_and_normalize(d: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    """Returnerar (sanerad_settings, warnings). Normaliserar weights till sum=1 om sum>0."""
    out = deepcopy(_DEFAULTS)
    warn: list[str] = []

    def pick(path: list[str], default: Any) -> Any:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # portfolio dates
    pb = out["portfolio"]
    sdb = pick(["portfolio","start_date_backtest"], pb["start_date_backtest"])
    sdl = pick(["portfolio","start_date_live"], pb["start_date_live"])
    pb["start_date_backtest"] = sdb if isinstance(sdb, str) else "today"
    pb["start_date_live"]     = sdl if isinstance(sdl, str) else "today"
    if _to_date(pb["start_date_backtest"]) is None:
        warn.append("Ogiltigt backtest-startdatum, satte 'today'.")
        pb["start_date_backtest"] = "today"
    if _to_date(pb["start_date_live"]) is None:
        warn.append("Ogiltigt live-startdatum, satte 'today'.")
        pb["start_date_live"] = "today"

    # panic rule
    prd = out["panic_rule"]
    src = pick(["panic_rule"], {})
    prd["enabled"]       = bool(src.get("enabled", prd["enabled"]))
    prd["index_ticker"]  = str(src.get("index_ticker", prd["index_ticker"]))
    trig                 = str(src.get("trigger", prd["trigger"]))
    prd["trigger"]       = trig if trig in ("daily_change","drawdown") else "daily_change"
    thr                  = float(src.get("threshold", prd["threshold"]))
    prd["threshold"]     = _clamp(thr, -0.2, 0.0)
    prd["cooldown_days"] = max(0, int(src.get("cooldown_days", prd["cooldown_days"])))

    # selection
    sel = out["selection"]
    s_src = pick(["selection"], {})
    sel["slots_per_day"] = max(0, int(s_src.get("slots_per_day", sel["slots_per_day"])))

    # weights → normalisera
    w_def = sel["weights"]
    w_src = s_src.get("weights", {})
    w: Dict[str, float] = {}
    for k, v in w_def.items():
        try:
            w[k] = float(w_src.get(k, v))
        except Exception:
            w[k] = v
    ssum = sum(max(0.0, x) for x in w.values())
    if ssum <= 0:
        warn.append("Alla weights var 0 – behåller defaults.")
        w = deepcopy(w_def)
        ssum = sum(w.values())
    w = {k: (max(0.0, v)/ssum) for k, v in w.items()}
    sel["weights"] = w

    # filters
    flt_def = sel["filters"]
    flt_src = s_src.get("filters", {})
    mdsle = max(0, int(flt_src.get("min_days_since_last_exit", flt_def["min_days_since_last_exit"])))
    excl  = flt_src.get("exclude", flt_def["exclude"])
    if not isinstance(excl, list): excl = flt_def["exclude"]
    sel["filters"] = {"min_days_since_last_exit": mdsle, "exclude": excl}

    # caps
    c_def = sel["caps"]
    c_src = s_src.get("caps", {})
    sel["caps"] = {
        "max_pct_per_asset": _clamp(float(c_src.get("max_pct_per_asset", c_def["max_pct_per_asset"])), 0.01, 1.0),
        "correlation_cap":   _clamp(float(c_src.get("correlation_cap",   c_def["correlation_cap"])),   0.0, 1.0),
        "sector_cap":        _clamp(float(c_src.get("sector_cap",        c_def["sector_cap"])),        0.05, 1.0),
        "allow_pyramiding":  bool(c_src.get("allow_pyramiding", c_def["allow_pyramiding"])),
        "backlog_days":      max(0, int(c_src.get("backlog_days", c_def["backlog_days"]))),
    }

    # tiebreaker/mode
    tb = str(s_src.get("tiebreaker", sel["tiebreaker"]))
    sel["tiebreaker"] = tb if tb in ("liquidity_then_cap",) else "liquidity_then_cap"
    md = str(s_src.get("mode", sel["mode"]))
    sel["mode"] = md if md in ("simple","advanced") else "simple"

    # leverage
    lev = out["leverage"]
    l_src = d.get("leverage", {})
    lev["allow"]   = bool(l_src.get("allow", lev["allow"]))
    try:
        lv = float(l_src.get("default", lev["default"]))
    except Exception:
        lv = lev["default"]
    if lv not in (1.0, 2.0, 4.0): lv = 1.0
    lev["default"] = lv

    return out, warn

def get() -> Dict[str, Any]:
    if CFG.exists():
        try:
            return json.loads(CFG.read_text(encoding="utf-8"))
        except Exception:
            # fall back till defaults om filen är trasig
            return deepcopy(_DEFAULTS)
    return deepcopy(_DEFAULTS)

def save(new_settings: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    clean, warn = _validate_and_normalize(new_settings)
    _atomic_write_json(CFG, clean)
    return clean, warn

def reset() -> Dict[str, Any]:
    clean, _ = _validate_and_normalize(_DEFAULTS)
    _atomic_write_json(CFG, clean)
    return clean

class _SettingsFacade:
    path = CFG
    def get(self) -> Dict[str, Any]: return get()
    def save(self, d: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]: return save(d)
    def reset(self) -> Dict[str, Any]: return reset()

SETTINGS = _SettingsFacade()
