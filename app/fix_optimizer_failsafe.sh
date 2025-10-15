#!/usr/bin/env bash
set -euo pipefail
cd /srv/trader/app

PAGE="pages/0_Optimizer.py"
MARK="# === DalaTrader: FAILSAFE OPT (begin) ==="

# 0) Backup
ts=$(date +%F_%H%M%S)
mkdir -p backups
cp -v "$PAGE" "backups/0_Optimizer.py.${ts}"

# 1) Inject failsafe block only if not present
if ! grep -qF "$MARK" "$PAGE"; then
  cat >>"$PAGE" <<'PYBLOCK'

# === DalaTrader: FAILSAFE OPT (begin) ===
# Minimal, self-contained optimizer block (no graphs) to verify engine + sampling loop.
# Appends at end to avoid clashing with any "from __future__" or earlier imports.

try:
    import streamlit as _st
    import random as _random, json as _json, os as _os
    from datetime import datetime as _dt
    try:
        from app.btwrap import run_backtest as _RUNBT
    except Exception as _e:
        _RUNBT = None
except Exception as _e:
    # If even Streamlit import fails here, just quietly exit this block
    _RUNBT = None

def __dt_draw_params(_rng):
    ur = lambda a,b: float(a + (b-a)*_rng.random())
    return {
        "use_rsi_filter": True,
        "rsi_window": int(_rng.randint(5,35)),
        "rsi_min": ur(5.0, 35.0),
        "rsi_max": ur(60.0, 85.0),

        "use_trend_filter": bool(_rng.choice([True, False])),
        "trend_ma_type": _rng.choice(["SMA", "EMA"]),
        "trend_ma_window": int(_rng.randint(20, 200)),

        "breakout_lookback": int(_rng.randint(20, 120)),
        "exit_lookback":     int(_rng.randint(10, 60)),

        "use_macd_filter": bool(_rng.choice([True, False])),
        "macd_fast":   int(_rng.randint(8, 16)),
        "macd_slow":   int(_rng.randint(18, 30)),
        "macd_signal": int(_rng.randint(8, 14)),

        "use_bb_filter": bool(_rng.choice([True, False])),
        "bb_window": int(_rng.randint(15, 30)),
        "bb_nstd":   ur(1.6, 2.4),
        "bb_min":    ur(0.0, 0.8),

        "use_stop_loss": bool(_rng.choice([True, False])),
        "stop_mode": _rng.choice(["pct", "atr"]),
        "stop_loss_pct": ur(0.03, 0.20),

        "atr_window": int(_rng.randint(10, 20)),
        "atr_mult":   ur(1.2, 3.2),

        "use_atr_trailing": bool(_rng.choice([True, False])),
        "atr_trail_mult":   ur(1.2, 3.5),
    }

def __dt_score(_m):
    # higher is better; mild penalties for drawdown, reward Sharpe
    tr  = float(_m.get("TotalReturn") or 0.0)
    dd  = abs(float(_m.get("MaxDD") or 0.0))
    shp = float(_m.get("SharpeD") or 0.0)
    return tr - 0.30*dd + 0.30*shp

def __dt_labelled(best3):
    # Return list of (label, params, metrics) for conservative/balanced/aggressive
    if not best3:
        return []
    # sort by score desc already; we derive 3 styles:
    # conservative: best Sharpe / low DD among top
    cons = sorted(best3, key=lambda x: (-float(x[2].get("SharpeD") or 0.0), abs(float(x[2].get("MaxDD") or 1.0))))[0]
    # aggressive: highest TotalReturn
    aggr = sorted(best3, key=lambda x: -(float(x[2].get("TotalReturn") or 0.0)))[0]
    # balanced: top score (first)
    bal  = best3[0]
    # de-duplicate while preserving order conservative, balanced, aggressive
    seen = set()
    out = []
    for name, item in [("conservative", cons), ("balanced", bal), ("aggressive", aggr)]:
        key = _json.dumps(item[1], sort_keys=True)
        if key in seen:  # if duplicate, pick next best unique
            for alt in best3:
                k2 = _json.dumps(alt[1], sort_keys=True)
                if k2 not in seen:
                    out.append((name, alt[1], alt[2]))
                    seen.add(k2)
                    break
        else:
            out.append((name, item[1], item[2]))
            seen.add(key)
    return out

try:
    _st.markdown("---")
    _st.subheader("⚙️ Failsafe-optimering (minimal)")

    # Always prefer existing session ticker; fallback to input only if missing.
    _sess = _st.session_state
    _tick = (_sess.get("ticker") or "").strip()
    if not _tick:
        _tick = _st.text_input("Ticker", key="__fs_ticker__", placeholder="t.ex. GETI B").strip()

    _fd = (_sess.get("from_date") or "").strip()
    _td = (_sess.get("to_date")   or "").strip()
    # simple inputs (names are unique → no key collisions)
    _fd = _st.text_input("Från (YYYY-MM-DD)", value=_fd, key="__fs_from__").strip()
    _td = _st.text_input("Till (YYYY-MM-DD)", value=_td, key="__fs_to__").strip()

    _sims = _st.number_input("Antal simuleringar", min_value=50, max_value=20000, value=1000, step=50, key="__fs_sims__")
    _seed = _st.number_input("Slumpfrö (seed)",   min_value=0,  max_value=10_000_000, value=42, step=1, key="__fs_seed__")

    if _st.button("Kör optimering (failsafe)", key="__fs_run_btn__", width='stretch'):
        if not _tick:
            _st.error("Ange en ticker först.")
        elif _RUNBT is None:
            _st.error("Backtest-funktion saknas (_RUNBT is None).")
        else:
            _rng = _random.Random(int(_seed))
            _best = []  # list of (score, params, metrics)
            for i in range(int(_sims)):
                p = __dt_draw_params(_rng)
                if _fd: p["from_date"] = _fd
                if _td: p["to_date"]   = _td
                try:
                    res = _RUNBT(p={"ticker": _tick, "params": p})
                    m = res.get("summary", {}) or {}
                    sc = __dt_score(m)
                    _best.append((sc, p, m))
                except Exception as e:
                    # skip bad draws
                    continue
            if not _best:
                _st.error("Inga giltiga körningar. Justera intervall eller minska begränsningar.")
            else:
                _best.sort(key=lambda x: x[0], reverse=True)
                _top3 = _best[:15]  # candidate pool
                labelled = __dt_labelled(_top3)  # [(name, params, metrics), ...]
                _st.success(f"Klar. Bästa score: { _best[0][0]:.3f }")
                # Save as three profiles
                _os.makedirs("profiles", exist_ok=True)
                outp = f"profiles/{_tick}.json"
                profs = []
                for lbl, p, m in labelled:
                    p = dict(p)  # ensure dates included
                    if _fd: p["from_date"] = _fd
                    if _td: p["to_date"]   = _td
                    profs.append({
                        "name": f"{_tick} – {lbl}",
                        "ticker": _tick,
                        "params": p,
                        "metrics": m,
                    })
                with open(outp, "w", encoding="utf-8") as f:
                    _json.dump({"profiles": profs}, f, ensure_ascii=False, indent=2)
                _st.write("Sparad profilfil:", outp)
                # show quick peek
                if profs:
                    _st.json(profs[0]["metrics"])
except Exception as _e:
    # Do not crash page if something is off
    try:
        _st.warning(f"Failsafe-optimering: {_e.__class__.__name__}: {_e}")
    except Exception:
        pass
# === DalaTrader: FAILSAFE OPT (end) ===

PYBLOCK
  echo "[patch] Failsafe-optimizer block appended to $PAGE"
else
  echo "[info] Failsafe-optimizer block already present. No changes."
fi

# 2) Syntax check (venv if present)
if [ -f .venv/bin/python ]; then
  . .venv/bin/activate
  python -m py_compile "$PAGE"
else
  python3 -m py_compile "$PAGE"
fi

# 3) Restart service
sudo systemctl restart trader-ui.service || true
echo "[done] Öppna Optimizer-sidan och klicka 'Kör optimering (failsafe)'."

