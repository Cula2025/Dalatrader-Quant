from __future__ import annotations
import json, sys, glob, os

REQ_PARAM_KEYS = {
    "from_date","to_date","breakout_lookback","exit_lookback",
    "use_rsi_filter","rsi_window","rsi_min","rsi_max",
    "use_macd_filter","macd_fast","macd_slow","macd_signal",
    "use_trend_filter","trend_ma_type","trend_ma_window",
    "use_bb_filter","bb_window","bb_nstd","bb_min",
    "use_stop_loss","stop_mode","stop_loss_pct",
    "atr_window","atr_mult","use_atr_trailing","atr_trail_mult",
}

def check_file(path: str) -> bool:
    ok = True
    try:
        data = json.loads(open(path,"r",encoding="utf-8-sig").read())
    except Exception as e:
        print(f"[FAIL] {path}: ogiltig JSON: {e}")
        return False

    profs = data.get("profiles")
    if not isinstance(profs, list):
        print(f"[FAIL] {path}: saknar 'profiles' (lista).")
        return False

    if len(profs) != 3:
        print(f"[WARN] {path}: förväntade 3 profiler, fick {len(profs)}.")

    for i, p in enumerate(profs, 1):
        if not isinstance(p, dict):
            print(f"[FAIL] {path} p#{i}: inte dict.")
            ok = False; continue
        name   = p.get("name","")
        ticker = p.get("ticker","")
        params = p.get("params", {})
        if not isinstance(name, str) or not name.strip():
            print(f"[FAIL] {path} p#{i}: 'name' saknas/tom.")
            ok = False
        if not isinstance(ticker, str) or not ticker.strip():
            print(f"[FAIL] {path} p#{i}: 'ticker' saknas/tom.")
            ok = False
        if not isinstance(params, dict):
            print(f"[FAIL] {path} p#{i}: 'params' saknas/ej dict.")
            ok = False
        else:
            miss = [k for k in REQ_PARAM_KEYS if k not in params]
            if miss:
                print(f"[WARN] {path} p#{i}: params saknar {miss}")

        met = p.get("metrics", {})
        if not isinstance(met, dict):
            print(f"[WARN] {path} p#{i}: 'metrics' saknas/ej dict (valfritt, men bra att ha).")

    if ok:
        print(f"[OK]   {path}: {len(profs)} profiler.")
    return ok

def main(paths: list[str]) -> int:
    if not paths:
        paths = sorted(glob.glob("profiles/*.json"))
    if not paths:
        print("Inga profiler hittade i profiles/*.json")
        return 1
    rc = 0
    for p in paths:
        if not check_file(p):
            rc = 2
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
