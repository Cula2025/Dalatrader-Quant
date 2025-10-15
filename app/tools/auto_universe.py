from __future__ import annotations
from pathlib import Path
import json, math

def best_per_file():
    out=[]
    for f in sorted(Path("profiles").glob("*.json")):
        try:
            d=json.loads(f.read_text(encoding="utf-8"))
            profs=d.get("profiles") or []
            if not profs: continue
            best=max(
                profs,
                key=lambda p: float((p.get("metrics") or {}).get("TotalReturn") or -math.inf)
            )
            if not best.get("ticker"):
                t=(best.get("params") or {}).get("ticker")
                if t: best["ticker"]=t
            best["_source"]=f.name
            out.append(best)
        except Exception:
            pass
    return out

if __name__=="__main__":
    picks=best_per_file()
    payload={"profiles":picks}
    Path("profiles/_auto_universe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Valda profiler:")
    for p in picks:
        tr=(p.get("metrics") or {}).get("TotalReturn")
        print(f" - {p.get('ticker')} | {p.get('name')} | TR={tr:.6f} | src={p.get('_source')}")
