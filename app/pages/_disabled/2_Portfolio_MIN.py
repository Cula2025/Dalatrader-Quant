from __future__ import annotations
import os, re, json, ast, inspect, traceback, datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd

# --- debug (failsafe) ---
try:
    from app.debuglog import setup_debug_ui, log_info, log_warn, log_error
except Exception:
    def setup_debug_ui(*a, **k): pass
    def log_info(*a, **k): pass
    def log_warn(*a, **k): pass
    def log_error(*a, **k): pass

st.set_page_config(page_title="Portfolio MIN", page_icon="📦", layout="wide")
st.title("📦 Portfolio MIN")
setup_debug_ui(st)

OUT_DIR = "trader/outputs"; os.makedirs(OUT_DIR, exist_ok=True)
TICKER_KEYS = ("Ticker","ticker","symbol","code","name","shortName","short_name")

# ---------- tolerant JSON-läsning ----------
def _strip_comments_and_trailing_commas(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

def _try_ast_literal_eval(text: str) -> Any:
    t = re.sub(r"\btrue\b","True", re.sub(r"\bfalse\b","False", re.sub(r"\bnull\b","None", text, flags=re.I), flags=re.I), flags=re.I)
    try: return ast.literal_eval(t)
    except Exception: return None

def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(d)
    for k in TICKER_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            d["Ticker"] = v.strip(); break
    for nest in ("params","config","settings","options"):
        if isinstance(d.get(nest), dict):
            for k,v in d[nest].items(): d.setdefault(k,v)
            del d[nest]
    return d

def _looks_single(d: Dict[str, Any]) -> bool:
    return any(k in d for k in TICKER_KEYS) or any(k in d for k in ("params","config","settings","options"))

def _infer_ticker_from_filename(path: str) -> Optional[str]:
    name = os.path.splitext(os.path.basename(path))[0]
    return (name.split("_")[0] if "_" in name else name).strip() or None

def _json_to_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [_flatten(x) for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("profiles","rows","items","results","data","list","values"):
            if isinstance(obj.get(k), list):
                return [_flatten(x) for x in obj[k] if isinstance(x, dict)]
        if _looks_single(obj): return [_flatten(obj)]
        if obj and all(isinstance(v, dict) for v in obj.values()):
            rows=[]
            for tck,payload in obj.items():
                r=_flatten(payload)
                if "Ticker" not in r and isinstance(tck,str) and tck.strip(): r["Ticker"]=tck.strip()
                rows.append(r)
            return rows
    return []

def parse_json_file(path: str) -> List[Dict[str, Any]]:
    try:
        text = open(path,"r",encoding="utf-8-sig").read().strip()
        if not text: return []
        # NDJSON?
        if "\n" in text and text.lstrip().startswith("{") and not text.strip().startswith("["):
            out=[]
            for ln in [ln.strip() for ln in text.splitlines() if ln.strip()]:
                try: out.append(_flatten(json.loads(ln))); continue
                except Exception: pass
                ln2=_strip_comments_and_trailing_commas(ln)
                try: out.append(_flatten(json.loads(ln2))); continue
                except Exception: pass
                obj=_try_ast_literal_eval(ln2)
                if isinstance(obj, dict): out.append(_flatten(obj))
            return out
        # vanlig JSON / tolerant
        try: obj=json.loads(_strip_comments_and_trailing_commas(text))
        except Exception: obj=_try_ast_literal_eval(_strip_comments_and_trailing_commas(text))
        return _json_to_rows(obj) if obj is not None else []
    except Exception as e:
        log_warn(f"Parse fail {path}: {e}")
        return []

@st.cache_data(ttl=120, show_spinner=False)
def list_json_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path): return []
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith((".json",".jsonl",".ndjson"))])

def build_df(dir_path: str, files: List[str], allow_filename_fallback: bool) -> pd.DataFrame:
    rows=[]
    for f in files:
        p=os.path.join(dir_path,f)
        recs=parse_json_file(p)
        if recs:
            for r in recs:
                r=dict(r); r["source_file"]=f
                if not r.get("Ticker"): r["Ticker"]=_infer_ticker_from_filename(p) or ""
                rows.append(r)
        else:
            tck=_infer_ticker_from_filename(p) if allow_filename_fallback else ""
            rows.append({"Ticker": tck or "", "source_file": f, "_note": "empty/parse-fail" if not tck else "from-filename"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df.columns=[str(c).replace("\ufeff","").strip() for c in df.columns]
        first=[c for c in ("Ticker","source_file") if c in df.columns]
        df=df[first + [c for c in df.columns if c not in first]]
    return df

# ---------- (valfri) batch-motor ----------
def _detect_engine() -> Tuple[Optional[Any], str]:
    cands=[("app.portfolio_engine","run_for_ticker"),
           ("app.portfolio_backtest","run_for_ticker"),
           ("app.backtest","run_backtest"),
           ("app.backtester","run_single")]
    for mod,attr in cands:
        try:
            m=__import__(mod, fromlist=[attr]); fn=getattr(m,attr,None)
            if callable(fn): return fn, f"{mod}.{attr}"
        except Exception: continue
    return None,""

def _call_engine(fn, ticker: str) -> Dict[str, Any]:
    try:
        sig=inspect.signature(fn); names={p.name for p in sig.parameters.values()}
        if "ticker" in names: res=fn(ticker=ticker)
        elif "symbol" in names: res=fn(symbol=ticker)
        elif "code" in names: res=fn(code=ticker)
        else: res=fn(ticker)  # positionellt
        return res if isinstance(res, dict) else {"Ticker": ticker, "result": res}
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# ---------- UI ----------
left,right = st.columns([3,1])
default_dir = st.session_state.get("portfolio_dir","/srv/trader/app/profiles")
with left:
    data_dir = st.text_input("Datakatalog (JSON)", value=default_dir, key="portfolio_dir")
with right:
    profiles="/srv/trader/app/profiles"
    if os.path.isdir(profiles) and st.button("Använd profiles/"): st.session_state["portfolio_dir"]=profiles; st.rerun()

files=list_json_files(data_dir)
if not files:
    st.warning(f"Inga JSON-filer i `{data_dir}`."); st.stop()

st.caption(f"Hittade {len(files)} fil(er) i `{data_dir}`")
picked = st.multiselect("Välj filer", files, default=files)
if not picked:
    st.info("Välj minst en fil."); st.stop()

allow_filename_fallback = st.checkbox("Tillåt Ticker från filnamn om JSON ej kan läsas", value=True)
df=build_df(data_dir, picked, allow_filename_fallback)
if df.empty or "Ticker" not in df.columns:
    st.error("Hittade inga profiler med Ticker."); st.stop()

c1,c2,c3=st.columns(3)
with c1: st.metric("Profiler", len(df))
with c2: st.metric("Unika tickers", df["Ticker"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
with c3: st.metric("Filer", len(set(df["source_file"].tolist())))

st.subheader("Profiler")
st.dataframe(df, width="stretch")

# Spara
ts=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv=os.path.join(OUT_DIR,f"portfolio_profiles_{ts}.csv")
out_json=os.path.join(OUT_DIR,f"portfolio_profiles_{ts}.json")
colA,colB=st.columns(2)
with colA:
    if st.button("💾 Spara CSV"): df.to_csv(out_csv, index=False, encoding="utf-8-sig"); st.success(f"Sparat: `{out_csv}`")
with colB:
    if st.button("💾 Spara JSON"):
        with open(out_json,"w",encoding="utf-8") as f: json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        st.success(f"Sparat: `{out_json}`")

# Valfri batch-körning
st.divider()
st.subheader("Batch-körning (valfritt)")
do_batch = st.checkbox("Aktivera batch-backtest för listade tickers", value=False)
if do_batch:
    ENGINE_FN, ENGINE_NAME = _detect_engine()
    st.caption(f"Motor: {'—' if ENGINE_FN is None else ENGINE_NAME}")
    workers = st.slider("Parallella jobb", 1, 6, 2)
    timeout_s = st.slider("Timeout per ticker (s)", 5, 120, 30)
    tickers = sorted(set([t for t in df["Ticker"].astype(str).str.strip().tolist() if t]))
    if st.button(f"Kör {len(tickers)} tickers"):
        import concurrent.futures as cf, time
        results=[]
        with st.spinner("Kör…"):
            if ENGINE_FN is None:
                st.error("Ingen motor hittad."); st.stop()
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs={ex.submit(_call_engine, ENGINE_FN, t): t for t in tickers}
                for fut in cf.as_completed(futs, timeout=timeout_s*len(tickers)+5):
                    t=futs[fut]
                    try: r=fut.result(timeout=timeout_s)
                    except Exception as e: r={"Ticker": t, "Error": f"Timeout/Exception: {e}"}
                    results.append(r)
        out=pd.DataFrame(results)
        st.dataframe(out, width="stretch")
        out_path=os.path.join(OUT_DIR, f"portfolio_result_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        st.success(f"Sparat: `{out_path}`")
