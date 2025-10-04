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

def build_profiles_df(dir_path: str, files: List[str]) -> pd.DataFrame:
    rows=[]
    for f in files:
        p=os.path.join(dir_path,f)
        recs=parse_json_file(p)
        if recs:
            for r in recs:
                rr=dict(r); rr["source_file"]=f
                if not rr.get("Ticker"): rr["Ticker"]=_infer_ticker_from_filename(p) or ""
                rows.append(rr)
        else:
            rows.append({"Ticker": _infer_ticker_from_filename(p) or "", "source_file": f, "_note": "empty/parse-fail"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df.columns=[str(c).replace("\ufeff","").strip() for c in df.columns]
        first=[c for c in ("Ticker","source_file") if c in df.columns]
        df=df[first + [c for c in df.columns if c not in first]]
    return df

# -------------------- Motor / kurvor --------------------
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

def _series_from_engine_result(res: Any) -> Optional[pd.Series]:
    if res is None: return None
    s=None
    if isinstance(res, dict):
        candidates=["equity","curve","balance","cumret","portfolio","portfolio_value","equity_curve"]
        for k in candidates:
            v=res.get(k)
            if v is None: continue
            if isinstance(v, dict):
                try: s=pd.Series(v); break
                except Exception: pass
            if isinstance(v, list) and v:
                try:
                    if isinstance(v[0], dict):
                        key_date=None; key_val=None
                        for kd in ("date","Date","ts","timestamp"):
                            if kd in v[0]: key_date=kd; break
                        for kv in ("equity","value","val","y","close","Close"):
                            if kv in v[0]: key_val=kv; break
                        if key_date and key_val:
                            idx=pd.to_datetime([x[key_date] for x in v])
                            vals=[x[key_val] for x in v]
                            s=pd.Series(vals, index=idx); break
                    else:
                        s=pd.Series(v); break
                except Exception: pass
    if s is None and isinstance(res, pd.Series): s=res
    if s is None and isinstance(res, pd.DataFrame):
        for c in ("equity","value","val","close","Close"):
            if c in res.columns:
                try: s=res[c]; break
                except Exception: pass
    if s is None: return None
    s=s.dropna()
    if len(s)>0 and s.iloc[0]>0: s = s / float(s.iloc[0])
    try: s.index = pd.to_datetime(s.index)
    except Exception: pass
    return s.sort_index()

def _call_engine_curve(fn, ticker: str) -> Optional[pd.Series]:
    try:
        sig=inspect.signature(fn); names={p.name for p in sig.parameters.values()}
        if "ticker" in names: res=fn(ticker=ticker)
        elif "symbol" in names: res=fn(symbol=ticker)
        elif "code" in names: res=fn(code=ticker)
        else: res=fn(ticker)
    except Exception as e:
        log_warn(f"Engine call failed for {ticker}: {e}"); 
        return None
    return _series_from_engine_result(res)

def _load_price_series(ticker: str) -> Optional[pd.Series]:
    try:
        df=_load_df_any_alias(symbol=ticker) or _load_df_any_alias(ticker=ticker) or _load_df_any_alias(code=ticker)
        if isinstance(df, pd.DataFrame) and not df.empty:
            for c in ("Adj Close","adj_close","Close","close","c"):
                if c in df.columns:
                    s=df[c].astype(float).dropna(); break
            else:
                if all(x in df.columns for x in ("open","high","low","close")):
                    s=df["close"].astype(float).dropna()
                elif all(x in df.columns for x in ("Open","High","Low","Close")):
                    s=df["Close"].astype(float).dropna()
                else:
                    return None
            if len(s)==0: return None
            s.index=pd.to_datetime(df.index); s=s.sort_index()
            return (s / s.iloc[0]).rename(ticker)
    except Exception as e:
        log_warn(f"Price load failed for {ticker}: {e}")
    return None

def _build_equal_weight(curves: Dict[str,pd.Series]) -> Optional[pd.Series]:
    if not curves: return None
    all_idx=None
    for s in curves.values():
        all_idx = s.index if all_idx is None else all_idx.intersection(s.index)
    if all_idx is None or len(all_idx)==0:
        idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
        df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    else:
        df = pd.DataFrame({k: v.reindex(all_idx) for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty: return None
    w = np.ones(df.shape[1]) / df.shape[1]
    port = (df * w).sum(axis=1)
    if port.iloc[0] != 0: port = port / port.iloc[0]
    return port.rename("Portfölj")

# -------------------- UI --------------------
tabs = st.tabs(["Översikt", "Universum", "Transaktioner"])

with tabs[0]:
    colL, colR = st.columns([3,1])
    with colL:
        data_dir = st.text_input("Datakatalog (JSON för profiler)", value=st.session_state.get("portfolio_dir", DEFAULT_DIR), key="portfolio_dir")
    with colR:
        profiles = DEFAULT_DIR
        if os.path.isdir(profiles) and st.button("Använd profiles/"):
            st.session_state["portfolio_dir"]=profiles; st.rerun()

    files = list_json_files(data_dir)
    if not files:
        st.warning(f"Inga JSON-filer i `{data_dir}`."); st.stop()

    picked = st.multiselect("Välj profilfiler", files, default=files)
    if not picked:
        st.info("Välj minst en fil."); st.stop()

    df_prof = build_profiles_df(data_dir, picked)
    if df_prof.empty or "Ticker" not in df_prof.columns:
        st.error("Kunde inte hämta profiler med Ticker."); st.stop()

    tickers = sorted(set([t for t in df_prof["Ticker"].astype(str).str.strip().tolist() if t]))
    st.caption(f"Hittade {len(tickers)} unika tickers.")

    with st.expander("⚙️ Inställningar", expanded=False):
        prefer_strategy = st.checkbox("Använd strategi-kurvor om tillgängligt (annars Buy&Hold)", value=True)
        index_symbol = st.text_input("Indexsymbol (för blå 'OMXS30 (index)')", value="OMXS30")
        show_bh = st.checkbox("Visa Buy&Hold i grafen", value=True)
        start_date = st.date_input("Startdatum (valfritt)", value=None)
        run_btn = st.button("Bygg portfölj")

    if run_btn:
        engine_fn, engine_name = _detect_engine() if prefer_strategy else (None,"")
        curves: Dict[str,pd.Series] = {}
        for t in tickers:
            s = _call_engine_curve(engine_fn, t) if engine_fn is not None else None
            if s is None: s = _load_price_series(t)
            if s is not None and len(s)>5:
                if start_date: s = s[s.index >= pd.Timestamp(start_date)]
                curves[t] = s

        if not curves:
            st.error("Hittade inga kurvor att rita."); st.stop()

        port = _build_equal_weight(curves)
        if port is None or port.empty:
            st.error("Kunde inte konstruera portföljen."); st.stop()

        idx_curve = None
        if index_symbol:
            idx_curve = _load_price_series(index_symbol)
            if idx_curve is not None and start_date:
                idx_curve = idx_curve[idx_curve.index >= pd.Timestamp(start_date)]

        bh = None
        if show_bh:
            bh_curves = {}
            for t in tickers:
                s=_load_price_series(t)
                if s is not None:
                    if start_date: s = s[s.index>=pd.Timestamp(start_date)]
                    bh_curves[t]=s
            bh = _build_equal_weight(bh_curves) if bh_curves else None
            if bh is not None: bh = bh.rename("Buy&Hold")

        import altair as alt
        plot_df = pd.DataFrame({"date": port.index})
        plot_df["Portfölj"] = port.values
        if bh is not None: plot_df["Buy&Hold"] = bh.reindex(port.index).ffill().values
        if idx_curve is not None: plot_df["OMXS30 (index)"] = idx_curve.reindex(port.index).ffill().values
        plot_df = plot_df.melt("date", var_name="Series", value_name="Norm")

        st.subheader("Kapitalutveckling (normaliserad till 1.0)")
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Norm:Q", title="Normaliserat (x)"),
            color=alt.Color("Series:N")
        ).properties(height=420)
        st.altair_chart(chart, theme="streamlit", width="stretch")

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(OUT_DIR, f"portfolio_equity_{ts}.csv")
        plot_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        st.caption(f"Sparat kurvdata: `{out_csv}`")

with tabs[1]:
    st.subheader("Universum")
    data_dir = st.session_state.get("portfolio_dir", DEFAULT_DIR)
    files = list_json_files(data_dir)
    df = build_profiles_df(data_dir, files) if files else pd.DataFrame()
    if df.empty:
        st.info("Inga profiler hittades.")
    else:
        st.dataframe(df_brief(df, rows=50, cols=12), width="stretch")

with tabs[2]:
    st.subheader("Transaktioner")
    st.info("Om motorn returnerar affärer kan vi lista dem här. Säg till så kopplar jag in exakt format.")
[PASTE END]
PY

sudo systemctl restart trader-ui.service
sudo journalctl -u trader-ui.service -n 30 --no-pager
cd /srv/trader/app
# 1) Ta bort ev. inklistrade markörer och Windows-CRLF
sudo sed -i -E '/^\[PASTE START\]/d;/^\[PASTE END\]/d' pages/2_Portfolio.py
sudo sed -i 's/\r$//' pages/2_Portfolio.py
sudo sed -i '1s/^\xEF\xBB\xBF//' pages/2_Portfolio.py  # rensa ev. BOM
# 2) Snabb syntaxtest (ger felrad om nåt fortfarande är knas)
. .venv/bin/activate 2>/dev/null || true
python -m py_compile pages/2_Portfolio.py
# 3) Starta om tjänsten och kolla loggen
sudo systemctl restart trader-ui.service
sudo journalctl -u trader-ui.service -n 60 --no-pager
cd /srv/trader/app
sudo systemctl stop trader-ui.service
# Säkerhetskopia
sudo cp pages/2_Portfolio.py pages/2_Portfolio.py.bak_$(date +%F_%H%M%S)
# Ta bort PASTE-markörer + ev. Windows-CRLF + ev. BOM
sudo sed -i -E '/^\[PASTE START\]/d;/^\[PASTE END\]/d' pages/2_Portfolio.py
sudo sed -i 's/\r$//' pages/2_Portfolio.py
sudo sed -i '1s/^\xEF\xBB\xBF//' pages/2_Portfolio.py
# Snabb syntaxtest – ska vara tyst om allt är OK
. .venv/bin/activate 2>/dev/null || true
python -m py_compile pages/2_Portfolio.py
# Starta igen och visa färska loggar
sudo systemctl start trader-ui.service
sudo journalctl -u trader-ui.service -n 40 --no-pager
# (frivilligt) dubbelkolla att första raden verkligen är Python-kod
head -n1 pages/2_Portfolio.py
python - <<'PY'
import os, re, json, ast
DIR="/srv/trader/app/profiles"
def clean(t):
    t=re.sub(r"//.*?$","",t,flags=re.M)
    t=re.sub(r"/\*.*?\*/","",t,flags=re.S)
    t=re.sub(r",\s*([}\]])",r"\1",t)
    return t.strip()
def parse(path):
    txt=open(path,"r",encoding="utf-8-sig").read().strip()
    out=[]
    if not txt: return out
    def flt(d):
        if not isinstance(d,dict): return {}
        d=dict(d)
        for k in ("Ticker","ticker","symbol","code","name","shortName","short_name"):
            v=d.get(k)
            if isinstance(v,str) and v.strip():
                d["Ticker"]=v.strip(); break
        return d
    try:
        obj=json.loads(clean(txt))
        if isinstance(obj,list): out=[flt(x) for x in obj if isinstance(x,dict)]
        elif isinstance(obj,dict):
            for k in ("profiles","rows","items","results","data","list","values"):
                if isinstance(obj.get(k),list):
                    out=[flt(x) for x in obj[k] if isinstance(x,dict)]
                    break
            else:
                if obj and all(isinstance(v,dict) for v in obj.values()):
                    out=[flt(v)|{"_key":k} for k,v in obj.items()]
                else:
                    out=[flt(obj)]
    except Exception:
        for ln in txt.splitlines():
            ln=ln.strip()
            if not ln: continue
            try: out.append(flt(json.loads(clean(ln))))
            except Exception:
                try: out.append(flt(ast.literal_eval(ln)))
                except Exception: pass
    return out
for f in sorted(os.listdir(DIR)):
    if f.lower().endswith(".json"):
        rows=parse(os.path.join(DIR,f))
        tick=[(i+1,r.get("Ticker","")) for i,r in enumerate(rows)]
        print(f"{f}: {len(rows)} profiler -> {tick[:6]}")
PY

cd /srv/trader/app
sudo systemctl stop trader-ui.service
sudo cp pages/2_Portfolio.py pages/2_Portfolio.py.bak_profiles_$(date +%F_%H%M%S) 2>/dev/null || true
# Skriv om med stöd för profilval & robust tickers
sudo tee pages/2_Portfolio.py >/dev/null <<'PY'
from __future__ import annotations
import os, re, json, ast, inspect, traceback, datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# --- failsafe debug ---
try:
    from app.debuglog import setup_debug_ui, log_info, log_warn, log_error, df_brief
except Exception:
    def setup_debug_ui(*a, **k): pass
    def log_info(*a, **k): pass
    def log_warn(*a, **k): pass
    def log_error(*a, **k): pass
    def df_brief(x, rows=5, cols=8):
        try: return x.iloc[:rows, :cols]
        except Exception: return x

# --- riktig dataladdare ---
try:
    from app.__init__ import _load_df_any_alias as _load_df_any_alias
except Exception:
    try:
        from app.btwrap import _load_df_any_alias as _load_df_any_alias
    except Exception:
        def _load_df_any_alias(*args, **kwargs):
            return kwargs.get("df", pd.DataFrame())

st.set_page_config(page_title="Portfolio (profiler)", page_icon="💼", layout="wide")
st.title("💼 Portfolio (profiler)")
setup_debug_ui(st)

OUT_DIR = "trader/outputs"; os.makedirs(OUT_DIR, exist_ok=True)
DEFAULT_DIR = "/srv/trader/app/profiles"

# -------------------- Tålig profil-läsare --------------------
TICKER_KEYS = ("Ticker","ticker","symbol","code","name","shortName","short_name")

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
    nm = os.path.splitext(os.path.basename(path))[0]
    return (nm.split("_")[0] if "_" in nm else nm).strip() or None

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

def build_profiles_df(dir_path: str, files: List[str]) -> pd.DataFrame:
    rows=[]
    for f in files:
        p=os.path.join(dir_path,f)
        recs=parse_json_file(p)
        if recs:
            for r in recs:
                rr=dict(r); rr["source_file"]=f
                if not rr.get("Ticker"): rr["Ticker"]=_infer_ticker_from_filename(p) or ""
                rows.append(rr)
        else:
            rows.append({"Ticker": _infer_ticker_from_filename(p) or "", "source_file": f, "_note": "empty/parse-fail"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df.columns=[str(c).replace("\ufeff","").strip() for c in df.columns]
        first=[c for c in ("Ticker","source_file") if c in df.columns]
        df=df[first + [c for c in df.columns if c not in first]]
        # Numrera profiler per fil (1..n)
        df["ProfileIdx"] = df.groupby("source_file").cumcount() + 1
    return df

# -------------------- Motor / kurvor --------------------
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

def _series_from_engine_result(res: Any) -> Optional[pd.Series]:
    if res is None: return None
    s=None
    if isinstance(res, dict):
        candidates=["equity","curve","balance","cumret","portfolio","portfolio_value","equity_curve"]
        for k in candidates:
            v=res.get(k)
            if v is None: continue
            if isinstance(v, dict):
                try: s=pd.Series(v); break
                except Exception: pass
            if isinstance(v, list) and v:
                try:
                    if isinstance(v[0], dict):
                        key_date=None; key_val=None
                        for kd in ("date","Date","ts","timestamp"):
                            if kd in v[0]: key_date=kd; break
                        for kv in ("equity","value","val","y","close","Close"):
                            if kv in v[0]: key_val=kv; break
                        if key_date and key_val:
                            idx=pd.to_datetime([x[key_date] for x in v])
                            vals=[x[key_val] for x in v]
                            s=pd.Series(vals, index=idx); break
                    else:
                        s=pd.Series(v); break
                except Exception: pass
    if s is None and isinstance(res, pd.Series): s=res
    if s is None and isinstance(res, pd.DataFrame):
        for c in ("equity","value","val","close","Close"):
            if c in res.columns:
                try: s=res[c]; break
                except Exception: pass
    if s is None: return None
    s=s.dropna()
    if len(s)>0 and s.iloc[0]>0: s = s / float(s.iloc[0])
    try: s.index = pd.to_datetime(s.index)
    except Exception: pass
    return s.sort_index()

def _ticker_variants(t: str) -> List[str]:
    b = (t or "").strip()
    u = re.sub(r"\s+"," ", b).upper()
    cands = [b, u]
    # Vanliga variationer: mellanslag, '-', '.', inget
    for sep in [" ", "-", ".", ""]:
        cands.append(u.replace(" ", sep))
    # Unika i ordning
    seen=set(); uniq=[]
    for x in cands:
        if x and x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _load_price_series(ticker: str) -> Optional[pd.Series]:
    """Hämtar pris och bygger Buy&Hold-kurva. Testar vanliga varianter (VOLV B, VOLV-B, VOLV.B, VOLVB)."""
    for cand in _ticker_variants(ticker):
        try:
            df=_load_df_any_alias(symbol=cand) or _load_df_any_alias(ticker=cand) or _load_df_any_alias(code=cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                for c in ("Adj Close","adj_close","Close","close","c"):
                    if c in df.columns:
                        s=df[c].astype(float).dropna(); break
                else:
                    if all(x in df.columns for x in ("open","high","low","close")):
                        s=df["close"].astype(float).dropna()
                    elif all(x in df.columns for x in ("Open","High","Low","Close")):
                        s=df["Close"].astype(float).dropna()
                    else:
                        continue
                if len(s)==0: 
                    continue
                s.index=pd.to_datetime(df.index); s=s.sort_index()
                s=(s / s.iloc[0]).rename(ticker)  # döp tillbaka till original-ticker i grafen
                log_info(f"Loaded price for {ticker} via variant '{cand}'")
                return s
        except Exception as e:
            log_warn(f"Price load failed for {ticker} cand={cand}: {e}")
    log_warn(f"Price not found for {ticker}")
    return None

def _build_equal_weight(curves: Dict[str,pd.Series]) -> Optional[pd.Series]:
    if not curves: return None
    all_idx=None
    for s in curves.values():
        all_idx = s.index if all_idx is None else all_idx.intersection(s.index)
    if all_idx is None or len(all_idx)==0:
        idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
        df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    else:
        df = pd.DataFrame({k: v.reindex(all_idx) for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty: return None
    w = np.ones(df.shape[1]) / df.shape[1]
    port = (df * w).sum(axis=1)
    if port.iloc[0] != 0: port = port / port.iloc[0]
    return port.rename("Portfölj")

# -------------------- UI --------------------
tabs = st.tabs(["Översikt", "Universum", "Transaktioner"])

with tabs[0]:
    colL, colR = st.columns([3,1])
    with colL:
        data_dir = st.text_input("Datakatalog (JSON för profiler)", value=st.session_state.get("portfolio_dir", DEFAULT_DIR), key="portfolio_dir")
    with colR:
        profiles = DEFAULT_DIR
        if os.path.isdir(profiles) and st.button("Använd profiles/"):
            st.session_state["portfolio_dir"]=profiles; st.rerun()

    files = list_json_files(data_dir)
    if not files:
        st.warning(f"Inga JSON-filer i `{data_dir}`."); st.stop()

    picked = st.multiselect("Välj profilfiler", files, default=files)
    if not picked:
        st.info("Välj minst en fil."); st.stop()

    df_prof = build_profiles_df(data_dir, picked)
    if df_prof.empty or "Ticker" not in df_prof.columns:
        st.error("Kunde inte hämta profiler med Ticker."); st.stop()

    # Välj profilindex 1..3 per fil
    max_idx = int(df_prof["ProfileIdx"].max())
    options = ["Alla"] + [f"Profil #{i}" for i in range(1, max_idx+1)]
    profile_choice = st.selectbox("Vilken profil i varje fil?", options, index=0)
    if profile_choice != "Alla":
        pick = int(profile_choice.split("#")[1])
        df_prof = df_prof[df_prof["ProfileIdx"] == pick]

    tickers = sorted(set([t for t in df_prof["Ticker"].astype(str).str.strip().tolist() if t]))
    st.caption(f"Hittade {len(tickers)} tickers från {len(df_prof['source_file'].unique())} filer (val: {profile_choice}).")

    with st.expander("⚙️ Inställningar", expanded=False):
        prefer_strategy = st.checkbox("Använd strategi-kurvor om tillgängligt (annars Buy&Hold)", value=True)
        index_symbol = st.text_input("Indexsymbol (för blå 'OMXS30 (index)')", value="OMXS30")
        show_bh = st.checkbox("Visa Buy&Hold i grafen", value=True)
        start_date = st.date_input("Startdatum (valfritt)", value=None)
        run_btn = st.button("Bygg portfölj")

    if run_btn:
        engine_fn, engine_name = _detect_engine() if prefer_strategy else (None,"")
        curves: Dict[str,pd.Series] = {}
        for t in tickers:
            s = _call_engine_curve(engine_fn, t) if engine_fn is not None else None
            if s is None: s = _load_price_series(t)
            if s is not None and len(s)>5:
                if start_date: s = s[s.index >= pd.Timestamp(start_date)]
                curves[t] = s

        if not curves:
            st.error("Hittade inga kurvor att rita."); st.stop()

        port = _build_equal_weight(curves)
        if port is None or port.empty:
            st.error("Kunde inte konstruera portföljen."); st.stop()

        idx_curve = None
        if index_symbol:
            idx_curve = _load_price_series(index_symbol)
            if idx_curve is not None and start_date:
                idx_curve = idx_curve[idx_curve.index >= pd.Timestamp(start_date)]

        bh = None
        if show_bh:
            bh_curves = {}
            for t in tickers:
                s=_load_price_series(t)
                if s is not None:
                    if start_date: s = s[s.index>=pd.Timestamp(start_date)]
                    bh_curves[t]=s
            bh = _build_equal_weight(bh_curves) if bh_curves else None
            if bh is not None: bh = bh.rename("Buy&Hold")

        import altair as alt
        plot_df = pd.DataFrame({"date": port.index})
        plot_df["Portfölj"] = port.values
        if bh is not None: plot_df["Buy&Hold"] = bh.reindex(port.index).ffill().values
        if idx_curve is not None: plot_df["OMXS30 (index)"] = idx_curve.reindex(port.index).ffill().values
        plot_df = plot_df.melt("date", var_name="Series", value_name="Norm")

        st.subheader("Kapitalutveckling (normaliserad till 1.0)")
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Norm:Q", title="Normaliserat (x)"),
            color=alt.Color("Series:N")
        ).properties(height=420)
        st.altair_chart(chart, theme="streamlit", width="stretch")

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(OUT_DIR, f"portfolio_equity_{ts}.csv")
        plot_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        st.caption(f"Sparat kurvdata: `{out_csv}`")

with tabs[1]:
    st.subheader("Universum")
    data_dir = st.session_state.get("portfolio_dir", DEFAULT_DIR)
    files = list_json_files(data_dir)
    df = build_profiles_df(data_dir, files) if files else pd.DataFrame()
    if df.empty:
        st.info("Inga profiler hittades.")
    else:
        st.dataframe(df_brief(df, rows=50, cols=12), width="stretch")

with tabs[2]:
    st.subheader("Transaktioner")
    st.info("Om motorn returnerar affärer kan vi lista dem här. Säg till så kopplar jag in exakt format.")
PY

# Verifiera och starta
. .venv/bin/activate 2>/dev/null || true
python -m py_compile pages/2_Portfolio.py
sudo systemctl start trader-ui.service
cd /srv/trader/app
sudo systemctl stop trader-ui.service
sudo cp pages/2_Portfolio.py pages/2_Portfolio.py.bak_bdonly_$(date +%F_%H%M%S) 2>/dev/null || true
sudo tee pages/2_Portfolio.py >/dev/null <<'PY'
from __future__ import annotations
import os, re, json, ast, inspect, datetime as dt
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# --- debug (failsafe) ---
try:
    from app.debuglog import setup_debug_ui, log_info, log_warn, log_error, df_brief
except Exception:
    def setup_debug_ui(*a, **k): pass
    def log_info(*a, **k): pass
    def log_warn(*a, **k): pass
    def log_error(*a, **k): pass
    def df_brief(x, rows=5, cols=8):
        try: return x.iloc[:rows, :cols]
        except Exception: return x

st.set_page_config(page_title="Portfolio (profiler)", page_icon="💼", layout="wide")
st.title("💼 Portfolio (profiler)")
setup_debug_ui(st)

DEFAULT_DIR = "/srv/trader/app/profiles"

# -------------------- BÖRSDATA-LOADER (ENDA KÄLLAN) --------------------
def _get_bors_loader():
    """
    Försök att hitta er Börsdata-laddare i projektet.
    Lägg gärna till rätt modul/funktion här om ni har en specifik.
    Denna funktion ska returnera en call: load(symbol) -> DataFrame med OHLC/Close.
    """
    candidates = [
        ("app.borsdata", "load_df"),
        ("app.borsdata", "load_ohlc"),
        ("app.dataproviders.borsdata", "load_df"),
        ("app.data", "borsdata_load_df"),
    ]
    for mod, attr in candidates:
        try:
            m = __import__(mod, fromlist=[attr])
            f = getattr(m, attr, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            continue
    # sista fallback – om er generiska loader i sin tur är kopplad till Börsdata
    try:
        from app.__init__ import _load_df_any_alias as f
        return lambda s: f(symbol=s)
    except Exception:
        pass
    try:
        from app.btwrap import _load_df_any_alias as f
        return lambda s: f(symbol=s)
    except Exception:
        pass
    return None

_BD_LOAD = _get_bors_loader()

def _ticker_variants_borsdata(t: str) -> List[str]:
    """Prova några vanliga Börsdata-varianter (mellanslag, bindestreck, punkt, ihop)."""
    b = (t or "").strip()
    u = re.sub(r"\s+"," ", b).upper()
    cands = [b, u]
    for sep in [" ", "-", ".", ""]:
        cands.append(u.replace(" ", sep))
    # unika i ordning
    seen=set(); out=[]
    for c in cands:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

def _load_price_series_borsdata(ticker: str) -> Optional[pd.Series]:
    if _BD_LOAD is None:
        log_error("Börsdata-laddare saknas – ange rätt modul i _get_bors_loader().")
        return None
    for cand in _ticker_variants_borsdata(ticker):
        try:
            df = _BD_LOAD(cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Vanliga kolumnnamn
                col = None
                for c in ("Adj Close","adj_close","Close","close","c"):
                    if c in df.columns: col=c; break
                if col is None:
                    if all(x in df.columns for x in ("open","high","low","close")):
                        col = "close"
                    elif all(x in df.columns for x in ("Open","High","Low","Close")):
                        col = "Close"
                if not col: 
                    continue
                s = df[col].astype(float).dropna()
                if s.empty: 
                    continue
                s.index = pd.to_datetime(df.index)
                s = s.sort_index()
                s = (s / s.iloc[0]).rename(ticker)  # visa originaltäckaren i grafen
                log_info(f"Börsdata OK: {ticker} via '{cand}'")
                return s
        except Exception as e:
            log_warn(f"Börsdata misslyckades för {ticker} cand={cand}: {e}")
    log_warn(f"Ingen Börsdata-serie för {ticker}")
    return None

# -------------------- PROFIL-LÄSARE --------------------
TICKER_KEYS = ("Ticker","ticker","symbol","code","name","shortName","short_name")
WEIGHT_KEYS = ("weight","Weight","w","alloc","allocation","Allocation","size")

def _strip_comments_and_trailing_commas(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()

def _try_ast_literal_eval(text: str) -> Any:
    t = re.sub(r"\btrue\b","True", re.sub(r"\bfalse\b","False", re.sub(r"\bnull\b","None", text, flags=re.I), flags=re.I), flags=re.I)
    try: return ast.literal_eval(t)
    except Exception: return None

def _flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    # Ticker
    for k in TICKER_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            d["Ticker"] = v.strip(); break
    # Eventuella kapslade params till top-level
    for nest in ("params","config","settings","options"):
        if isinstance(d.get(nest), dict):
            for k,v in d[nest].items(): d.setdefault(k,v)
            del d[nest]
    # Vikt
    for wk in WEIGHT_KEYS:
        if wk in d:
            try:
                d["Weight"] = float(d[wk])
            except Exception:
                pass
            break
    return d

def _json_to_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [_flatten(x) for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("profiles","rows","items","results","data","list","values"):
            if isinstance(obj.get(k), list):
                return [_flatten(x) for x in obj[k] if isinstance(x, dict)]
        # dict-of-dicts: { "GETI B": {...}, "HM B": {...} }
        if obj and all(isinstance(v, dict) for v in obj.values()):
            out=[]
            for tk, payload in obj.items():
                r=_flatten(payload)
                if "Ticker" not in r and isinstance(tk,str) and tk.strip():
                    r["Ticker"]=tk.strip()
                out.append(r)
            return out
        # single
        return [_flatten(obj)]
    return []

def parse_json_file(path: str) -> List[Dict[str, Any]]:
    try:
        text = open(path,"r",encoding="utf-8-sig").read().strip()
        if not text: return []
        # NDJSON?
        if "\n" in text and text.lstrip().startswith("{") and not text.strip().startswith("["):
            out=[]
            for ln in [ln.strip() for ln in text.splitlines() if ln.strip()]:
                for attempt in (lambda x: json.loads(x), _try_ast_literal_eval):
                    try:
                        obj = attempt(_strip_comments_and_trailing_commas(ln))
                        if isinstance(obj, dict): out.append(_flatten(obj)); break
                    except Exception:
                        pass
            return out
        # vanlig JSON
        try:
            obj = json.loads(_strip_comments_and_trailing_commas(text))
        except Exception:
            obj = _try_ast_literal_eval(_strip_comments_and_trailing_commas(text))
        return _json_to_rows(obj) if obj is not None else []
    except Exception as e:
        log_warn(f"Parse fail {path}: {e}")
        return []

@st.cache_data(ttl=120, show_spinner=False)
def list_json_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path): return []
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith((".json",".jsonl",".ndjson"))])

def build_profiles_df(dir_path: str, files: List[str]) -> pd.DataFrame:
    rows=[]
    for f in files:
        p=os.path.join(dir_path,f)
        recs=parse_json_file(p)
        if recs:
            for r in recs:
                rr=dict(r); rr["source_file"]=f
                if not rr.get("Ticker"):
                    rr["Ticker"]=os.path.splitext(f)[0]
                rows.append(rr)
        else:
            rows.append({"Ticker": os.path.splitext(f)[0], "source_file": f, "_note": "empty/parse-fail"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df.columns=[str(c).replace("\ufeff","").strip() for c in df.columns]
        # Numrera profiler per fil (1..n)
        df["ProfileIdx"] = df.groupby("source_file").cumcount() + 1
    return df

# -------------------- Portfölj-beräkning --------------------
def _build_portfolio(curves: Dict[str,pd.Series], weights: Optional[Dict[str,float]]=None) -> Optional[pd.Series]:
    if not curves: return None
    # gemensam tidsaxel
    idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty: return None
    if not weights:
        w = np.ones(df.shape[1]) / df.shape[1]
    else:
        ww = np.array([max(0.0, float(weights.get(k, 0.0))) for k in df.columns], dtype=float)
        if ww.sum() <= 0:
            ww = np.ones_like(ww)
        w = ww / ww.sum()
    port = (df * w).sum(axis=1)
    if len(port)>0 and port.iloc[0] != 0:
        port = port / port.iloc[0]
    return port.rename("Portfölj")

# -------------------- UI --------------------
tabs = st.tabs(["Översikt", "Universum"])

with tabs[0]:
    colL, colR = st.columns([3,1], gap="large")
    with colL:
        data_dir = st.text_input("Datakatalog (JSON med tre profiler per fil)", value=st.session_state.get("portfolio_dir", DEFAULT_DIR), key="portfolio_dir")
    with colR:
        if os.path.isdir(DEFAULT_DIR) and st.button("Använd profiles/"):
            st.session_state["portfolio_dir"]=DEFAULT_DIR; st.rerun()

    files = list_json_files(data_dir)
    if not files:
        st.warning(f"Inga JSON-filer i `{data_dir}`."); st.stop()

    picked = st.multiselect("Välj profilfiler", files, default=files)
    if not picked:
        st.info("Välj minst en fil."); st.stop()

    df_prof = build_profiles_df(data_dir, picked)
    if df_prof.empty or "Ticker" not in df_prof.columns:
        st.error("Kunde inte hämta profiler med Ticker."); st.stop()

    max_idx = int(df_prof["ProfileIdx"].max())
    options = ["Alla"] + [f"Profil #{i}" for i in range(1, max_idx+1)]
    profile_choice = st.selectbox("Vilken profil i varje fil ska användas?", options, index=1)  # default Profil #1
    if profile_choice != "Alla":
        pick = int(profile_choice.split("#")[1])
        df_prof = df_prof[df_prof["ProfileIdx"] == pick]

    # extrahera vikter om de finns
    weight_col = next((c for c in WEIGHT_KEYS if c in df_prof.columns), None)
    if weight_col and "Weight" not in df_prof.columns:
        df_prof["Weight"] = pd.to_numeric(df_prof[weight_col], errors="coerce")

    tickers = [t for t in df_prof["Ticker"].astype(str).str.strip().tolist() if t]
    weights = None
    if "Weight" in df_prof.columns:
        wmap = {}
        for t, w in zip(df_prof["Ticker"], df_prof["Weight"].fillna(0)):
            wmap[str(t).strip()] = float(w)
        weights = wmap

    st.caption(f"{len(tickers)} tickers från {len(df_prof['source_file'].unique())} filer (val: {profile_choice}).")

    with st.expander("⚙️ Inställningar", expanded=False):
        index_symbol = st.text_input("Indexsymbol (t.ex. OMXS30)", value="OMXS30")
        start_date = st.date_input("Startdatum (valfritt)", value=None)
        run_btn = st.button("Bygg portfölj")

    if run_btn:
        curves: Dict[str,pd.Series] = {}
        for t in sorted(set(tickers)):
            s = _load_price_series_borsdata(t)
            if s is not None and len(s)>5:
                if start_date: s = s[s.index >= pd.Timestamp(start_date)]
                curves[t] = s

        if not curves:
            st.error("Hittade inga kurvor att rita (Börsdata returnerade inget för valda tickers)."); st.stop()

        port = _build_portfolio(curves, weights=weights)
        if port is None or port.empty:
            st.error("Kunde inte konstruera portföljen."); st.stop()

        # Index (från Börsdata)
        idx_curve = None
        if index_symbol.strip():
            idx_curve = _load_price_series_borsdata(index_symbol.strip())
            if idx_curve is not None and start_date:
                idx_curve = idx_curve[idx_curve.index >= pd.Timestamp(start_date)]

        import altair as alt
        plot_df = pd.DataFrame({"date": port.index, "Portfölj": port.values})
        if idx_curve is not None:
            plot_df["Index"] = idx_curve.reindex(port.index).ffill().values
        plot_df = plot_df.melt("date", var_name="Serie", value_name="Norm")

        st.subheader("Kapitalutveckling (normaliserad till 1.0)")
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Norm:Q", title="Normaliserat (x)"),
            color=alt.Color("Serie:N")
        ).properties(height=420)
        st.altair_chart(chart, theme="streamlit", width="stretch")

with tabs[1]:
    st.subheader("Universum (profiler och eventuella vikter)")
    data_dir = st.session_state.get("portfolio_dir", DEFAULT_DIR)
    files = list_json_files(data_dir)
    df = build_profiles_df(data_dir, files) if files else pd.DataFrame()
    if df.empty:
        st.info("Inga profiler hittades.")
    else:
        # visa relevanta kolumner först
        cols = [c for c in ["source_file","ProfileIdx","Ticker","Weight"] if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        st.dataframe(df[cols], width="stretch")
PY

# Syntaxtest & start
. .venv/bin/activate 2>/dev/null || true
python -m py_compile pages/2_Portfolio.py
sudo systemctl start trader-ui.service
Hittade inga kurvor att rita (Börsdata returnerade inget för valda tickers).
cd /srv/trader/app
. .venv/bin/activate 2>/dev/null || true
python - <<'PY'
import importlib, inspect, re, pandas as pd

# Hitta vilken loader vi har tillgänglig
src = None
loader = None
for m,a in [
    ("app.borsdata","load_df"),
    ("app.borsdata","load_ohlc"),
    ("app.dataproviders.borsdata","load_df"),
    ("app.data","borsdata_load_df"),
]:
    try:
        mod = importlib.import_module(m); f = getattr(mod, a, None)
        if callable(f):
            loader = lambda s, f=f: f(symbol=s)
            src = f"{m}.{a}"
            break
    except Exception:
        pass

if loader is None:
    try:
        from app.__init__ import _load_df_any_alias as f
        loader = lambda s, f=f: f(symbol=s)
        src = "app.__init__._load_df_any_alias"
    except Exception:
        try:
            from app.btwrap import _load_df_any_alias as f
            loader = lambda s, f=f: f(symbol=s)
            src = "app.btwrap._load_df_any_alias"
        except Exception:
            src = "NONE"

print("Loader source:", src)

def variants(t):
    b = (t or "").strip()
    u = re.sub(r"\s+"," ", b).upper()
    base = [b, u]
    out = []
    for base_ in base:
        for sep in [" ","-",".",""]:
            v = base_.replace(" ", sep)
            out += [v, v+".ST"]
    # unika i ordning
    seen=set(); uniq=[]
    for x in out:
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

syms = ["GETI B","HM B","VOLV B","YUBICO","OMXS30"]
for s in syms:
    ok=None
    tried=[]
    if loader is None:
        print(f"{s}: NO LOADER")
        continue
    for v in variants(s):
        try:
            df = loader(v)
            good = isinstance(df, pd.DataFrame) and not df.empty
            tried.append((v, good, (list(df.columns)[:6] if good else [])))
            if good:
                ok=v; break
        except Exception as e:
            tried.append((v, False, [f"EXC:{type(e).__name__}"]))
    print(f"{s}: FOUND={ok}")
    for v,g,cols in tried[:10]:
        print("  ", v, "OK" if g else "fail", cols)
PY

cd /srv/trader/app
sudo tee app/opt_api.py >/dev/null <<'PY'
from __future__ import annotations
import re, inspect, importlib
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

# -------- Börsdata-laddare (samma som Optimizer använder) --------
def _get_bors_loader():
    # Lägg gärna till er exakta loader här om ni har en specifik modul/funktion
    for m,a in [
        ("app.borsdata","load_df"),
        ("app.borsdata","load_ohlc"),
        ("app.dataproviders.borsdata","load_df"),
        ("app.data","borsdata_load_df"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    # fallback: generiska loadern som Optimizer kan ha bundit till Börsdata
    for m,a in [
        ("app.__init__", "_load_df_any_alias"),
        ("app.btwrap",   "_load_df_any_alias"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    return None

_BD_LOAD = _get_bors_loader()

def _ticker_variants_borsdata(t: str) -> List[str]:
    """Generera varianter som Börsdata ofta accepterar, inkl .ST."""
    b=(t or "").strip()
    u=re.sub(r"\s+"," ", b).upper()
    cands=[]
    for base in (b, u):
        for sep in [" ","-",".",""]:
            v = base.replace(" ", sep)
            cands.extend([v, v + ".ST"])
    # unika i ordning
    seen=set(); out=[]
    for c in cands:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def load_price_series(ticker: str) -> Optional[pd.Series]:
    """Hämtar Buy&Hold från Börsdata (normaliserad till 1.0)."""
    if _BD_LOAD is None:
        return None
    for cand in _ticker_variants_borsdata(ticker):
        try:
            df = _BD_LOAD(cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                col = None
                for c in ("Adj Close","adj_close","Close","close","c"):
                    if c in df.columns: col=c; break
                if col is None:
                    if all(x in df.columns for x in ("open","high","low","close")):
                        col="close"
                    elif all(x in df.columns for x in ("Open","High","Low","Close")):
                        col="Close"
                if not col:
                    continue
                s = df[col].astype(float).dropna()
                if s.empty: 
                    continue
                s.index = pd.to_datetime(df.index)
                s = s.sort_index()
                return (s / s.iloc[0]).rename(ticker)
        except Exception:
            pass
    return None

# -------- Optimizer-motor (strategikurva) --------
# Vi försöker hitta en vettig "run" i era moduler
_ENGINE_CANDIDATES = [
    ("app.optimizer",             "run_for_ticker"),
    ("app.optimizer",             "run_profile"),
    ("app.optimizer",             "run_single"),
    ("app.portfolio_engine",      "run_for_ticker"),
    ("app.portfolio_backtest",    "run_for_ticker"),
    ("app.backtest",              "run_backtest"),
    ("app.backtester",            "run_single"),
]

def detect_engine() -> Tuple[Optional[Any], str]:
    for m, a in _ENGINE_CANDIDATES:
        try:
            mod = importlib.import_module(m)
            fn  = getattr(mod, a, None)
            if callable(fn):
                return fn, f"{m}.{a}"
        except Exception:
            pass
    return None, ""

def _series_from_result(res: Any) -> Optional[pd.Series]:
    """Plocka ut equity/kurva från typiska returformat."""
    if res is None:
        return None
    s=None
    if isinstance(res, dict):
        for k in ["equity","curve","balance","cumret","portfolio","portfolio_value","equity_curve"]:
            v=res.get(k)
            if v is None: 
                continue
            if isinstance(v, dict):
                try: s=pd.Series(v); break
                except Exception: pass
            if isinstance(v, list) and v:
                try:
                    if isinstance(v[0], dict):
                        kd = next((x for x in ["date","Date","ts","timestamp"] if x in v[0]), None)
                        kv = next((x for x in ["equity","value","val","y","close","Close"] if x in v[0]), None)
                        if kd and kv:
                            idx=pd.to_datetime([x[kd] for x in v])
                            vals=[x[kv] for x in v]
                            s=pd.Series(vals, index=idx); break
                    else:
                        s=pd.Series(v); break
                except Exception: pass
    if s is None and isinstance(res, pd.Series): s=res
    if s is None and isinstance(res, pd.DataFrame):
        for c in ("equity","value","val","close","Close"):
            if c in res.columns:
                try: s=res[c]; break
                except Exception: pass
    if s is None: 
        return None
    s = s.dropna()
    if len(s)>0 and s.iloc[0]>0:
        s = s / float(s.iloc[0])
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()

def get_strategy_curve(ticker: str, profile: Optional[Dict[str,Any]]=None, profile_index: Optional[int]=None) -> Optional[pd.Series]:
    """Kör Optimizer/strategi för en ticker. Skickar profil/params om motorn tar emot dem."""
    fn, name = detect_engine()
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        # Försök matcha parametrar hos motorn:
        if "ticker" in sig.parameters: kwargs["ticker"] = ticker
        elif "symbol" in sig.parameters: kwargs["symbol"] = ticker
        elif "code"   in sig.parameters: kwargs["code"]   = ticker

        # profil/params om det stöds
        for key in ["profile", "params", "config", "settings"]:
            if key in sig.parameters and profile is not None:
                kwargs[key] = profile

        if "profile_index" in sig.parameters and profile_index is not None:
            kwargs["profile_index"] = profile_index

        res = fn(**kwargs) if kwargs else fn(ticker)
        return _series_from_result(res)
    except Exception:
        return None

def build_portfolio(curves: Dict[str,pd.Series], weights: Optional[Dict[str,float]]=None) -> Optional[pd.Series]:
    if not curves:
        return None
    idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty:
        return None
    if not weights:
        w = np.ones(df.shape[1]) / df.shape[1]
    else:
        ww = np.array([max(0.0, float(weights.get(k, 0.0))) for k in df.columns], dtype=float)
        if ww.sum() <= 0: ww = np.ones_like(ww)
        w = ww / ww.sum()
    port = (df * w).sum(axis=1)
    if len(port)>0 and port.iloc[0] != 0:
        port = port / port.iloc[0]
    return port.rename("Portfölj")
PY

sudo systemctl stop trader-ui.service
python - <<'PY'
p="pages/2_Portfolio.py"
s=open(p,'r',encoding='utf-8').read()

# Importera vårt API högst upp
if "from app.opt_api import " not in s:
    s = s.replace(
        "import streamlit as st",
        "import streamlit as st\nfrom app.opt_api import load_price_series, get_strategy_curve, build_portfolio"
    )

# Ersätt interna funktioner (om de finns) med anrop till opt_api
s = re.sub(r"def _load_price_series\(.*?^def ", "def ", s, flags=re.S|re.M)  # ta bort gamla loader-funktionen
s = re.sub(r"def _build_equal_weight\(.*?^def ", "def ", s, flags=re.S|re.M) # ta bort gammal equal-weight

# Byt anrop vid körning
s = s.replace("_build_equal_weight(", "build_portfolio(")
s = s.replace("_load_price_series(", "load_price_series(")

# Använd optimizer-kurva där prefer_strategy är True
s = s.replace(
    "s = _call_engine_curve(engine_fn, t) if engine_fn is not None else None",
    "s = get_strategy_curve(t, profile=None, profile_index=None) if prefer_strategy else None"
)

open(p,'w',encoding='utf-8').write(s)
print("Patchad pages/2_Portfolio.py")
PY

# Syntaxtest & start
. .venv/bin/activate 2>/dev/null || true
python -m py_compile app/opt_api.py pages/2_Portfolio.py
sudo systemctl start trader-ui.service
cd /srv/trader/app
sudo systemctl stop trader-ui.service
sudo tee app/opt_api.py >/dev/null <<'PY'
from __future__ import annotations
import re, inspect, importlib
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

def _get_bors_loader():
    for m,a in [
        ("app.borsdata","load_df"),
        ("app.borsdata","load_ohlc"),
        ("app.dataproviders.borsdata","load_df"),
        ("app.data","borsdata_load_df"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    for m,a in [
        ("app.__init__", "_load_df_any_alias"),
        ("app.btwrap",   "_load_df_any_alias"),
    ]:
        try:
            mod = importlib.import_module(m)
            f = getattr(mod, a, None)
            if callable(f):
                return lambda s: f(symbol=s)
        except Exception:
            pass
    return None

_BD_LOAD = _get_bors_loader()

def _ticker_variants_borsdata(t: str) -> List[str]:
    b=(t or "").strip()
    u=re.sub(r"\s+"," ", b).upper()
    cands=[]
    for base in (b, u):
        for sep in [" ","-",".",""]:
            v = base.replace(" ", sep)
            cands.extend([v, v + ".ST"])
    seen=set(); out=[]
    for c in cands:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def load_price_series(ticker: str) -> Optional[pd.Series]:
    if _BD_LOAD is None:
        return None
    for cand in _ticker_variants_borsdata(ticker):
        try:
            df = _BD_LOAD(cand)
            if isinstance(df, pd.DataFrame) and not df.empty:
                col=None
                for c in ("Adj Close","adj_close","Close","close","c"):
                    if c in df.columns: col=c; break
                if col is None:
                    if all(x in df.columns for x in ("open","high","low","close")):
                        col="close"
                    elif all(x in df.columns for x in ("Open","High","Low","Close")):
                        col="Close"
                if not col: 
                    continue
                s = df[col].astype(float).dropna()
                if s.empty: 
                    continue
                s.index = pd.to_datetime(df.index)
                s = s.sort_index()
                return (s / s.iloc[0]).rename(ticker)
        except Exception:
            pass
    return None

_ENGINE_CANDIDATES = [
    ("app.optimizer",             "run_for_ticker"),
    ("app.optimizer",             "run_profile"),
    ("app.optimizer",             "run_single"),
    ("app.portfolio_engine",      "run_for_ticker"),
    ("app.portfolio_backtest",    "run_for_ticker"),
    ("app.backtest",              "run_backtest"),
    ("app.backtester",            "run_single"),
]

def detect_engine() -> Tuple[Optional[Any], str]:
    for m, a in _ENGINE_CANDIDATES:
        try:
            mod = importlib.import_module(m)
            fn  = getattr(mod, a, None)
            if callable(fn):
                return fn, f"{m}.{a}"
        except Exception:
            pass
    return None, ""

def _series_from_result(res: Any) -> Optional[pd.Series]:
    if res is None:
        return None
    s=None
    if isinstance(res, dict):
        for k in ["equity","curve","balance","cumret","portfolio","portfolio_value","equity_curve"]:
            v=res.get(k)
            if v is None: 
                continue
            if isinstance(v, dict):
                try: s=pd.Series(v); break
                except Exception: pass
            if isinstance(v, list) and v:
                try:
                    if isinstance(v[0], dict):
                        kd = next((x for x in ["date","Date","ts","timestamp"] if x in v[0]), None)
                        kv = next((x for x in ["equity","value","val","y","close","Close"] if x in v[0]), None)
                        if kd and kv:
                            idx=pd.to_datetime([x[kd] for x in v])
                            vals=[x[kv] for x in v]
                            s=pd.Series(vals, index=idx); break
                    else:
                        s=pd.Series(v); break
                except Exception: pass
    if s is None and isinstance(res, pd.Series): s=res
    if s is None and isinstance(res, pd.DataFrame):
        for c in ("equity","value","val","close","Close"):
            if c in res.columns:
                try: s=res[c]; break
                except Exception: pass
    if s is None: 
        return None
    s = s.dropna()
    if len(s)>0 and s.iloc[0]>0:
        s = s / float(s.iloc[0])
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    return s.sort_index()

def get_strategy_curve(ticker: str, profile: Optional[Dict[str,Any]]=None, profile_index: Optional[int]=None) -> Optional[pd.Series]:
    fn, name = detect_engine()
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        kwargs = {}
        if "ticker" in sig.parameters: kwargs["ticker"] = ticker
        elif "symbol" in sig.parameters: kwargs["symbol"] = ticker
        elif "code"   in sig.parameters: kwargs["code"]   = ticker

        for key in ["profile", "params", "config", "settings"]:
            if key in sig.parameters and profile is not None:
                kwargs[key] = profile

        if "profile_index" in sig.parameters and profile_index is not None:
            kwargs["profile_index"] = profile_index

        res = fn(**kwargs) if kwargs else fn(ticker)
        return _series_from_result(res)
    except Exception:
        return None

def build_portfolio(curves: Dict[str,pd.Series], weights: Optional[Dict[str,float]]=None) -> Optional[pd.Series]:
    if not curves:
        return None
    idx = sorted(set().union(*[set(s.index) for s in curves.values()]))
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()})
    df = df.dropna(how="all")
    if df.empty:
        return None
    if not weights:
        w = np.ones(df.shape[1]) / df.shape[1]
    else:
        ww = np.array([max(0.0, float(weights.get(k, 0.0))) for k in df.columns], dtype=float)
        if ww.sum() <= 0: ww = np.ones_like(ww)
        w = ww / ww.sum()
    port = (df * w).sum(axis=1)
    if len(port)>0 and port.iloc[0] != 0:
        port = port / port.iloc[0]
    return port.rename("Portfölj")
PY

python - <<'PY'
import re, sys
p="pages/2_Portfolio.py"
s=open(p,'r',encoding='utf-8').read()

# 1) Lägg in opt_api-import om den saknas
if "from app.opt_api import load_price_series" not in s:
    s=s.replace("import streamlit as st",
                "import streamlit as st\nfrom app.opt_api import load_price_series, get_strategy_curve, build_portfolio")

# 2) Byt anropsställen till vårt API
s=s.replace("_build_equal_weight(", "build_portfolio(")
s=s.replace("_load_price_series(", "load_price_series(")
s=s.replace("s = _call_engine_curve(engine_fn, t) if engine_fn is not None else None",
            "s = get_strategy_curve(t, profile=None, profile_index=None) if prefer_strategy else None")

# 3) Ta bort lokala helper-defs om de finns (för att undvika skuggning)
s=re.sub(r"(?s)def\s+_build_equal_weight\([^)]*\):.*?(?=^\s*def\s|\Z)", "", s, flags=re.M)
s=re.sub(r"(?s)def\s+_load_price_series\([^)]*\):.*?(?=^\s*def\s|\Z)", "", s, flags=re.M)

# 4) Ta bort ev. CSV-skrivning
s=re.sub(r".*\.to_csv\(.*\)\s*\n", "", s)

open(p,'w',encoding='utf-8').write(s)
print("OK: patched", p)
PY

. .venv/bin/activate 2>/dev/null || true
python -m py_compile app/opt_api.py pages/2_Portfolio.py && echo "Syntax OK"
sudo systemctl start trader-ui.service
sudo journalctl -u trader-ui.service -n 40 --no-pager
python - <<'PY'
from app.opt_api import detect_engine, load_price_series
fn,name = detect_engine()
print("Engine:", name or "NOT FOUND")
for t in ["GETI B","HM B","VOLV B","YUBICO","OMXS30"]:
    s = load_price_series(t)
    print(t, "BH", "OK" if s is not None and len(s)>5 else "FAIL")
PY

cd /srv/trader/app
echo "=== (1) Optimizer → var hamnar de 3 profilerna? ==="
# De här fanns tidigare; lista senaste 10 profiler + tider
find profiles -maxdepth 1 -type f -iname '*.json' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort | tail -n 10
echo
echo "=== (2) Backtest MIN → var läser den profiler? ==="
# Visa imports + ev. sökvägar/profilhantering
nl -ba pages/1_Backtest_MIN.py | sed -n '1,160p'
echo
echo "=== (3) Finns 'Universum' i koden (mapp/flagga/fil)? ==="
grep -RIn --line-number --color -E 'universum|universe|univers' pages app 2>/dev/null | head -n 50
echo
echo "=== (4) Vilken dataladdare (Börsdata) använder Optimizer/Backtest? ==="
grep -RIn --line-number --color -E 'borsdata|Borsdata|load_ohlc|load_df|_load_df_any|price|ohlc' app pages 2>/dev/null | head -n 80
echo
echo "=== (5) Visa 1 exempelfil (profil) precis som systemet sparar den ==="
# Välj en av dina riktiga profiler (byt filnamn om du vill)
FILE="$(ls -1t profiles/*.json 2>/dev/null | head -n 1)"
echo "Profilfil: $FILE"
[ -n "$FILE" ] && sed -n '1,200p' "$FILE"
cd /srv/trader/app
sudo systemctl stop trader-ui.service
# 1) Backup av nuvarande portföljsida
[ -f pages/2_Portfolio.py ] && sudo cp -v pages/2_Portfolio.py "pages/2_Portfolio.py.bak_$(date +%F_%H%M%S)"
# 2) Skriv en minimal, robust portföljsida som använder samma väg som Backtest MIN
sudo tee pages/2_Portfolio.py >/dev/null <<'PY'
from __future__ import annotations
import json, re
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Samma dataväg som Backtest MIN ---
GET = None           # app.data_providers.get_ohlcv (Börsdata)
RUN_WRAP = None      # app.btwrap.run_backtest (om finns)
RUN_RAW  = None      # backtest.run_backtest (fallback)

try:
    from app.data_providers import get_ohlcv as GET
except Exception as e:
    st.error(f"Kunde inte importera app.data_providers.get_ohlcv: {type(e).__name__}: {e}")

try:
    from app.btwrap import run_backtest as RUN_WRAP
except Exception:
    RUN_WRAP = None

try:
    from backtest import run_backtest as RUN_RAW
except Exception:
    RUN_RAW = None

st.set_page_config(page_title="Dala Trader – Portfolio (Profiler)", page_icon="🧺", layout="wide")
st.title("🧺 Portfolio – Profiler")

# ---------- Profilhantering ----------
PROF_DIR = Path("/srv/trader/app/profiles") if Path("/srv/trader/app/profiles").exists() else Path("profiles")

def read_profiles() -> Tuple[List[str], Dict[str, List[dict]]]:
    """Läs alla profiler från profiles/*.json och returnera (tickers, map[ticker->list[profile]])"""
    tickers: List[str] = []
    pmap: Dict[str, List[dict]] = {}
    if not PROF_DIR.exists():
        return tickers, pmap
    files = sorted(PROF_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            arr = data.get("profiles", [])
            if not isinstance(arr, list): 
                continue
            for prof in arr:
                if not isinstance(prof, dict):
                    continue
                t = prof.get("ticker") or prof.get("Ticker") or ""
                t = str(t).strip()
                if not t:
                    continue
                pmap.setdefault(t, []).append(prof)
        except Exception:
            # håll robust – bara hoppa över trasiga filer
            continue
    tickers = sorted(pmap.keys())
    return tickers, pmap

def pick_profile(profs: List[dict], flavor: str) -> Optional[dict]:
    """Välj profil efter flavor: conservative/balanced/aggressive, case-insensitive."""
    if not profs:
        return None
    f = flavor.lower()
    # träffa på namn
    for p in profs:
        n = str(p.get("name","")).lower()
        if f in n:
            return p
    # fallback: försök index 0/1/2 enligt ordning (conservative/balanced/aggressive)
    idx = {"conservative":0, "balanced":1, "aggressive":2}.get(f, 0)
    if 0 <= idx < len(profs):
        return profs[idx]
    return profs[0]

# ---------- Hjälpare för BH/kurvor ----------
def _ensure_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for c in ["Adj Close","adj_close","Close","close","c"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty:
                s.index = pd.to_datetime(df.index)
                s = s.sort_index()
                return s
    # OHLC?
    if all(x in df.columns for x in ("open","high","low","close")):
        s = pd.to_numeric(df["close"], errors="coerce").dropna()
        if not s.empty:
            s.index = pd.to_datetime(df.index)
            return s.sort_index()
    if all(x in df.columns for x in ("Open","High","Low","Close")):
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if not s.empty:
            s.index = pd.to_datetime(df.index)
            return s.sort_index()
    return None

def buyhold_equity_from_price(s: pd.Series) -> pd.Series:
    s = s.astype(float).dropna().sort_index()
    eq = s / float(s.iloc[0])
    eq.name = "Buy&Hold"
    return eq

def _series_from_result(res: dict) -> Optional[pd.Series]:
    """Plocka ut equity-serie ur bt-resultat (olika fältnamn tolereras)."""
    if not isinstance(res, dict):
        return None
    cand = res.get("equity") or res.get("equity_curve") or res.get("portfolio") or res.get("balance") or res.get("cumret")
    if cand is None:
        return None
    try:
        if isinstance(cand, dict):
            s = pd.Series(cand)
        elif isinstance(cand, list) and cand and isinstance(cand[0], dict):
            kd = next((k for k in ["date","Date","ts","timestamp"] if k in cand[0]), None)
            kv = next((k for k in ["equity","value","val","y","close","Close"] if k in cand[0]), None)
            if kd and kv:
                idx = pd.to_datetime([x[kd] for x in cand])
                vals = [x[kv] for x in cand]
                s = pd.Series(vals, index=idx)
            else:
                s = pd.Series(cand)
        else:
            s = pd.Series(cand)
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        if len(s)>0 and float(s.iloc[0])!=0.0:
            s = s / float(s.iloc[0])
        return s.rename("Strategy")
    except Exception:
        return None

def run_strategy_or_bh(ticker: str, p: dict) -> Tuple[pd.Series, pd.Series]:
    """För en ticker: försök strategi via RUN_WRAP/RUN_RAW; fallback BH; return (equity_strategy_or_bh, bh_always)"""
    # Hämta data (behövs för BH och ev. RUN_RAW)
    if GET is None:
        raise RuntimeError("Ingen dataladdare (GET) tillgänglig.")
    try:
        df = GET(ticker, start=p.get("from_date"), end=p.get("to_date"))
    except TypeError:
        df = GET(ticker, p.get("from_date"), p.get("to_date"))
    s_close = _ensure_close_series(df)
    if s_close is None or s_close.empty:
        raise ValueError(f"Börsdata returnerade ingen Close-serie för {ticker}.")
    bh = buyhold_equity_from_price(s_close).rename(f"{ticker} · B&H")

    # Försök btwrap
    if RUN_WRAP is not None:
        try:
            res = RUN_WRAP(p={"ticker": ticker, "params": dict(p)})
            s = _series_from_result(res)
            if s is not None and not s.empty:
                return s.rename(f"{ticker} · Strat"), bh
        except Exception:
            try:
                res = RUN_WRAP(ticker, dict(p))
                s = _series_from_result(res)
                if s is not None and not s.empty:
                    return s.rename(f"{ticker} · Strat"), bh
            except Exception:
                pass

    # Fallback raw-backtest om finns
    if RUN_RAW is not None:
        try:
            res = RUN_RAW(df, dict(p))
            s = _series_from_result(res if isinstance(res, dict) else {"equity": res})
            if s is not None and not s.empty:
                return s.rename(f"{ticker} · Strat"), bh
        except Exception:
            pass

    # Sista utväg: endast BH
    return bh.rename(f"{ticker} · B&H"), bh

# ---------- UI ----------
st.caption("Läser universum från **profiles/**. Använder Börsdata via samma loader som Backtest MIN. Ingen CSV.")

tickers_all, prof_map = read_profiles()

with st.sidebar:
    st.header("Universum & period")
    use_auto_universe = st.checkbox("Använd alla tickers i profilkatalogen", value=True)
    custom = st.multiselect("Eller välj manuellt", options=tickers_all, default=tickers_all, disabled=use_auto_universe)
    sel = tickers_all if use_auto_universe else custom

    flavor = st.radio("Profil-variant", ["conservative","balanced","aggressive"], index=1, horizontal=True)

    # Period (hämtas in i prof-parametrar)
    from_date = st.text_input("Från (YYYY-MM-DD)", value="2020-10-01")
    to_date   = st.text_input("Till (YYYY-MM-DD)",  value=date.today().isoformat())

    st.markdown("---")
    go = st.button("🚀 Bygg portfölj", type="primary", use_container_width=True)

if not sel:
    st.info("Ingen ticker hittad – lägg till profiler i `profiles/` eller avmarkera auto-universum.")
    st.stop()

# ---------- Körning ----------
if go:
    curves: Dict[str, pd.Series] = {}
    bh_curves: Dict[str, pd.Series] = {}

    for t in sel:
        profs = prof_map.get(t, [])
        prof  = pick_profile(profs, flavor) or {}
        params = dict(prof.get("params", {}))
        # tvinga period från UI att gälla (Backtest MIN gör liknande)
        params["from_date"] = from_date
        params["to_date"]   = to_date

        try:
            strat, bh = run_strategy_or_bh(t, params)
            curves[t]   = strat
            bh_curves[t]= bh
        except Exception as e:
            st.warning(f"{t}: {e}")

    if not curves:
        st.error("Hittade inga kurvor att rita (Börsdata eller motor returnerade inget).")
        st.stop()

    # Align & equal-weight
    idx = sorted(set().union(*[s.index for s in curves.values()]))
    dfS = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in curves.items()}).dropna(how="all")
    dfB = pd.DataFrame({k: v.reindex(idx).ffill() for k,v in bh_curves.items()}).dropna(how="all")

    # Normalisera & EW
    if not dfS.empty:
        portS = dfS.mean(axis=1).rename("Portfölj · Strat")
    else:
        portS = None
    if not dfB.empty:
        portB = dfB.mean(axis=1).rename("Portfölj · B&H")
    else:
        portB = None

    st.markdown("### Kurvor")
    # Portfolio först
    if portS is not None:
        st.line_chart(portS, width='stretch', height=300)
    if portB is not None:
        st.line_chart(portB, width='stretch', height=300)

    # Enskilda (kort)
    with st.expander("Visa enskilda tickers"):
        if not dfS.empty:
            st.line_chart(dfS, width='stretch', height=300)
        if not dfB.empty:
            st.line_chart(dfB, width='stretch', height=300)

else:
    st.info("Välj universum och klicka **🚀 Bygg portfölj**.")
PY

# 3) Snabb syntaxtest
. .venv/bin/activate 2>/dev/null || true
python -m py_compile pages/2_Portfolio.py && echo "Syntax OK"
# 4) Starta om tjänsten och visa färska loggar
sudo systemctl start trader-ui.service
sudo journalctl -u trader-ui.service -n 60 --no-pager
cd /srv/trader/app
. .venv/bin/activate 2>/dev/null || true
# VÄLJ vilken variant du vill jämföra
export TICKER="YUBICO"                 # din enstaka ticker i universum
export FLAVOR="balanced"               # conservative | balanced | aggressive
python - <<'PY'
import os, json, pandas as pd
from pathlib import Path
from app.data_providers import get_ohlcv as GET
try:
    from app.btwrap import run_backtest as RUN_WRAP
except Exception:
    RUN_WRAP=None

T = os.environ.get("TICKER","YUBICO")
FL = os.environ.get("FLAVOR","balanced").lower()

# --- hämta profil (samma som Optimizer sparar) ---
profs=[]
for fp in sorted(Path("profiles").glob("*.json")):
    try:
        data=json.loads(fp.read_text(encoding="utf-8"))
        for pr in data.get("profiles",[]):
            if str(pr.get("ticker","")).strip().lower()==T.lower():
                profs.append(pr)
    except Exception:
        pass

def pick(ps, flavor):
    for p in ps:
        if flavor in p.get("name","").lower(): return p
    idx={"conservative":0,"balanced":1,"aggressive":2}.get(flavor,0)
    return ps[idx] if ps else None

P = pick(profs, FL); assert P, f"Ingen profil för {T} ({FL})"
params = dict(P.get("params",{}))
fd, td = params.get("from_date"), params.get("to_date")
print(f"\nProfil: {P.get('name')}  period {fd} → {td}")
m = P.get("metrics",{})
if m:
    exp = 1.0 + float(m.get("TotalReturn", 0.0))
    print(f"Optimizer-facit (1+TotalReturn): {exp:.4f}×  | FinalEquity≈ {float(m.get('FinalEquity',0.0)):.0f}")

# --- Börsdata + buy&hold ---
try:
    df = GET(T, start=fd, end=td)
except TypeError:
    df = GET(T, fd, td)

def close(df):
    for c in ("Adj Close","adj_close","Close","close","c"):
        if c in df.columns: 
            s = pd.to_numeric(df[c],errors="coerce").dropna()
            s.index = pd.to_datetime(df.index); return s.sort_index()
    return None

s = close(df); assert s is not None and len(s)>5, "Ingen Close-serie"
bh = (s/s.iloc[0]).rename("BH")

# --- strategi via btwrap (samma som UI ska göra) ---
eq=None
if RUN_WRAP:
    try:
        res = RUN_WRAP(p={"ticker":T,"params":params})
    except TypeError:
        res = RUN_WRAP(T, params)
    cand = res.get("equity") if isinstance(res,dict) else res
    try:
        eq = pd.Series(cand).sort_index()
        eq = pd.to_numeric(eq, errors="coerce").dropna()
        if eq.iloc[0]!=0: eq = eq/eq.iloc[0]
        eq.name="STRAT"
    except Exception:
        eq=None

print("\nBH first/last:     {:.6f} → {:.6f}".format(float(bh.iloc[0]), float(bh.iloc[-1])))
if eq is not None:
    print(  "STRAT first/last:  {:.6f} → {:.6f}".format(float(eq.iloc[0]), float(eq.iloc[-1])))
else:
    print("STRAT saknas (btwrap gav inget) – då ska portföljsidan visa BH=STRAT.")

PY

# På servern
sudo mkdir -p /root/backups
cd /srv/trader
# App + profiler (exkludera .venv)
sudo tar --exclude='app/.venv' -czf /root/backups/dalatrader_app_$(date +%F_%H%M%S).tgz app
# Systemdelar som berörs
sudo tar -czf /root/backups/dalatrader_sys_$(date +%F_%H%M%S).tgz   /etc/caddy/Caddyfile   /etc/systemd/system/trader-ui.service
ls -lh /root/backups
sudo ls -lh /root/backups
# Kopiera alla tgz-filer till din hemkatalog
sudo cp /root/backups/*.tgz ~claes/ 2>/dev/null || true
# Sätt rätt ägarskap så du kan läsa dem
sudo chown claes: ~claes/*.tgz
# Verifiera
ls -lh ~claes/*.tgz
cd /srv/trader/app
cat > .gitignore <<'EOF'
# venv & python cache
.venv/
**/__pycache__/
*.py[cod]
*.so

# lokala/temporära filer
*.log
*.tmp
*.DS_Store
.ipynb_checkpoints/

# hemligheter & miljö
.env
.secrets
.streamlit/secrets.toml
app/keys*
app/*secret*
app/*token*
**/*apikey*
**/*api_key*

# data/output/cacher (behövs ej i repo)
trader/outputs/
data/
cache/
EOF

grep -RIn --color -E 'BORS|borsdata.*(key|token)|API[_-]?KEY' app pages || true
