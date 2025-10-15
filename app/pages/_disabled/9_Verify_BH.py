import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.bh_std import calc_bh

st.set_page_config(page_title="Verifiera Buy&Hold", page_icon="🔍", layout="wide")
st.title("🔍 Verifiera Buy & Hold mot profiler")

base = Path("/srv/trader/app/profiles") if Path("/srv/trader/app/profiles").exists() else Path("profiles")
files = sorted(base.glob("*.json"))

st.caption("Regel: start = första handelsdag ≥ from_date, slut = sista handelsdag ≤ to_date. BH = Close[slut]/Close[start]−1.")
run = st.button("Kör kontroll", type="primary")

if run:
    rows = []
    for pf in files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            for prof in (data.get("profiles") or []):
                name   = prof.get("name","?")
                ticker = prof.get("ticker") or (prof.get("params") or {}).get("ticker","")
                prms   = prof.get("params") or {}
                met    = prof.get("metrics") or {}
                start  = prms.get("from_date")
                end    = prms.get("to_date")
                try:
                    bh = calc_bh(ticker, start, end)
                    bh_calc  = bh["bh"]
                    bh_facit = float(met.get("BuyHold")) if met and "BuyHold" in met else None
                    ok = (bh_facit is not None) and (abs(bh_calc - bh_facit) <= max(1e-6, (abs(bh_calc)+abs(bh_facit))*1e-9))
                    rows.append({
                        "fil": pf.name, "profil": name, "ticker": ticker,
                        "BH(calc)": round(bh_calc, 6),
                        "BH(facit)": (None if bh_facit is None else round(bh_facit,6)),
                        "PASS": ok,
                        "from→resolved": f"{start} → {bh['first_date']}",
                        "to→resolved":   f"{end} → {bh['last_date']}",
                        "first_px": round(bh["first_price"], 6),
                        "last_px":  round(bh["last_price"],  6),
                        "rows": bh["rows_used"],
                    })
                except Exception as e:
                    rows.append({
                        "fil": pf.name, "profil": name, "ticker": ticker,
                        "BH(calc)": None, "BH(facit)": met.get("BuyHold"),
                        "PASS": False, "from→resolved": str(start), "to→resolved": str(end),
                        "first_px": None, "last_px": None, "rows": None,
                        "error": f"{type(e).__name__}: {e}",
                    })
        except Exception as e:
            st.error(f"Kunde inte läsa {pf.name}: {type(e).__name__}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        passed = int(df["PASS"].sum())
        total  = len(df)
        st.success(f"PASS: {passed}/{total}") if passed == total else st.warning(f"PASS: {passed}/{total}")
    else:
        st.info("Inga profiler hittades.")
else:
    st.info("Klicka på **Kör kontroll** för att verifiera alla profiler.")
