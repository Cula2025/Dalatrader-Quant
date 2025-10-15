import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Nightly Optimizer – Status", layout="wide")
st.title("🌙 Nightly Optimizer – Status")

status_path = Path("runtime/nightly_status.json")
if not status_path.exists():
    st.info("Ingen status ännu. Fyll i tickers i config/tickers_nightly.txt och starta nattjobbet.")
else:
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Kunde inte läsa status: {e}")
    else:
        st.json(data)
        state = data.get("state")
        if state == "running":
            st.success(f"Kör: {data.get('current')}  ({data.get('done')}/{data.get('total')})")
        elif state == "starting":
            st.warning("Startar…")
        elif state == "done":
            st.balloons()
            st.success(f"Klar. Output: {data.get('out')}")
        else:
            st.info(f"Läge: {state}")

st.caption("Status uppdateras av tools/nightly_opt_job.sh. Logg per körning finns i results/opt_nightly_*/run.log")
