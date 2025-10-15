from __future__ import annotations
import logging, os, datetime as dt
from typing import Optional

_LOG_PATH = os.getenv("TRADER_UI_LOG", "/srv/trader/app/logs/ui.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

_logger: Optional[logging.Logger] = None
def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("ui")
    logger.setLevel(logging.INFO)
    if not any(getattr(h, "baseFilename", None) == os.path.abspath(_LOG_PATH) for h in logger.handlers):
        fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    _logger = logger
    return logger

def log_info(msg: str) -> None: _get_logger().info(msg)
def log_warn(msg: str) -> None: _get_logger().warning(msg)
def log_error(msg: str) -> None: _get_logger().error(msg)
def log_debug(msg: str) -> None: _get_logger().debug(msg)

def df_brief(df, rows: int = 5, cols: int = 8):
    try:
        sub = df.copy()
        if getattr(sub, "columns", None) is not None: sub = sub.iloc[:, :cols]
        if rows: sub = sub.head(rows)
        return sub
    except Exception:
        return df

def setup_debug_ui(st):
    try:
        with st.expander("🔧 Debug", expanded=False):
            st.caption(f"Loggfil: `{_LOG_PATH}`")
            if st.button("Skriv testlogg"):
                log_info(f"Testlogg från UI {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
                st.success("Skrev en rad till loggen.")
    except Exception:
        pass

# Fallback för gamla funktionsnamn: returnera no-op för alla okända attribut
def __getattr__(name):
    def _noop(*a, **k): pass
    return _noop
