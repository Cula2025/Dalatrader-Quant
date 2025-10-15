
# --- AUTOFIX (global fallback för _load_df_any_alias) ---
try:
    _load_df_any_alias  # finns redan?
except NameError:
    try:
        from app.data import load_df_any as _load_df_any_alias
    except Exception:
        def _load_df_any_alias(*args, **kwargs):
            try:
                import pandas as pd
                return kwargs.get("df", pd.DataFrame())  # pass-through om df= gavs
            except Exception:
                return kwargs.get("df")
# --- /AUTOFIX ---
