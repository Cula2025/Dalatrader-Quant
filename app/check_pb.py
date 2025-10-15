import importlib, inspect
import app.portfolio_backtest as PB

print("Loaded file:", getattr(PB, "__file__", "?"))
print("Version:", getattr(PB, "__version__", "?"))
print("run_portfolio_backtest signature:", inspect.signature(PB.run_portfolio_backtest))
