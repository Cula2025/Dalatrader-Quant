# Trader 2.0 – Plan
Ordning: Optimizer → Backtest → Portfolio.

**Milstolpar**
1) Data-adapter (Börsdata ONLY) + deterministiska tester
2) Optimizer CLI (grid/bruteforce) som sparar 3 profiler/fil: conservative/balanced/aggressive
3) Backtest (samma motor som Optimizer använder) med tydligt equity-kontrakt
4) Portfolio som läser profiler → bygger universum → jämför BH vs STRAT
5) Reproducerbarhet: seed, version i fil, checksumma

**Artefakter**
- `profiles/*.json` med exakt struktur:
  {
    "profiles": [
      {"name": "...", "ticker": "TICK", "params": {...}, "metrics": {...}},
      {"name": "...", ...},
      {"name": "...", ...}
    ],
    "meta": {"engine": "v2", "seed": 123, "created_at": "YYYY-MM-DDTHH:MM"}
  }
