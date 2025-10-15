import os, sys, runpy
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Skicka vidare CLI-argumenten till gen_profiles.py
argv = [ "gen_profiles.py", *sys.argv[1:] ]
sys.argv[:] = argv
print(">>> run_gen.py: startar gen_profiles.py med argv:", argv, flush=True)
runpy.run_path(os.path.join(os.path.dirname(__file__), "gen_profiles.py"), run_name="__main__")
