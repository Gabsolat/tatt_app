"""
Smoke test that executes every dashboard page with Streamlit and Plotly replaced by recording stubs.
It exercises all NumPy/data/model logic of app.py on the real result files. It does not check the
visual output or Streamlit/Plotly API usage; run `streamlit run app.py` for that.
    python tests/smoke_test.py
"""
import io, os, runpy, sys, traceback
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
CSV = None


class Ctx:
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(self._st, n)


class StopApp(Exception): pass


class SS(dict):
    def __getattr__(self, k): return self[k]
    def __setattr__(self, k, v): self[k] = v


def make_st(page, submit=True, upload=None):
    st = MagicMock(name="streamlit"); st.session_state = SS(); calls = {"n": 0}
    st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.cache_data = st.cache_resource
    st.sidebar = Ctx(st)
    st.radio = lambda label, options, **k: page
    def num(label, mn=0, mx=1, value=None, step=None, **k):
        key = k.get("key")
        if key and key in st.session_state: return st.session_state[key]
        return mn if value is None else value
    st.number_input = num
    st.slider = lambda label, mn, mx, value=None, step=None, **k: (mn if value is None else value)
    def sel(label, options, **k):
        key = k.get("key")
        if key and key in st.session_state: return st.session_state[key]
        return list(options)[k.get("index", 0)]
    st.selectbox = sel
    st.multiselect = lambda label, options, default=None, **k: list(default or [])
    st.checkbox = lambda label, value=False, **k: (st.session_state.get(k["key"], value) if k.get("key") else value)
    st.columns = lambda spec, **k: [Ctx(st) for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda names: [Ctx(st) for _ in names]
    st.form = lambda *a, **k: Ctx(st)
    st.form_submit_button = lambda *a, **k: submit
    st.file_uploader = lambda *a, **k: upload
    def stop(): raise StopApp()
    st.stop = stop
    return st


class FakeFile(io.BytesIO): pass


def run(page, **kw):
    st = make_st(page, **kw)
    go = MagicMock(); sub = MagicMock()
    mods = {"streamlit": st, "plotly": MagicMock(), "plotly.graph_objects": go, "plotly.subplots": sub}
    old = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
        return "ok"
    except StopApp:
        return "stopped (expected only if files are missing)"
    except Exception:
        return "ERROR\n" + traceback.format_exc()
    finally:
        for k, v in old.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


if __name__ == "__main__":
    pages = ["Overview", "Cohort and data", "Preprocessing", "Model architecture", "Training", "Evaluation", "Interpretability", "Patient simulator", "Methods and limitations"]
    bad = 0
    for p in pages:
        r = run(p); print(f"{p:26s} {r.splitlines()[0]}"); 
        if r.startswith("ERROR"): bad += 1; print(r)
    # uploaded-CSV path of the simulator
    import numpy as np, pandas as pd, pickle, data as D
    P = pickle.load(open(os.path.join(ROOT, "results", "sample_patients.pkl"), "rb")); ex = P[0]; k = (ex["time"] > 5) & (ex["time"] < 60)
    df = pd.DataFrame({"variable": [D.VAR_NAMES[i] for i in ex["var"][k]], "value": ex["val"][k], "time_h": ex["time"][k]})
    r = run("Patient simulator", upload=FakeFile(df.to_csv(index=False).encode())); print(f"{'Simulator (CSV upload)':26s} {r.splitlines()[0]}")
    if r.startswith("ERROR"): bad += 1; print(r)
    print("FAILED" if bad else "ALL PAGES EXECUTED WITHOUT PYTHON ERRORS"); sys.exit(1 if bad else 0)
