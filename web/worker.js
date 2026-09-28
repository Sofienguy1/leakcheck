// Runs Python (Pyodide) off the main thread so the page stays responsive during big checks.
// Installs the published leakcheck-ml package from PyPI: the site runs exactly what `pip install` gives you.
// Files are written to Pyodide's in-memory filesystem; nothing is sent to any server.

// A module worker (see app.js): `import` is more widely allowed than classic importScripts() from a CDN.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const LEAKCHECK = "leakcheck-ml==0.3.0";

const HELPERS = `
import json, time
from dataclasses import asdict

import pandas as pd
import leakcheck
from leakcheck.checks import run_all, _group_columns, _detect_time_columns
from leakcheck.cli import load as _load

frames = {}

def load_frame(name, path):
    frames[name] = df = _load(path)
    return json.dumps({"rows": len(df), "cols": df.shape[1], "columns": [str(c) for c in df.columns]})

def suggest():
    train, test = frames["train"], frames["test"]
    only_in_train = [c for c in train.columns if c not in test.columns]
    target = only_in_train[0] if len(only_in_train) == 1 else next(
        (c for c in train.columns if str(c).lower() in {"target", "label", "y", "class", "outcome"}), None)
    return json.dumps({"target": target,
                       "groups": _group_columns(train, test, target),
                       "times": _detect_time_columns(train, test)}, default=str)

def run(target, group_col, time_col, similarity):
    start = time.perf_counter()
    findings = run_all(frames["train"], frames["test"], target=target or None, time_col=time_col or None,
                       similarity=similarity, group_col=group_col or None)
    return json.dumps({"findings": [asdict(f) for f in findings],
                       "seconds": round(time.perf_counter() - start, 2),
                       "train": list(frames["train"].shape), "test": list(frames["test"].shape)}, default=str)

versions = json.dumps({"leakcheck": leakcheck.__version__, "pandas": pd.__version__})
`;

let py;
const post = (msg) => self.postMessage(msg);

const ready = (async () => {
  post({ type: "status", text: "Downloading Python…" });
  py = await loadPyodide({ indexURL: PYODIDE });
  post({ type: "status", text: "Loading pandas…" });
  await py.loadPackage(["micropip", "pandas", "numpy"]);
  post({ type: "status", text: "Installing leakcheck-ml…" });
  await py.pyimport("micropip").install(LEAKCHECK);
  py.runPython(HELPERS);
  post({ type: "ready", versions: JSON.parse(py.globals.get("versions")) });
})().catch((err) => post({ type: "fatal", error: String(err) }));

self.onmessage = async ({ data: msg }) => {
  await ready;
  try {
    if (msg.type === "load") {
      const ext = /\.(parquet|tsv|tab)$/i.exec(msg.filename)?.[1]?.toLowerCase() ?? "csv";
      if (ext === "parquet") await py.loadPackage("pyarrow");
      const path = `/tmp/${msg.name}.${ext}`;
      py.FS.writeFile(path, new Uint8Array(msg.bytes));
      const info = JSON.parse(py.globals.get("load_frame")(msg.name, path));
      post({ type: "loaded", name: msg.name, filename: msg.filename, ...info });
    } else if (msg.type === "suggest") {
      post({ type: "suggested", ...JSON.parse(py.globals.get("suggest")()) });
    } else if (msg.type === "run") {
      const out = py.globals.get("run")(msg.target, msg.group, msg.time, msg.similarity);
      post({ type: "result", ...JSON.parse(out) });
    }
  } catch (err) {
    // Python tracebacks are long; the last line carries the actual error.
    const lines = String(err.message ?? err).trim().split("\n");
    post({ type: "error", request: msg.type, name: msg.name, error: lines[lines.length - 1] });
  }
};
