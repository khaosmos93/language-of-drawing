// Pyodide-driven UI. Runs the same Python `core/` package in the browser.
// On GitHub Pages, the CI workflow copies `core/` into `web/core/` so the
// fetch URLs below resolve.

const REP_FILES = [
  "__init__.py", "normalize.py", "io_utils.py", "pipeline.py", "symbolic.py",
  "demo_image.py",
  "representations/__init__.py", "representations/base.py",
  "representations/fourier.py", "representations/wavelet.py",
  "representations/gradient.py", "representations/levelset.py",
  "representations/graph.py", "representations/pde.py",
  "representations/probability.py", "representations/fractal.py",
  "representations/manifold.py",
  "reconstruction/__init__.py", "reconstruction/linear.py",
  "reconstruction/nonlinear.py", "reconstruction/pde_fusion.py",
];

// Order MUST match core/representations/__init__.py::ORDER
const ORDER = [
  "fourier", "wavelet", "gradient", "levelset", "graph",
  "pde", "probability", "fractal", "manifold",
];
const EQS = {
  fourier:    "F(u,v) = ℱ{I}",
  wavelet:    "W(s,x,y) = ⟨I,ψ_{s,x,y}⟩",
  gradient:   "∇I = (∂ₓI, ∂ᵧI)",
  levelset:   "{φ = c} are level sets of I",
  graph:      "L = D − W ; Lv = λv",
  pde:        "∇²I = f(x,y)",
  probability:"p(I), H_local",
  fractal:    "D(x,y) = −log N(ε)/log ε",
  manifold:   "g_ij = G_σ * ∂ᵢI ∂ⱼI",
};

const $ = (id) => document.getElementById(id);
const status = (msg, level = "info") => {
  const el = $("status");
  el.textContent = msg;
  el.style.color = level === "error" ? "#ef4444" : level === "ok" ? "#6ee7b7" : "#f59e0b";
  console.log(`[status:${level}]`, msg);
};
const err = (e) => {
  const msg = (e && (e.message || String(e))) || "unknown error";
  status(msg, "error");
  console.error(e);
};

let pyodide = null;
let haveFields = false;
let weights = ORDER.map(() => 1.0 / ORDER.length);
let toggles = ORDER.map(() => 1.0);

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
  return r.text();
}

async function bootPyodide() {
  status("Loading Pyodide runtime…");
  pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
  status("Loading numpy / scipy / scikit-image / pywavelets / pillow / matplotlib…");
  await pyodide.loadPackage([
    "numpy", "scipy", "scikit-image", "pywavelets", "pillow", "matplotlib",
  ]);
  status("Mounting core/ Python package…");
  pyodide.FS.mkdirTree("/home/pyodide/lod/core/representations");
  pyodide.FS.mkdirTree("/home/pyodide/lod/core/reconstruction");
  for (const rel of REP_FILES) {
    const src = await fetchText(`core/${rel}`);
    pyodide.FS.writeFile(`/home/pyodide/lod/core/${rel}`, src);
  }
  await pyodide.runPythonAsync(`
import sys, traceback
if "/home/pyodide/lod" not in sys.path:
    sys.path.insert(0, "/home/pyodide/lod")
import core
import core.representations as R
import core.reconstruction as C
from core.io_utils import load_image, reattach_chroma, png_bytes
from core.normalize import to_uint8
from core.pipeline import compute_fields, reconstruct
from core.symbolic import fit_symbolic, expression_string
from core.demo_image import generate_demo
import numpy as np
ORDER = R.ORDER
REGISTRY = R.REGISTRY
print("[pyodide] ready. ORDER=", ORDER)
`);
  status("Ready. Pick an image or use the synthetic demo.", "ok");
}

// --- canvas helpers ---
function drawBytesTo(canvasId, bytes) {
  return new Promise((resolve, reject) => {
    try {
      const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      const blob = new Blob([u8], { type: "image/png" });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const c = $(canvasId);
        if (!c) { URL.revokeObjectURL(url); return resolve(); }
        c.width = img.width; c.height = img.height;
        c.getContext("2d").drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        resolve();
      };
      img.onerror = (e) => { URL.revokeObjectURL(url); reject(new Error("image decode failed")); };
      img.src = url;
    } catch (e) { reject(e); }
  });
}

async function fileToBytes(file) {
  const buf = await file.arrayBuffer();
  return new Uint8Array(buf);
}

function pyBytesToU8(name) {
  // Pyodide returns Python bytes as a PyProxy; explicitly convert to a JS Uint8Array.
  const proxy = pyodide.globals.get(name);
  try {
    if (proxy && typeof proxy.toJs === "function") {
      return proxy.toJs();   // bytes -> Uint8Array
    }
    if (proxy && proxy.buffer) return new Uint8Array(proxy.buffer);
    return new Uint8Array(proxy);
  } finally {
    if (proxy && typeof proxy.destroy === "function") proxy.destroy();
  }
}

async function loadImageBytes(bytes) {
  haveFields = false;
  try {
    status("Decoding image and computing 9 mathematical representations…");
    pyodide.FS.writeFile("/tmp/in.bin", bytes);
    await pyodide.runPythonAsync(`
try:
    with open("/tmp/in.bin","rb") as f:
        _b = f.read()
    Y, CbCr = load_image(_b)
    fields_np, raw = compute_fields(Y)
    _FIELDS = fields_np; _RAW = raw; _Y = Y; _CBCR = CbCr
    _in_png = png_bytes((Y*255+0.5).astype("uint8"))
    _rep_pngs_list = [png_bytes(REGISTRY[n].visualize(raw[n])) for n in ORDER]
    _err = None
except Exception as _e:
    import traceback
    _err = traceback.format_exc()
`);
    const errMsg = pyodide.globals.get("_err");
    if (errMsg) { throw new Error(errMsg.toString()); }

    await drawBytesTo("inCanvas", pyBytesToU8("_in_png"));
    const listProxy = pyodide.globals.get("_rep_pngs_list");
    const n = listProxy.length;
    for (let i = 0; i < n; i++) {
      const item = listProxy.get(i);
      const u8 = item.toJs();
      await drawBytesTo(`rep-${ORDER[i]}`, u8);
      if (item.destroy) item.destroy();
    }
    if (listProxy.destroy) listProxy.destroy();

    haveFields = true;
    status("Computing reconstruction…");
    await runReconstruction();
    status("Done — adjust sliders to explore.", "ok");
  } catch (e) { err(e); }
}

async function runReconstruction() {
  if (!haveFields) return;
  try {
    const strategy = $("strategy").value;
    const beta = parseFloat($("beta").value);
    pyodide.globals.set("_w", weights);
    pyodide.globals.set("_t", toggles);
    pyodide.globals.set("_strategy", strategy);
    pyodide.globals.set("_beta", beta);
    await pyodide.runPythonAsync(`
try:
    w = np.asarray(list(_w), dtype=np.float32)
    t = np.asarray(list(_t), dtype=np.float32)
    Y_hat = reconstruct(_FIELDS, strategy=_strategy, weights=w, toggles=t, beta=float(_beta))
    _rgb_bytes = png_bytes(reattach_chroma(Y_hat, _CBCR))
    _Y_HAT = Y_hat
    _err = None
except Exception as _e:
    import traceback
    _err = traceback.format_exc()
`);
    const errMsg = pyodide.globals.get("_err");
    if (errMsg) throw new Error(errMsg.toString());
    await drawBytesTo("outCanvas", pyBytesToU8("_rgb_bytes"));
  } catch (e) { err(e); }
}

async function runSymbolic() {
  if (!haveFields) { status("Compute a reconstruction first.", "error"); return; }
  try {
    const basis = $("basis").value;
    const degree = parseInt($("degree").value, 10);
    pyodide.globals.set("_basis", basis);
    pyodide.globals.set("_degree", degree);
    status("Fitting symbolic basis expansion…");
    await pyodide.runPythonAsync(`
try:
    fit = fit_symbolic(_Y_HAT, basis=_basis, degree=int(_degree), max_k=int(_degree))
    _expr = expression_string(fit, top_n=24)
    _sym_png = png_bytes(to_uint8(fit["approx"]))
    _err = None
except Exception as _e:
    import traceback
    _err = traceback.format_exc()
`);
    const errMsg = pyodide.globals.get("_err");
    if (errMsg) throw new Error(errMsg.toString());
    $("exprOut").textContent = pyodide.globals.get("_expr");
    await drawBytesTo("symCanvas", pyBytesToU8("_sym_png"));
    status("Symbolic expression updated.", "ok");
  } catch (e) { err(e); }
}

// --- UI scaffolding ---
function buildRepControls() {
  const container = $("reps");
  container.innerHTML = "";
  ORDER.forEach((name, i) => {
    const card = document.createElement("div");
    card.className = "rep";
    card.innerHTML = `
      <div class="head">
        <label><input type="checkbox" data-i="${i}" class="tog" checked /> ${name}</label>
        <span class="eq">${EQS[name] || ""}</span>
      </div>
      <canvas id="rep-${name}" width="200" height="200"></canvas>
      <div class="w">w<sub>${i + 1}</sub>
        <input type="range" data-i="${i}" class="ws" min="0" max="2" step="0.01" value="${(1 / ORDER.length).toFixed(3)}" />
        <span class="wv" data-i="${i}">${(1 / ORDER.length).toFixed(2)}</span>
      </div>
    `;
    container.appendChild(card);
  });
  container.querySelectorAll(".ws").forEach((el) => {
    el.addEventListener("input", debounce(async (e) => {
      const i = parseInt(e.target.dataset.i, 10);
      weights[i] = parseFloat(e.target.value);
      container.querySelector(`.wv[data-i="${i}"]`).textContent = weights[i].toFixed(2);
      await runReconstruction();
    }, 80));
  });
  container.querySelectorAll(".tog").forEach((el) => {
    el.addEventListener("change", async (e) => {
      const i = parseInt(e.target.dataset.i, 10);
      toggles[i] = e.target.checked ? 1.0 : 0.0;
      await runReconstruction();
    });
  });
}

function debounce(fn, ms) {
  let h = null;
  return (...args) => { clearTimeout(h); h = setTimeout(() => fn(...args), ms); };
}

function refreshWeightDisplays() {
  document.querySelectorAll(".ws").forEach((el) => {
    const i = parseInt(el.dataset.i, 10);
    el.value = weights[i].toFixed(3);
    document.querySelector(`.wv[data-i="${i}"]`).textContent = weights[i].toFixed(2);
  });
  document.querySelectorAll(".tog").forEach((el) => {
    const i = parseInt(el.dataset.i, 10);
    el.checked = toggles[i] > 0;
  });
}

function bindGlobalControls() {
  $("strategy").addEventListener("change", runReconstruction);
  $("beta").addEventListener("input", debounce((e) => {
    $("betaV").textContent = parseFloat(e.target.value).toFixed(1);
    runReconstruction();
  }, 80));
  $("perturb").addEventListener("click", async () => {
    weights = weights.map((w) => Math.max(0, Math.min(2, w + (Math.random() - 0.5) * 0.4)));
    refreshWeightDisplays();
    await runReconstruction();
  });
  $("reset").addEventListener("click", async () => {
    weights = ORDER.map(() => 1.0 / ORDER.length);
    toggles = ORDER.map(() => 1.0);
    refreshWeightDisplays();
    await runReconstruction();
  });
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const bytes = await fileToBytes(f);
    await loadImageBytes(bytes);
  });
  $("useDemo").addEventListener("click", async () => {
    try {
      status("Generating synthetic demo image…");
      await pyodide.runPythonAsync(`
generate_demo("/tmp/demo.png", size=192)
with open("/tmp/demo.png","rb") as f:
    _bb = f.read()
`);
      const u8 = pyBytesToU8("_bb");
      await loadImageBytes(u8);
    } catch (e) { err(e); }
  });
  $("symbolic").addEventListener("click", runSymbolic);

  window.addEventListener("unhandledrejection", (e) => err(e.reason));
  window.addEventListener("error", (e) => err(e.error || e.message));
}

(async function main() {
  buildRepControls();
  bindGlobalControls();
  try {
    await bootPyodide();
  } catch (e) { err(e); }
})();
