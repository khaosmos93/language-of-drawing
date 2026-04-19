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
const status = (msg) => { $("status").textContent = msg; };

let pyodide = null;
let fields = null;       // (9,H,W) Float32 — cached after first compute
let rawCbCr = null;      // saved across reloads
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
  status("Loading scientific packages (numpy, scipy, scikit-image, pywavelets, pillow, matplotlib)…");
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
import sys
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
import numpy as np, io
ORDER = R.ORDER
REGISTRY = R.REGISTRY
print("[pyodide] core ready, ORDER=", ORDER)
`);
  status("Ready. Pick an image or use the synthetic demo.");
}

// --- canvas helpers ---
function drawImageBytesTo(canvasId, bytes) {
  return new Promise((resolve) => {
    const blob = new Blob([bytes], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const c = $(canvasId);
      c.width = img.width; c.height = img.height;
      c.getContext("2d").drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.src = url;
  });
}

async function imageBytesFromFile(file) {
  const buf = await file.arrayBuffer();
  return new Uint8Array(buf);
}

// --- pipeline calls ---
async function loadImageBytes(bytes) {
  status("Computing 9 mathematical representations…");
  pyodide.FS.writeFile("/tmp/in.bin", bytes);
  await pyodide.runPythonAsync(`
with open("/tmp/in.bin","rb") as f:
    _b = f.read()
Y, CbCr = load_image(_b)
fields_np, raw = compute_fields(Y)
_FIELDS = fields_np
_RAW = raw
_Y = Y
_CBCR = CbCr
_in_png = png_bytes((Y*255+0.5).astype("uint8"))
_rep_pngs = {n: png_bytes(REGISTRY[n].visualize(raw[n])) for n in ORDER}
`);
  const inPng = pyodide.globals.get("_in_png").toJs({ create_proxies: false });
  await drawImageBytesTo("inCanvas", inPng);
  const repMap = pyodide.globals.get("_rep_pngs").toJs({ dict_converter: Object.fromEntries, create_proxies: false });
  for (const name of ORDER) {
    await drawImageBytesTo(`rep-${name}`, repMap[name]);
  }
  status("Computing reconstruction…");
  await runReconstruction();
  status("Done.");
}

async function runReconstruction() {
  const strategy = $("strategy").value;
  const beta = parseFloat($("beta").value);
  pyodide.globals.set("_w", weights);
  pyodide.globals.set("_t", toggles);
  pyodide.globals.set("_strategy", strategy);
  pyodide.globals.set("_beta", beta);
  await pyodide.runPythonAsync(`
import numpy as np
w = np.asarray(list(_w), dtype=np.float32)
t = np.asarray(list(_t), dtype=np.float32)
Y_hat = reconstruct(_FIELDS, strategy=_strategy, weights=w, toggles=t, beta=float(_beta))
_rgb_bytes = png_bytes(reattach_chroma(Y_hat, _CBCR))
_Y_HAT = Y_hat
`);
  const png = pyodide.globals.get("_rgb_bytes").toJs({ create_proxies: false });
  await drawImageBytesTo("outCanvas", png);
}

async function runSymbolic() {
  const basis = $("basis").value;
  const degree = parseInt($("degree").value, 10);
  pyodide.globals.set("_basis", basis);
  pyodide.globals.set("_degree", degree);
  status("Fitting symbolic basis expansion…");
  await pyodide.runPythonAsync(`
fit = fit_symbolic(_Y_HAT, basis=_basis, degree=int(_degree), max_k=int(_degree))
_expr = expression_string(fit, top_n=24)
_sym_png = png_bytes(to_uint8(fit["approx"]))
`);
  $("exprOut").textContent = pyodide.globals.get("_expr");
  const png = pyodide.globals.get("_sym_png").toJs({ create_proxies: false });
  await drawImageBytesTo("symCanvas", png);
  status("Symbolic expression updated.");
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
      if (fieldsReady()) await runReconstruction();
    }, 60));
  });
  container.querySelectorAll(".tog").forEach((el) => {
    el.addEventListener("change", async (e) => {
      const i = parseInt(e.target.dataset.i, 10);
      toggles[i] = e.target.checked ? 1.0 : 0.0;
      if (fieldsReady()) await runReconstruction();
    });
  });
}

function fieldsReady() {
  return pyodide && pyodide.globals.get("_FIELDS") !== undefined;
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
  $("strategy").addEventListener("change", () => fieldsReady() && runReconstruction());
  $("beta").addEventListener("input", debounce((e) => {
    $("betaV").textContent = parseFloat(e.target.value).toFixed(1);
    if (fieldsReady()) runReconstruction();
  }, 60));
  $("perturb").addEventListener("click", async () => {
    weights = weights.map((w) => Math.max(0, Math.min(2, w + (Math.random() - 0.5) * 0.4)));
    refreshWeightDisplays();
    if (fieldsReady()) await runReconstruction();
  });
  $("reset").addEventListener("click", async () => {
    weights = ORDER.map(() => 1.0 / ORDER.length);
    toggles = ORDER.map(() => 1.0);
    refreshWeightDisplays();
    if (fieldsReady()) await runReconstruction();
  });
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const bytes = await imageBytesFromFile(f);
    await loadImageBytes(bytes);
  });
  $("useDemo").addEventListener("click", async () => {
    status("Generating synthetic demo image…");
    await pyodide.runPythonAsync(`
generate_demo("/tmp/demo.png", size=192)
with open("/tmp/demo.png","rb") as f:
    _bb = f.read()
`);
    const bytes = pyodide.globals.get("_bb").toJs({ create_proxies: false });
    await loadImageBytes(bytes);
  });
  $("symbolic").addEventListener("click", runSymbolic);
}

(async function main() {
  buildRepControls();
  bindGlobalControls();
  try {
    await bootPyodide();
  } catch (e) {
    status(`Pyodide init failed: ${e.message}`);
    console.error(e);
  }
})();
