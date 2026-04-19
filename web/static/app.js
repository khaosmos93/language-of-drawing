import {
  createSizedCanvas,
  drawImageDataToCanvas,
  drawImageToCanvas,
  fitImageInsideBox,
  sanitizeNumericArray,
  scalarFieldToImageData,
  rgbFieldToImageData,
} from "./canvasUtils.js";
import { CORE_BUNDLE } from "./coreBundle.js";

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

const ORDER = [
  "fourier", "wavelet", "gradient", "levelset", "graph",
  "pde", "probability", "fractal", "manifold",
];

const EQS = {
  fourier: "F(u,v) = ℱ{I}",
  wavelet: "W(s,x,y) = ⟨I,ψ_{s,x,y}⟩",
  gradient: "∇I = (∂ₓI, ∂ᵧI)",
  levelset: "{φ = c} are level sets of I",
  graph: "L = D − W ; Lv = λv",
  pde: "∇²I = f(x,y)",
  probability: "p(I), H_local",
  fractal: "D(x,y) = −log N(ε)/log ε",
  manifold: "g_ij = G_σ * ∂ᵢI ∂ⱼI",
};

const MAX_UPLOAD_MB = 15;
const MAX_PREVIEW_BOX = 360;

const $ = (id) => document.getElementById(id);
const status = (msg, level = "info") => {
  const el = $("status");
  el.textContent = msg;
  el.dataset.level = level;
};

let pyodide = null;
let coreBasePath = "core";
let coreLoadMode = "http";
let weights = ORDER.map(() => 1.0 / ORDER.length);
let toggles = ORDER.map(() => 1.0);
const STAGES = [
  { id: "runtime", label: "1) Pyodide runtime boot" },
  { id: "packages", label: "2) Scientific packages load" },
  { id: "core", label: "3) core/ Python package mount" },
  { id: "image", label: "4) Image decode → grayscale/chroma split" },
  { id: "representations", label: "5) 9 mathematical representations" },
  { id: "reconstruction", label: "6) Weighted reconstruction" },
  { id: "symbolic", label: "7) Symbolic basis approximation (optional)" },
];

function readBaseUrl() {
  if (typeof import.meta !== "undefined" && import.meta?.env?.BASE_URL) {
    return import.meta.env.BASE_URL;
  }
  return `${window.location.origin}/`;
}

const appState = {
  imageLoaded: false,
  imageWidth: 0,
  imageHeight: 0,
  processing: false,
  fatalError: null,
  reconstructionStatus: "idle",
  repDiagnostics: {},
  baseUrl: readBaseUrl(),
  pathname: window.location.pathname,
  stages: Object.fromEntries(STAGES.map((s) => [s.id, "pending"])),
};

function updateDebugPanel() {
  const lines = [
    `baseURL: ${appState.baseUrl}`,
    `pathname: ${appState.pathname}`,
    `corePath: ${coreBasePath}`,
    `coreLoadMode: ${coreLoadMode}`,
    `imageLoaded: ${appState.imageLoaded}`,
    `imageDimensions: ${appState.imageWidth} x ${appState.imageHeight}`,
    `processing: ${appState.processing}`,
    `reconstruction: ${appState.reconstructionStatus}`,
  ];

  ORDER.forEach((name) => {
    const d = appState.repDiagnostics[name];
    if (!d) {
      lines.push(`rep:${name} -> not computed`);
      return;
    }
    if (!d.ok) {
      lines.push(`rep:${name} -> ERROR: ${d.error}`);
      return;
    }
    lines.push(`rep:${name} -> ok min=${d.min.toFixed(4)} max=${d.max.toFixed(4)} nan=${d.nanCount}`);
  });

  if (appState.fatalError) {
    lines.push(`fatalError: ${appState.fatalError}`);
  }
  $("debugOut").textContent = lines.join("\n");
}

function setStage(stageId, value) {
  appState.stages[stageId] = value;
  const el = document.querySelector(`.procStep[data-stage="${stageId}"]`);
  if (!el) return;
  const text = value === "done"
    ? "done"
    : value === "active"
      ? "running"
      : value === "error"
        ? "error"
        : "pending";
  el.dataset.state = value;
  const badge = el.querySelector(".state");
  if (badge) badge.textContent = text;
}

function buildProcessChecklist() {
  const container = $("processList");
  if (!container) return;
  container.innerHTML = "";
  STAGES.forEach((stage) => {
    const li = document.createElement("li");
    li.className = "procStep";
    li.dataset.stage = stage.id;
    li.dataset.state = "pending";
    li.innerHTML = `<span>${stage.label}</span><span class="state">pending</span>`;
    container.appendChild(li);
  });
}

function setPanelMessage(canvasId, text, kind = "placeholder") {
  const canvas = $(canvasId);
  const ctx = createSizedCanvas(canvas, 320, 180);
  ctx.fillStyle = "#050608";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = kind === "error" ? "#ffb4b4" : "#94a3b8";
  ctx.font = "16px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

function setAllPanelPlaceholders(text) {
  setPanelMessage("inCanvas", text);
  setPanelMessage("outCanvas", text);
  setPanelMessage("symCanvas", text);
  for (const name of ORDER) {
    setPanelMessage(`rep-${name}`, text);
    setRepError(name, "");
  }
}

function setRepError(name, message) {
  const badge = $(`rep-status-${name}`);
  if (!badge) return;
  badge.textContent = message;
  badge.className = message ? "repStatus error" : "repStatus";
}

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
  return r.text();
}

async function resolveCoreBasePath() {
  const base = readBaseUrl().replace(/\/$/, "");
  const candidates = [
    `${base}/core`,
    `${window.location.origin}${window.location.pathname.replace(/\/[^/]*$/, "")}/../core`,
    "core",
    "../core",
    "/core",
  ].filter(Boolean);

  for (const candidate of candidates) {
    const normalized = candidate.replace(/\/$/, "");
    try {
      const response = await fetch(`${normalized}/__init__.py`, { cache: "no-cache" });
      if (response.ok) {
        return normalized;
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

function listMissingCoreFiles() {
  return REP_FILES.filter((rel) => typeof CORE_BUNDLE[rel] !== "string");
}

async function bootPyodide() {
  setStage("runtime", "active");
  status("Loading Pyodide runtime…");
  coreBasePath = await resolveCoreBasePath();
  coreLoadMode = coreBasePath ? "http" : "embedded";
  pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
  setStage("runtime", "done");
  setStage("packages", "active");
  status("Loading scientific packages (numpy, scipy, scikit-image, pywavelets, pillow, matplotlib)…");
  await pyodide.loadPackage([
    "numpy", "scipy", "scikit-image", "pywavelets", "pillow", "matplotlib",
  ]);
  setStage("packages", "done");
  setStage("core", "active");
  status("Mounting core/ Python package…");
  pyodide.FS.mkdirTree("/home/pyodide/lod/core/representations");
  pyodide.FS.mkdirTree("/home/pyodide/lod/core/reconstruction");
  if (coreLoadMode === "embedded") {
    const missing = listMissingCoreFiles();
    if (missing.length) {
      throw new Error(`Embedded core bundle incomplete. Missing: ${missing.join(", ")}`);
    }
  }
  for (const rel of REP_FILES) {
    const src = coreLoadMode === "http"
      ? await fetchText(`${coreBasePath}/${rel}`)
      : CORE_BUNDLE[rel];
    pyodide.FS.writeFile(`/home/pyodide/lod/core/${rel}`, src);
  }
  await pyodide.runPythonAsync(`
import json
import sys
import traceback
if "/home/pyodide/lod" not in sys.path:
    sys.path.insert(0, "/home/pyodide/lod")
import core
import core.representations as R
from core.io_utils import load_image, reattach_chroma
from core.reconstruction import COMBINERS
from core.symbolic import fit_symbolic, expression_string
from core.demo_image import generate_demo
import numpy as np
ORDER = R.ORDER
REGISTRY = R.REGISTRY
print("[pyodide] core ready, ORDER=", ORDER)
`);
  setStage("core", "done");
  status(coreLoadMode === "embedded"
    ? "Ready (using embedded core fallback). Pick an image or use the synthetic demo."
    : "Ready. Pick an image or use the synthetic demo.", "ok");
  updateDebugPanel();
}

function to1d(data2d, width, height) {
  const out = new Float32Array(width * height);
  let k = 0;
  for (let y = 0; y < height; y += 1) {
    const row = data2d[y] || [];
    for (let x = 0; x < width; x += 1) {
      out[k] = Number(row[x] ?? 0);
      k += 1;
    }
  }
  return out;
}

function getPyJson(name) {
  const value = pyodide.globals.get(name);
  const text = value.toString();
  value.destroy?.();
  return JSON.parse(text);
}

async function drawUploadedPreview(file) {
  const objectUrl = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = objectUrl;
    await img.decode();
    appState.imageLoaded = true;
    appState.imageWidth = img.naturalWidth;
    appState.imageHeight = img.naturalHeight;
    const fitted = fitImageInsideBox(img.naturalWidth, img.naturalHeight, MAX_PREVIEW_BOX, MAX_PREVIEW_BOX);
    drawImageToCanvas($("inCanvas"), img, fitted.width, fitted.height);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function imageBytesFromFile(file) {
  const buf = await file.arrayBuffer();
  return new Uint8Array(buf);
}

function renderRepresentation(name, field2d, width, height) {
  const flattened = sanitizeNumericArray(to1d(field2d, width, height));
  const nanCount = flattened.reduce((acc, v) => acc + (Number.isFinite(v) ? 0 : 1), 0);
  const { imageData, min, max } = scalarFieldToImageData(flattened, width, height);
  drawImageDataToCanvas($(`rep-${name}`), imageData);
  setRepError(name, "");
  appState.repDiagnostics[name] = { ok: true, min, max, nanCount };
}

function renderRepresentationRgb(name, rgb3d, width, height, diag) {
  const flat = new Float32Array(width * height * 3);
  let k = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = rgb3d[y][x] || [0, 0, 0];
      flat[k++] = p[0];
      flat[k++] = p[1];
      flat[k++] = p[2];
    }
  }
  const imageData = rgbFieldToImageData(flat, width, height);
  drawImageDataToCanvas($(`rep-${name}`), imageData);
  setRepError(name, "");
  appState.repDiagnostics[name] = { ...diag };
}

async function loadImageBytes(bytes) {
  appState.processing = true;
  appState.fatalError = null;
  appState.reconstructionStatus = "processing";
  updateDebugPanel();
  status("Computing mathematical representations…", "info");
  for (const name of ORDER) {
    setPanelMessage(`rep-${name}`, "Processing...");
  }
  setPanelMessage("outCanvas", "Processing...");
  try {
    setStage("image", "active");
    pyodide.FS.writeFile("/tmp/in.bin", bytes);
    await pyodide.runPythonAsync(`
import json, traceback
with open("/tmp/in.bin", "rb") as f:
    _b = f.read()
Y, CbCr = load_image(_b)
_Y = Y.astype(np.float32)
_CBCR = CbCr.astype(np.float32)
_repr = {}
_repr_rgb = {}
_diag = {}
for name in ORDER:
    try:
        rep = REGISTRY[name]
        raw = rep.compute(_Y)
        field = np.nan_to_num(rep.to_field(raw).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        color = np.nan_to_num(rep.visualize(raw).astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        _repr[name] = field
        _repr_rgb[name] = color
        _diag[name] = {
            "ok": True,
            "min": float(np.min(field)),
            "max": float(np.max(field)),
            "nanCount": int(np.isnan(field).sum()),
        }
    except Exception as e:
        _diag[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

if _repr:
    _FIELDS = np.stack([_repr.get(name, np.zeros_like(_Y, dtype=np.float32)) for name in ORDER], axis=0)
else:
    _FIELDS = np.zeros((len(ORDER),) + _Y.shape, dtype=np.float32)

_diag_json = json.dumps(_diag)
`);
    setStage("image", "done");
    setStage("representations", "active");

    const diag = getPyJson("_diag_json");
    appState.repDiagnostics = diag;

    const yProxy = pyodide.globals.get("_Y");
    const y2d = yProxy.toJs({ create_proxies: false });
    yProxy.destroy?.();
    const height = y2d.length;
    const width = y2d[0]?.length || 0;

    ORDER.forEach((name) => {
      const d = diag[name];
      if (!d?.ok) {
        setRepError(name, d?.error || "Representation failed");
        setPanelMessage(`rep-${name}`, "Error", "error");
        return;
      }
      const repProxy = pyodide.globals.get("_repr_rgb").get(name);
      const rep2d = repProxy.toJs({ create_proxies: false });
      repProxy.destroy?.();
      renderRepresentationRgb(name, rep2d, width, height, d);
    });
    setStage("representations", "done");

    setStage("reconstruction", "active");
    await runReconstruction();
    setStage("reconstruction", "done");
    status("Done.", "ok");
  } catch (e) {
    console.error(e);
    appState.fatalError = e.message;
    status(`Image processing failed: ${e.message}`, "error");
    setPanelMessage("outCanvas", "Error", "error");
    if (appState.stages.image === "active") setStage("image", "error");
    if (appState.stages.representations === "active") setStage("representations", "error");
    if (appState.stages.reconstruction === "active") setStage("reconstruction", "error");
  } finally {
    appState.processing = false;
    updateDebugPanel();
  }
}

async function runReconstruction() {
  if (!fieldsReady()) {
    setPanelMessage("outCanvas", "No image loaded");
    return;
  }
  const strategy = $("strategy").value;
  const beta = parseFloat($("beta").value);
  pyodide.globals.set("_w", weights);
  pyodide.globals.set("_t", toggles);
  pyodide.globals.set("_strategy", strategy);
  pyodide.globals.set("_beta", beta);
  try {
    await pyodide.runPythonAsync(`
import numpy as np
w = np.asarray(list(_w), dtype=np.float32)
t = np.asarray(list(_t), dtype=np.float32)
try:
    Y_hat = COMBINERS[_strategy](_FIELDS, weights=w, toggles=t, beta=float(_beta))
except Exception:
    Y_hat = _Y
_Y_HAT = np.nan_to_num(Y_hat.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
_rgb = reattach_chroma(_Y_HAT, _CBCR)
`);

    const rgbProxy = pyodide.globals.get("_rgb");
    const rgb = rgbProxy.toJs({ create_proxies: false });
    rgbProxy.destroy?.();
    const height = rgb.length;
    const width = rgb[0]?.length || 0;
    const flat = new Float32Array(width * height * 3);
    let k = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const p = rgb[y][x] || [0, 0, 0];
        flat[k++] = p[0];
        flat[k++] = p[1];
        flat[k++] = p[2];
      }
    }
    const imageData = rgbFieldToImageData(flat, width, height);
    drawImageDataToCanvas($("outCanvas"), imageData);
    appState.reconstructionStatus = "ok";
  } catch (e) {
    console.error(e);
    appState.reconstructionStatus = `error: ${e.message}`;
    status(`Reconstruction failed, fallback shown: ${e.message}`, "error");
    setPanelMessage("outCanvas", "Reconstruction error", "error");
  }
  updateDebugPanel();
}

async function runSymbolic() {
  if (!fieldsReady()) {
    setPanelMessage("symCanvas", "No image loaded");
    return;
  }
  const basis = $("basis").value;
  const degree = parseInt($("degree").value, 10);
  pyodide.globals.set("_basis", basis);
  pyodide.globals.set("_degree", degree);
  status("Fitting symbolic basis expansion…", "info");
  setStage("symbolic", "active");
  try {
    await pyodide.runPythonAsync(`
fit = fit_symbolic(_Y_HAT, basis=_basis, degree=int(_degree), max_k=int(_degree))
_expr = expression_string(fit, top_n=24)
_sym = np.nan_to_num(fit["approx"].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
`);
    const exprProxy = pyodide.globals.get("_expr");
    $("exprOut").textContent = exprProxy.toString();
    exprProxy.destroy?.();

    const symProxy = pyodide.globals.get("_sym");
    const sym2d = symProxy.toJs({ create_proxies: false });
    symProxy.destroy?.();
    const height = sym2d.length;
    const width = sym2d[0]?.length || 0;
    const field1d = to1d(sym2d, width, height);
    const { imageData } = scalarFieldToImageData(field1d, width, height);
    drawImageDataToCanvas($("symCanvas"), imageData);
    setStage("symbolic", "done");
    status("Symbolic expression updated.", "ok");
  } catch (e) {
    console.error(e);
    status(`Symbolic export failed: ${e.message}`, "error");
    setPanelMessage("symCanvas", "Symbolic error", "error");
    setStage("symbolic", "error");
  }
}

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
      <div id="rep-status-${name}" class="repStatus"></div>
      <canvas id="rep-${name}" width="320" height="180"></canvas>
      <div class="w">w<sub>${i + 1}</sub>
        <input type="range" data-i="${i}" class="ws" min="0" max="2" step="0.01" value="${(1 / ORDER.length).toFixed(3)}" />
        <span class="wv" data-i="${i}">${(1 / ORDER.length).toFixed(2)}</span>
      </div>
    `;
    container.appendChild(card);
  });
  setAllPanelPlaceholders("No image loaded");

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
  try {
    return Boolean(pyodide && pyodide.globals.get("_FIELDS"));
  } catch {
    return false;
  }
}

function debounce(fn, ms) {
  let h = null;
  return (...args) => {
    clearTimeout(h);
    h = setTimeout(() => fn(...args), ms);
  };
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

async function handleFileSelection(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    const message = `Unsupported file type: ${file.type || "unknown"}`;
    status(message, "error");
    appState.fatalError = message;
    setAllPanelPlaceholders("Upload an image file");
    updateDebugPanel();
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    const message = `File is too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Max ${MAX_UPLOAD_MB}MB.`;
    status(message, "error");
    appState.fatalError = message;
    setAllPanelPlaceholders("File too large");
    updateDebugPanel();
    return;
  }

  try {
    await drawUploadedPreview(file);
    const bytes = await imageBytesFromFile(file);
    await loadImageBytes(bytes);
  } catch (e) {
    console.error(e);
    appState.fatalError = e.message;
    status(`Failed to load image: ${e.message}`, "error");
    setAllPanelPlaceholders("Image load failed");
    updateDebugPanel();
  }
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
    await handleFileSelection(e.target.files?.[0]);
  });
  $("useDemo").addEventListener("click", async () => {
    status("Generating synthetic demo image…", "info");
    try {
      await pyodide.runPythonAsync(`
generate_demo("/tmp/demo.png", size=192)
with open("/tmp/demo.png", "rb") as f:
    _bb = f.read()
`);
      const bytesProxy = pyodide.globals.get("_bb");
      const bytes = bytesProxy.toJs({ create_proxies: false });
      bytesProxy.destroy?.();
      const file = new File([bytes], "demo.png", { type: "image/png" });
      await handleFileSelection(file);
    } catch (e) {
      console.error(e);
      status(`Demo generation failed: ${e.message}`, "error");
      updateDebugPanel();
    }
  });
  $("symbolic").addEventListener("click", runSymbolic);
}

(async function main() {
  buildProcessChecklist();
  buildRepControls();
  bindGlobalControls();
  updateDebugPanel();
  try {
    await bootPyodide();
  } catch (e) {
    if (appState.stages.runtime === "active") setStage("runtime", "error");
    else if (appState.stages.packages === "active") setStage("packages", "error");
    else if (appState.stages.core === "active") setStage("core", "error");
    appState.fatalError = e.message;
    status(`Pyodide init failed: ${e.message}`, "error");
    console.error(e);
    updateDebugPanel();
  }
})();
