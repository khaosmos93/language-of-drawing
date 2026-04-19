# Language of Drawing

Transform an image `I(x,y)` into **9 mathematically-explicit representations**, then
reconstruct a new image `Î(x,y)` from those representations using **3 different
combination strategies** — purely from math equations and their coloring. No GANs,
no diffusion, no pretrained black boxes.

```
input image  →  R₁,…,R₉ (9 representations)  →  Î = combine(R̃₁,…,R̃₉)
```

The same Python core runs:
- as a **CLI** in GitHub Codespaces (`python run.py`)
- as a **fully client-side web UI** on **GitHub Pages** via [Pyodide](https://pyodide.org)

## The 9 representations

| # | Name        | Equation / definition                                          |
|---|-------------|-----------------------------------------------------------------|
| 1 | Fourier     | `F(u,v) = ∫∫ I(x,y) e^{-2πi(ux+vy)} dx dy`                      |
| 2 | Wavelet     | `W(s,x,y) = ⟨I, ψ_{s,x,y}⟩` (multi-scale `db2`)                 |
| 3 | Gradient    | `∇I = (∂ₓI, ∂ᵧI)` (Sobel)                                       |
| 4 | Level set   | Signed distance `φ` to median-intensity contour, `tanh(φ/σ)`     |
| 5 | Graph       | SLIC superpixels → Laplacian `L = D − W` → Fiedler vector       |
| 6 | PDE source  | `f = ∇²I` (inverse-source view of the Poisson eqn)              |
| 7 | Probability | Histogram density `p(I(x,y))` ⊕ local entropy                   |
| 8 | Fractal     | Local box-counting dimension `D(x,y)`                           |
| 9 | Manifold    | Structure-tensor metric `g_ij`; field = coherence               |

Each `R_i` is normalized to `R̃_i ∈ [0,1]` via percentile clipping (`core/normalize.py`).

## The 3 reconstruction strategies

```
A. Linear         Î = robust01( Σ wᵢ · R̃ᵢ )
B. Nonlinear      Î = σ( β · ( Σ wᵢ φᵢ(R̃ᵢ) − μ ) )
C. PDE fusion     ∇² Î = Σ wᵢ Gᵢ(R̃ᵢ),    solved with Neumann BCs (DCT)
```

Color images: the pipeline runs on luminance (Y) only and re-attaches the
original chroma (CbCr) for output, so the math acts on intensity but the result
keeps the input's color identity.

## Running locally (Python CLI)

```bash
pip install -r requirements.txt
python run.py --demo
python run.py --input data/demo.png --out out/
pytest -q
```

## Running locally (Web app with Vite)

```bash
npm install
npm run dev
```

Open the printed local URL (usually `http://localhost:5173/`).

### Web debugging behavior

The UI now includes a Diagnostics panel with:
- upload dimensions and load status
- representation min/max and NaN counts
- per-representation failures (fail-soft)
- reconstruction state
- `baseURL` and `pathname` to debug GitHub Pages paths

## GitHub Pages deployment

Build once locally:

```bash
# For project pages: https://<user>.github.io/<repo>/
VITE_BASE_PATH="/REPO_NAME/" npm run build

# For custom domain/root hosting
VITE_BASE_PATH="/" npm run build
```

This creates `dist/` and copies `core/` into `dist/core/`, so Pyodide can fetch the
Python modules from static hosting.

For environments that serve only `web/` (for example
`python -m http.server --directory web 8000`), the app now includes an embedded
fallback copy of the Python `core` package and automatically uses it when
`/core/__init__.py` is not reachable over HTTP.

### Notes
- `vite.config.js` contains the `base` setting and comments for both deployment modes.
- The app also auto-detects fallback `core/` paths (`base/core`, `core`, `../core`, `/core`) for local/static hosting compatibility.

## Artifacts written by Python pipeline (`run.py`)

- `rep_<name>.png`  — colorized visualization for each of the 9 representations
- `field_<name>.png` — raw normalized scalar field
- `grid.png` — 3×3 grid of all representations
- `recon_linear.png`, `recon_nonlinear.png`, `recon_pde.png`
- `recon_*_luma.png` — luminance-only versions
- `fields.npz` — all 9 normalized fields stacked
- `symbolic.txt`, `symbolic.npz`, `symbolic_approx.png`

## Hard constraints

- ❌ no pretrained generative models
- ❌ no diffusion, no GANs
- ✅ every transformation is mathematically explicit and lives in a single, readable file

## License

MIT (add `LICENSE` if redistributing).
