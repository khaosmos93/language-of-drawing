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

## Symbolic export

Approximates `Î(x,y) ≈ Σₖ aₖ ψₖ(x,y)` via least-squares regression onto a
polynomial or Fourier basis. Writes `out/symbolic.txt` with the top-N terms as a
human-readable algebraic expression (also exposed in the UI).

## Project layout

```
core/
  representations/    # the 9 reps (one file each)
  reconstruction/     # linear / nonlinear / pde_fusion combiners
  pipeline.py         # orchestrator
  symbolic.py         # basis-expansion regression
  normalize.py        # percentile [0,1]
  io_utils.py         # Pillow-based load / save / chroma reattach
  demo_image.py       # synthetic demo generator

run.py                # CLI entry point
web/                  # static site, deployable to GitHub Pages
  index.html
  static/{app.js, style.css}
.github/workflows/pages.yml   # Pages CI (copies core/ next to web/)
tests/                # pytest sanity + smoke tests
```

## Running locally (Codespaces)

```bash
pip install -r requirements.txt
python run.py --demo                       # generates data/demo.png and runs
python run.py --input data/demo.png --out out/
pytest -q
```

Artifacts written under `out/`:

- `rep_<name>.png`  — colorized visualization for each of the 9 representations
- `field_<name>.png` — raw normalized scalar field
- `grid.png` — 3×3 grid of all representations
- `recon_linear.png`, `recon_nonlinear.png`, `recon_pde.png`
- `recon_*_luma.png` — luminance-only versions
- `fields.npz` — all 9 normalized fields stacked
- `symbolic.txt`, `symbolic.npz`, `symbolic_approx.png`

## Web UI

Open `web/index.html` directly (or via any static server). On load, Pyodide
fetches `core/*.py` and runs the same Python pipeline in your browser. You get:

- file picker + synthetic-demo button
- live preview of each Rᵢ, with on/off toggles and weight sliders (0–2)
- strategy selector (linear / nonlinear / PDE), `β` slider for the nonlinear gain
- random perturb button (`wᵢ += 𝒩(0, 0.2)`)
- symbolic export panel — choose basis (polynomial / Fourier) and degree

### Deploying to GitHub Pages

The included workflow `.github/workflows/pages.yml` copies `core/` next to
`web/` and uploads the result to Pages. Enable Pages in repo settings
(*Settings → Pages → Build from GitHub Actions*) and it deploys on every push.

## Hard constraints

- ❌ no pretrained generative models
- ❌ no diffusion, no GANs
- ✅ every transformation is mathematically explicit and lives in a single, readable file

## License

MIT (add `LICENSE` if redistributing).
