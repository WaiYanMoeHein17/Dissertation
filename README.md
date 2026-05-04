# VisualPDE — Dissertation fork

This is a fork of [VisualPDE](https://visualpde.com) extended with an ADER-DG
timestepping pipeline and a CPU/Float64 reference benchmark for the
dissertation. The upstream README follows below; this section documents the
dissertation-specific additions.

## What's been added

**ADER-DG pipeline (in `sim/scripts/RD/`):**

- `simulation_shaders.js` — predictor + Picard + corrector shader generators
  parameterised on polynomial degree `p`. Includes the LGL D and D² matrices
  baked in as compile-time GLSL constants, an MRT-fused variant of each
  pass, and a Local Lax-Friedrichs (Rusanov) flux at element interfaces
  with auto-detection of 2D Shallow-Water flux when the user's preset
  declares `g` and `H_e` in `kineticParams`.
- `main.js` — wires the ADER3 and ADER4 schemes into VisualPDE's timestepping
  dropdown, with the hybrid 1D-DG-x + FD-y layout described in the report.
  The element-local RHS is built from the user's `reactionStr_*` expressions
  via the parser adapter `buildDGComputeRHSBody`.
- `clear_shader.js` — projects user initial conditions onto the LGL nodes
  in DG mode.

**Benchmark + CPU reference (in `sim/scripts/dg/`):**

- `bench_linswe.mjs` — the canonical convergence benchmark used to produce
  the dissertation's tables. Runs ADER-DG and RK-DG on linearised SWE in
  both 1D and 2D for orders 2–5, computes L² errors at refined meshes, fits
  rates, and emits LaTeX tables.
- `aderdg.js`, `rkdg.js`, `dg1d.js`, `dg2d.js`, `basis.js`, `swe_lin.js`,
  `swe.js`, `mms.js` — supporting CPU/Float64 implementations.
- `sandbox/` — GPU/CPU equivalence tests (`sandbox.mjs`,
  `_preamble_check.mjs`, `_cpu_check.mjs`, `_dump_shaders.mjs`).

## Running the benchmark

The convergence benchmark is a pure Node script:

```
cd sim/scripts/dg
node bench_linswe.mjs
```

This prints text tables for both schemes in 1D and 2D, plus the LaTeX
fragments used in the report. A captured run lives at
`sim/scripts/dg/linswe_output.txt` for reference.

## Verifying the GPU pipeline

The GLSL preamble's differentiation matrices are verified against the
CPU `basis.js` reference to machine epsilon at both polynomial degrees:

```
node sim/scripts/dg/sandbox/_preamble_check.mjs
```

A live GPU/CPU equivalence test for the predictor runs in the browser
via `sim/scripts/dg/sandbox/index.html` (open it through the local Jekyll
server described below).

## Trying ADER-DG in the browser

The browser path defaults to a **visual mode** that silently runs the
midpoint method when the user picks ADER3 / ADER4 in the timestepping
dropdown. This keeps the demo visually clean (no LGL element artefacts on
screen) without affecting any of the dissertation's CPU/Float64 numbers.

To run the actual ADER-DG pipeline (for examiner verification or
benchmarking on the GPU), append `?aderVisualMode=false` to the URL:

```
http://localhost:4000/sim/?preset=ShallowWaterEqns&timesteppingScheme=ADER4&dt=0.003&aderVisualMode=false
```

The `dt` value above is conservative — feel free to lower it if the
problem destabilises at your chosen brush amplitude. Visual-mode demos
can use the preset's default `dt = 0.005`.

## Convention

The report and the code both use the convention `method order = p + 1`
where `p` is the polynomial degree of the LGL basis. So:

| Scheme label | Polynomial degree `p` | LGL nodes / element | Method order |
|---|---|---|---|
| `ADER3` | 2 | 3 | 3 |
| `ADER4` | 3 | 4 | 4 |

The CPU benchmark also tests `p ∈ {1, 4}` for orders 2 and 5; the GPU
pipeline only ships `p ∈ {2, 3}` because those are the orders the
dissertation reports for browser-side performance.

---

## Interactive solutions of partial differential equations, live on your device.

VisualPDE is a browser-based simulator of a broad range of [partial differential equations](https://en.wikipedia.org/wiki/Partial_differential_equation), with solve-as-you-type speed and no knowledge of numerical methods required.

The site, hosted at [VisualPDE.com](https://visualpde.com), contains a range of educational and scientific material, including a collection of Visual Stories written with the layperson in mind.

For more information on the technology and philosophy behind VisualPDE, check out our [open-access publication](https://doi.org/10.1007/s11538-023-01218-4).

## Hosting a local copy

VisualPDE is updated regularly, so using [VisualPDE.com](https://visualpde.com) is recommended for most users. However, you may wish to host your own local copy of the VisualPDE site to guarantee stability or privacy. Local versions of VisualPDE automatically have analytics disabled and are entirely self-contained.

### Best hosting

The best, most customisable way to do this is via [Jekyll](https://jekyllrb.com), which requires a [Ruby](https://www.ruby-lang.org/) installation (version 3.1.x is required; newer versions are not compatible with Jekyll).

For instance, on macOS with Homebrew installed, the following will install and configure Ruby and Jekyll:

```
brew install ruby@3.1
gem install bundler jekyll
bundle install
```

To build and serve the site locally, download the entire VisualPDE source from this repo and navigate to it in your terminal. Hosting the site at `http://localhost:4000` is then as simple as running

```
bundle exec jekyll serve
```

You can then customise your local version of VisualPDE by editing any of the various Markdown files used to build the site.

### Simple hosting

A simpler but less flexible method of hosting your own version of the site is to [download](https://benjaminwalker.info/visual-pde.zip) an archived version of the built site.

This can then be served with any local webserver. For instance, with Python3 installed you can simply run

```
cd path/to/visual-pde
python3 -m http.server
```

## Having trouble?

VisualPDE has extensive documentation, so we recommend trying out any suggestions found on the main site. For anything else, please get in touch with us on [hello@visualpde.com](mailto:hello@visualpde.com) or raise an issue on GitHub.
