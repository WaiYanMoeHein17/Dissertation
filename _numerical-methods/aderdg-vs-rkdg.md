---
layout: page
title: ADER-DG vs RK-DG for the Shallow Water Equations
lesson_number: 11
extract: High-order DG time-stepping benchmarked in the browser
categories: [hyperbolic, numerical-methods]
---

This page implements and benchmarks two high-order time-stepping schemes for the
[Shallow Water Equations](/fluids/shallow-water) on a periodic Cartesian mesh:

- **RK-DG** — Runge-Kutta Discontinuous Galerkin with [SSP](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method)
  schemes (SSPRK2 for $p=1$, SSPRK3 for $p \ge 2$).
- **ADER-DG** — Arbitrary high-order DERivatives DG, implemented via a
  *local space-time DG predictor* (Picard iteration over $p+1$ Legendre-Gauss-Lobatto
  collocation nodes in time) followed by a single explicit corrector that uses
  time-averaged volume fluxes and time-averaged Riemann fluxes at element faces.

Both schemes share the same spatial DG operator (nodal LGL basis, Rusanov flux),
so the comparison isolates the time integrator. Convergence is verified by the
Method of Manufactured Solutions: for the smooth ansatz
$h(x,t) = H_0 + a\sin(kx-\omega t)$, $u(x,t) = U_0 + b\cos(kx-\omega t)$
(2D analogue separates the $y$ dependence), the analytic source term that makes
this an exact solution of the SWE is added to the right-hand side. The reported
error is the discrete $L^2$ norm against the exact field.

A polynomial degree of $p$ is expected to give convergence rate $p+1$. ADER-DG
also achieves order $p+1$ in time at standard CFL, whereas RK-DG with SSPRK3
caps at order 3 in time and so requires an artificially shrunk time step
($\Delta t \sim \Delta x^{(p+1)/3}$) to expose the spatial order at $p \ge 3$.

<div id="dg-controls" class="dg-panel">
  <h3>Convergence study</h3>
  <div class="dg-row">
    <label>Dimension:
      <select id="dg-dim"><option value="1" selected>1D</option><option value="2">2D</option></select>
    </label>
    <label>Polynomial degree p:
      <select id="dg-p"><option>1</option><option selected>2</option><option>3</option></select>
    </label>
    <label>Mesh sizes (N):
      <input id="dg-N" type="text" value="16, 32, 64, 128" size="22">
    </label>
    <label>Final time T:
      <input id="dg-T" type="number" step="0.01" value="0.1" min="0.001" max="2.0" style="width:5em">
    </label>
    <label>CFL:
      <input id="dg-cfl" type="number" step="0.05" value="0.25" min="0.01" max="1.0" style="width:4em">
    </label>
    <button id="dg-run" type="button">Run convergence</button>
  </div>
  <div id="dg-status" class="dg-status">Idle.</div>
  <div id="dg-tables"></div>
  <div class="dg-chart-wrap"><canvas id="dg-conv-chart"></canvas></div>
</div>

<div id="live-controls" class="dg-panel">
  <h3>Live demo</h3>
  <p>A smooth Gaussian bump in $h$ on flat water, evolved in real time by the
  chosen scheme. Use this to sanity-check stability and shape preservation.</p>
  <div class="dg-row">
    <label>Dimension:
      <select id="live-dim"><option value="1" selected>1D</option><option value="2">2D</option></select>
    </label>
    <label>p: <select id="live-p"><option>1</option><option selected>2</option><option>3</option></select></label>
    <label>N: <input id="live-N" type="number" min="4" max="256" value="64" style="width:5em"></label>
    <label>Scheme:
      <select id="live-scheme">
        <option value="rk">RK-DG</option>
        <option value="ader" selected>ADER-DG</option>
      </select>
    </label>
    <button id="live-start" type="button">Start</button>
    <button id="live-stop" type="button">Stop</button>
  </div>
  <div id="live-status" class="dg-status">Idle.</div>
  <canvas id="live-canvas" width="640" height="280" style="background:#111;border-radius:6px;display:block;margin:0.6em auto;max-width:100%"></canvas>
</div>

<style>
.dg-panel { border: 1px solid var(--tile-border-color, #ccc); border-radius: 8px; padding: 1em; margin: 1em 0; }
.dg-panel h3 { margin-top: 0; }
.dg-row { display: flex; flex-wrap: wrap; gap: 0.8em; align-items: center; margin-bottom: 0.6em; }
.dg-row label { font-size: 0.9em; }
.dg-row button { padding: 0.4em 1em; font-size: 0.95em; cursor: pointer; }
.dg-status { font-family: monospace; font-size: 0.9em; padding: 0.4em 0; }
.dg-table-wrap { display: inline-block; vertical-align: top; margin-right: 1.5em; margin-bottom: 0.5em; }
.dg-table { border-collapse: collapse; font-family: monospace; font-size: 0.85em; }
.dg-table th, .dg-table td { border: 1px solid var(--tile-border-color, #ccc); padding: 0.2em 0.6em; text-align: right; }
.dg-table th { background: rgba(0,0,0,0.04); }
.dg-chart-wrap { position: relative; height: 320px; max-width: 720px; margin-top: 0.8em; }
</style>

<script src="/sim/scripts/charts.umd.min.js"></script>
{% raw %}
<script type="module">
  import { initApp } from "/sim/scripts/dg/app.js";
  initApp();
</script>
{% endraw %}

### Implementation notes

The Picard iteration in the predictor solves, on each cell independently, the
local space-time problem
$$ \mathbf{D}_\tau\, \mathbf{Q} = \Delta t\, \big( -\tfrac{2}{\Delta x}\partial_\xi F(\mathbf{Q}) + S \big), \qquad \mathbf{Q}(\tau{=}0) = \mathbf{q}^n , $$
where $\mathbf{D}_\tau$ is the time differentiation matrix on $p+1$ LGL nodes
on $[0,1]$. The initial condition is enforced strongly, leaving a $p \times p$
reduced system per spatial node which is solved by Picard sweeps starting from
the constant-in-time initial guess $\mathbf{Q}^{(0)} = \mathbf{q}^n$. Sufficient
sweeps for order $p+1$ accuracy are obtained with $p+1$ iterations.

The corrector is a single explicit DG step with time-averaged volume flux
$\bar F$ and time-averaged numerical flux $\bar F^*$ at every face, both formed
as $\sum_l w^\tau_l \cdot (\cdot)$ over the converged predictor. The corrector
shares its spatial operator with RK-DG, so both schemes use exactly the same
mass and stiffness matrices, the same Rusanov interface flux, and the same
manufactured-solution source.
