// bench.js — Benchmark harness for ADER-DG vs RK-DG.
//
// Two studies:
//   1) runConvergence1D({ p, NList, scheme, T, CFL, mms })  -> { Ns, dxs, errs, rates, runtimes }
//   2) runConvergence2D({ p, NList, scheme, T, CFL, mms })  -> ditto
//   3) runRuntime1D / runRuntime2D — run a single (N, p) pair, report wall-time.
//
// CFL choice: temporal order matches spatial order for ADER-DG (p+1) but is
// capped at 3 for RK-DG via SSPRK3. To make convergence rates reflect the
// spatial scheme order rather than time-stepping limits, dt is chosen as
//
//     dt = CFL / ((2 p + 1) * a_max)           [standard 1D DG CFL]
//          * (h / h_ref) ** ((p+1)/3)          [for RK-DG when p > 2]
//
// so that temporal error stays subdominant. ADER-DG keeps dt = CFL / ((2p+1) a_max).

import { makeBasis } from "./basis.js";
import { Mesh1D, projectField1D, l2Error1D } from "./dg1d.js";
import { Mesh2D, projectField2D, l2Error2D } from "./dg2d.js";
import { makeRKDG1D, makeRKDG2D } from "./rkdg.js";
import { makeADERDG1D, makeADERDG2D } from "./aderdg.js";
import { makeMMS1D, makeMMS2D } from "./mms.js";
import { maxWavespeed1Dflat, maxWavespeed2D } from "./swe.js";

function dtFor(p, dx, amax, CFL, scheme, schemeOrder) {
  // Standard DG CFL constraint: dt <= CFL_safe * dx / ((2p+1) a_max).
  let dt = CFL * dx / ((2 * p + 1) * amax);
  // For RK-DG with p > 2 the temporal scheme is order 3; shrink dt to keep
  // temporal error subdominant: dt ~ dx^{(p+1)/3}. We do this only when
  // schemeOrder < p+1 (i.e., RK-DG).
  if (scheme === "rk" && p + 1 > schemeOrder) {
    const exponent = (p + 1) / schemeOrder;
    dt = dt * Math.pow(dx, exponent - 1);
  }
  return dt;
}

function rkOrder(p) { return p === 1 ? 2 : 3; }

// =====================================================================
// 1D
// =====================================================================
export function runConvergence1D({ p, NList, scheme, T = 0.1, CFL = 0.25,
                                   mms = makeMMS1D(), L = 1.0, g = 1.0,
                                   picardIters = -1 } = {}) {
  const basis = makeBasis(p);
  const dxs = [], errs = [], runtimes = [], steps = [];
  for (const Nx of NList) {
    const mesh = new Mesh1D(L, Nx, basis);
    const q = projectField1D(mesh, mms.exact, 0.0);
    const stepper = scheme === "rk"
      ? makeRKDG1D(mesh, g, mms.source)
      : makeADERDG1D(mesh, g, mms.source, picardIters);
    const order = scheme === "rk" ? rkOrder(p) : (p + 1);
    const t0 = performance.now();
    let t = 0;
    let nSteps = 0;
    while (t < T) {
      const amax = Math.max(1e-10, maxWavespeed1Dflat(q, 2, g));
      let dt = dtFor(p, mesh.dx, amax, CFL, scheme, order);
      if (t + dt > T) dt = T - t;
      t = stepper.step(q, dt, t);
      nSteps++;
    }
    const wall = performance.now() - t0;
    const e = l2Error1D(mesh, q, mms.exact, T);
    dxs.push(mesh.dx);
    errs.push(e);
    runtimes.push(wall);
    steps.push(nSteps);
  }
  const rates = [];
  for (let i = 1; i < dxs.length; i++) {
    rates.push(Math.log(errs[i - 1] / errs[i]) / Math.log(dxs[i - 1] / dxs[i]));
  }
  return { Ns: NList.slice(), dxs, errs, rates, runtimes, steps };
}

// =====================================================================
// 2D
// =====================================================================
export function runConvergence2D({ p, NList, scheme, T = 0.05, CFL = 0.2,
                                   mms = makeMMS2D(), L = 1.0, g = 1.0,
                                   picardIters = -1 } = {}) {
  const basis = makeBasis(p);
  const dxs = [], errs = [], runtimes = [], steps = [];
  for (const N of NList) {
    const mesh = new Mesh2D(L, L, N, N, basis);
    const q = projectField2D(mesh, mms.exact, 0.0);
    const stepper = scheme === "rk"
      ? makeRKDG2D(mesh, g, mms.source)
      : makeADERDG2D(mesh, g, mms.source, picardIters);
    const order = scheme === "rk" ? rkOrder(p) : (p + 1);
    const t0 = performance.now();
    let t = 0;
    let nSteps = 0;
    while (t < T) {
      const amax = Math.max(1e-10, maxWavespeed2D(q, 3, g));
      let dt = dtFor(p, mesh.dx, amax, CFL, scheme, order);
      if (t + dt > T) dt = T - t;
      t = stepper.step(q, dt, t);
      nSteps++;
    }
    const wall = performance.now() - t0;
    const e = l2Error2D(mesh, q, mms.exact, T);
    dxs.push(mesh.dx);
    errs.push(e);
    runtimes.push(wall);
    steps.push(nSteps);
  }
  const rates = [];
  for (let i = 1; i < dxs.length; i++) {
    rates.push(Math.log(errs[i - 1] / errs[i]) / Math.log(dxs[i - 1] / dxs[i]));
  }
  return { Ns: NList.slice(), dxs, errs, rates, runtimes, steps };
}
