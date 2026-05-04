// rkdg.js — Strong Stability Preserving Runge-Kutta DG.
//
// SSPRK2 (p=1):     u^{(1)} = u^n + dt L(u^n)
//                   u^{n+1} = 0.5 u^n + 0.5 (u^{(1)} + dt L(u^{(1)}))
// SSPRK3 (p>=2):    u^{(1)} = u^n + dt L(u^n)
//                   u^{(2)} = 0.75 u^n + 0.25 (u^{(1)} + dt L(u^{(1)}))
//                   u^{n+1} = (1/3) u^n + (2/3)(u^{(2)} + dt L(u^{(2)}))
// Both stages take the source at their own "stage time" so MMS forcing is
// integrated with matching order of accuracy.

import { allocWork1D, dgRhs1D } from "./dg1d.js";
import { allocWork2D, dgRhs2D } from "./dg2d.js";
import { buildSourceField1D, buildSourceField2D } from "./mms.js";

function pickStepper(p) {
  // p=1 -> SSPRK2 (order 2). p>=2 -> SSPRK3 (order 3 in time).
  // For p>=3 the spatial order >3, so SSPRK3 caps temporal accuracy at 3.
  // To keep the experiment "matched", many DG papers report results using
  // dt = C h^{(p+1)/3} so the SSPRK3 temporal error stays subdominant.
  // Our run scripts compute dt accordingly.
  return p === 1 ? "SSPRK2" : "SSPRK3";
}

// ============= 1D =============
export function makeRKDG1D(mesh, g, mmsSource = null) {
  const work = allocWork1D(mesh);
  const N = mesh.Nx * mesh.n * mesh.M;
  const rhs = new Float64Array(N);
  const u1 = new Float64Array(N);
  const u2 = new Float64Array(N);
  const stepper = pickStepper(mesh.basis.p);
  let lastDt = 0;

  function step(q, dt, t) {
    if (mmsSource) {
      const s0 = buildSourceField1D(mesh, mmsSource, t);
      dgRhs1D(rhs, q, mesh, work, g, null, null, s0);
    } else {
      dgRhs1D(rhs, q, mesh, work, g, null, null, null);
    }
    if (stepper === "SSPRK2") {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      const s1 = mmsSource ? buildSourceField1D(mesh, mmsSource, t + dt) : null;
      dgRhs1D(rhs, u1, mesh, work, g, null, null, s1);
      for (let i = 0; i < N; i++) q[i] = 0.5 * q[i] + 0.5 * (u1[i] + dt * rhs[i]);
    } else { // SSPRK3
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      const s1 = mmsSource ? buildSourceField1D(mesh, mmsSource, t + dt) : null;
      dgRhs1D(rhs, u1, mesh, work, g, null, null, s1);
      for (let i = 0; i < N; i++) u2[i] = 0.75 * q[i] + 0.25 * (u1[i] + dt * rhs[i]);
      const s2 = mmsSource ? buildSourceField1D(mesh, mmsSource, t + 0.5 * dt) : null;
      dgRhs1D(rhs, u2, mesh, work, g, null, null, s2);
      for (let i = 0; i < N; i++) q[i] = (q[i] + 2 * (u2[i] + dt * rhs[i])) / 3;
    }
    lastDt = dt;
    return t + dt;
  }
  return { step, name: `RK-DG (${stepper})`, get lastDt() { return lastDt; }, mesh };
}

// ============= 2D =============
export function makeRKDG2D(mesh, g, mmsSource = null) {
  const work = allocWork2D(mesh);
  const N = mesh.Nx * mesh.Ny * mesh.n * mesh.n * mesh.M;
  const rhs = new Float64Array(N);
  const u1 = new Float64Array(N);
  const u2 = new Float64Array(N);
  const stepper = pickStepper(mesh.basis.p);

  function step(q, dt, t) {
    const s0 = mmsSource ? buildSourceField2D(mesh, mmsSource, t) : null;
    dgRhs2D(rhs, q, mesh, work, g, null, null, null, null, s0);
    if (stepper === "SSPRK2") {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      const s1 = mmsSource ? buildSourceField2D(mesh, mmsSource, t + dt) : null;
      dgRhs2D(rhs, u1, mesh, work, g, null, null, null, null, s1);
      for (let i = 0; i < N; i++) q[i] = 0.5 * q[i] + 0.5 * (u1[i] + dt * rhs[i]);
    } else {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      const s1 = mmsSource ? buildSourceField2D(mesh, mmsSource, t + dt) : null;
      dgRhs2D(rhs, u1, mesh, work, g, null, null, null, null, s1);
      for (let i = 0; i < N; i++) u2[i] = 0.75 * q[i] + 0.25 * (u1[i] + dt * rhs[i]);
      const s2 = mmsSource ? buildSourceField2D(mesh, mmsSource, t + 0.5 * dt) : null;
      dgRhs2D(rhs, u2, mesh, work, g, null, null, null, null, s2);
      for (let i = 0; i < N; i++) q[i] = (q[i] + 2 * (u2[i] + dt * rhs[i])) / 3;
    }
    return t + dt;
  }
  return { step, name: `RK-DG (${stepper})`, mesh };
}
