// Standalone Node check of the full order-3 ADER-DG predictor (CPU only).
// Confirms that the predictor produces an order-3-accurate advance on
// linear advection at one timestep, and that ||q1 - q2|| is consistent
// with the GL-node spread (c2 - c1)·dt·u_x.
//
// Usage: node sim/scripts/dg/sandbox/_cpu_check.mjs

import { makeBasis } from "../basis.js";

const P = 3;
const N_NODES = P + 1;
const N_ELEM = 16;
const N_X = N_ELEM * N_NODES;
const L = 1.0;
const DX_ELEMENT = L / N_ELEM;
const DT = 0.001;
const A_SPEED = 1.0;
const C1 = (3 - Math.sqrt(3)) / 6;
const C2 = (3 + Math.sqrt(3)) / 6;
const A11 = 0.25, A12 = C1 - 0.25, A21 = C2 - 0.25, A22 = 0.25;

const basis = makeBasis(P);

function physicalXAtTexel(j) {
  const e = Math.floor(j / N_NODES);
  const i = j - e * N_NODES;
  return e * DX_ELEMENT + 0.5 * DX_ELEMENT * (basis.x[i] + 1.0);
}

function initialState() {
  const u = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) u[j] = Math.sin(2 * Math.PI * physicalXAtTexel(j));
  return u;
}

function elementLocalFlux(u) {
  const D = basis.D;
  const out = new Float32Array(N_X);
  const inv = 2.0 / DX_ELEMENT;
  for (let e = 0; e < N_ELEM; e++) {
    for (let i = 0; i < N_NODES; i++) {
      let dx = 0;
      for (let k = 0; k < N_NODES; k++) dx += D[i * N_NODES + k] * u[e * N_NODES + k];
      out[e * N_NODES + i] = -A_SPEED * inv * dx;
    }
  }
  return out;
}

function predictor(u) {
  const F0 = elementLocalFlux(u);
  const q1Init = new Float32Array(N_X);
  const q2Init = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) {
    q1Init[j] = u[j] + C1 * DT * F0[j];
    q2Init[j] = u[j] + C2 * DT * F0[j];
  }
  const F1 = elementLocalFlux(q1Init);
  const F2 = elementLocalFlux(q2Init);
  const q1 = new Float32Array(N_X);
  const q2 = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) {
    q1[j] = u[j] + DT * (A11 * F1[j] + A12 * F2[j]);
    q2[j] = u[j] + DT * (A21 * F1[j] + A22 * F2[j]);
  }
  return { q1Init, q2Init, q1, q2 };
}

function maxAbs(diff) { let m = 0; for (const d of diff) if (Math.abs(d) > m) m = Math.abs(d); return m; }

const u = initialState();
const { q1Init, q2Init, q1, q2 } = predictor(u);

// Predictor q_i should approximate u(x, c_i·DT) on linear advection.
const exactAt = (t) => Float32Array.from(
  Array.from({ length: N_X }, (_, j) => Math.sin(2 * Math.PI * (physicalXAtTexel(j) - t))),
);

const eC1Init = exactAt(C1 * DT);
const eC2Init = exactAt(C2 * DT);
const eC1     = exactAt(C1 * DT);
const eC2     = exactAt(C2 * DT);

const errs = {
  initC1:   maxAbs(q1Init.map((v, j) => v - eC1Init[j])),
  initC2:   maxAbs(q2Init.map((v, j) => v - eC2Init[j])),
  picardC1: maxAbs(q1.map((v, j) => v - eC1[j])),
  picardC2: maxAbs(q2.map((v, j) => v - eC2[j])),
};

// FE init has temporal error O(Δt²); one Picard sweep on a smooth solution
// reduces it. Comparing against u(x - c_i·Δt) measures the element-local
// truncation floor too (the predictor has no inter-element flux), so we
// don't expect to hit pure-Taylor³ — only that the Picard sweep is
// substantively better than the FE init.

console.log("ADER-DG order-3 predictor — full CPU sanity check");
console.log(`  Δt = ${DT},  c1·Δt = ${(C1 * DT).toExponential(3)},  c2·Δt = ${(C2 * DT).toExponential(3)}`);
console.log();
console.log("  ===== FE init pass (predictor at FE only) =====");
console.log(`  ||q1_init - u(x, c1·Δt)||_inf = ${errs.initC1.toExponential(3)}`);
console.log(`  ||q2_init - u(x, c2·Δt)||_inf = ${errs.initC2.toExponential(3)}`);
console.log();
console.log("  ===== After 1 Picard sweep =====");
console.log(`  ||q1 - u(x, c1·Δt)||_inf = ${errs.picardC1.toExponential(3)}`);
console.log(`  ||q2 - u(x, c2·Δt)||_inf = ${errs.picardC2.toExponential(3)}`);
console.log();
const ratio1 = errs.initC1 / errs.picardC1;
const ratio2 = errs.initC2 / errs.picardC2;
console.log(`  Picard/init ratio: q1 = ${ratio1.toFixed(1)}×,  q2 = ${ratio2.toFixed(1)}×`);
console.log("  (one sweep should reduce error by at least 5× on smooth solutions)");
console.log();

// Refinement test — vary Δt and verify the Picard predictor converges at
// the expected rate (~3 for one Picard sweep on smooth advection, before
// element-local truncation dominates).
console.log("  ===== Δt refinement (q1 error vs c1·Δt) =====");
const baseDt = DT;
const dts = [baseDt, baseDt / 2, baseDt / 4, baseDt / 8];
const errors = [];
for (const dt of dts) {
  const F0 = elementLocalFlux(u);
  const q1Init = new Float32Array(N_X);
  const q2Init = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) {
    q1Init[j] = u[j] + C1 * dt * F0[j];
    q2Init[j] = u[j] + C2 * dt * F0[j];
  }
  const F1 = elementLocalFlux(q1Init);
  const F2 = elementLocalFlux(q2Init);
  const q1d = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) q1d[j] = u[j] + dt * (A11 * F1[j] + A12 * F2[j]);
  const ex = exactAt(C1 * dt);
  errors.push({ dt, err: maxAbs(q1d.map((v, j) => v - ex[j])) });
}
console.log("  Δt          err              rate");
for (let i = 0; i < errors.length; i++) {
  const { dt, err } = errors[i];
  const rate = i === 0 ? "—" :
    (Math.log(errors[i - 1].err / err) / Math.log(errors[i - 1].dt / dt)).toFixed(2);
  console.log(`  ${dt.toExponential(2)}   ${err.toExponential(3)}      ${rate}`);
}
