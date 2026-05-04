// swe.js — Shallow water equations: physical flux, max wavespeed, Riemann.
//
// Conservative state q = (h, hu)            in 1D (m = 2)
//                  q = (h, hu, hv)         in 2D (m = 3)
//
// Operates on flat scalar arrays (Float64Array) where the trailing axis is
// the variable index. Helpers take pointers/strides explicitly so we avoid
// per-call allocations on the hot path.
//
// All routines assume flat bottom, no topography source. The user-facing
// MMS source is added externally in the DG operator.

export const G_DEFAULT = 1.0;

// ---------- 1D ----------

// flux1D(qIn, qInOff, fOut, fOutOff, g) — write F(q) to fOut at offset.
// q layout: [h, hu]
export function flux1D(qIn, qInOff, fOut, fOutOff, g) {
  const h = qIn[qInOff];
  const hu = qIn[qInOff + 1];
  const u = hu / h;
  fOut[fOutOff]     = hu;
  fOut[fOutOff + 1] = hu * u + 0.5 * g * h * h;
}

export function maxWavespeed1D(qL, qLOff, qR, qROff, g) {
  const hL = qL[qLOff], uL = qL[qLOff + 1] / hL;
  const hR = qR[qROff], uR = qR[qROff + 1] / hR;
  const cL = Math.sqrt(g * Math.max(hL, 1e-14));
  const cR = Math.sqrt(g * Math.max(hR, 1e-14));
  return Math.max(Math.abs(uL) + cL, Math.abs(uR) + cR);
}

// Local Lax-Friedrichs (Rusanov) flux: F* = 0.5(F_L+F_R) - 0.5*alpha*(qR-qL).
export function rusanov1D(qL, qLOff, qR, qROff, fOut, fOutOff, g) {
  const FL0_h  = qL[qLOff + 1];
  const FL1_h  = qL[qLOff + 1] * (qL[qLOff + 1] / qL[qLOff]) + 0.5 * g * qL[qLOff] * qL[qLOff];
  const FR0_h  = qR[qROff + 1];
  const FR1_h  = qR[qROff + 1] * (qR[qROff + 1] / qR[qROff]) + 0.5 * g * qR[qROff] * qR[qROff];
  const alpha  = maxWavespeed1D(qL, qLOff, qR, qROff, g);
  fOut[fOutOff]     = 0.5 * (FL0_h + FR0_h) - 0.5 * alpha * (qR[qROff]     - qL[qLOff]);
  fOut[fOutOff + 1] = 0.5 * (FL1_h + FR1_h) - 0.5 * alpha * (qR[qROff + 1] - qL[qLOff + 1]);
}

// ---------- 2D ----------

// fluxX_2D(q, qOff, F, FOff, g)
export function fluxX2D(qIn, qOff, F, FOff, g) {
  const h  = qIn[qOff];
  const hu = qIn[qOff + 1];
  const hv = qIn[qOff + 2];
  const u  = hu / h;
  F[FOff]     = hu;
  F[FOff + 1] = hu * u + 0.5 * g * h * h;
  F[FOff + 2] = hv * u;
}

export function fluxY2D(qIn, qOff, F, FOff, g) {
  const h  = qIn[qOff];
  const hu = qIn[qOff + 1];
  const hv = qIn[qOff + 2];
  const v  = hv / h;
  F[FOff]     = hv;
  F[FOff + 1] = hu * v;
  F[FOff + 2] = hv * v + 0.5 * g * h * h;
}

// Rotated Rusanov for face with outward normal (nx, ny).
export function rusanov2D(qL, qLOff, qR, qROff, fOut, fOutOff, nx, ny, g) {
  const hL = qL[qLOff], uL = qL[qLOff + 1] / hL, vL = qL[qLOff + 2] / hL;
  const hR = qR[qROff], uR = qR[qROff + 1] / hR, vR = qR[qROff + 2] / hR;
  // Physical fluxes
  const FL0 = qL[qLOff + 1] * nx + qL[qLOff + 2] * ny;
  const FL1 = (qL[qLOff + 1] * uL + 0.5 * g * hL * hL) * nx + (qL[qLOff + 1] * vL) * ny;
  const FL2 = (qL[qLOff + 2] * uL) * nx + (qL[qLOff + 2] * vL + 0.5 * g * hL * hL) * ny;
  const FR0 = qR[qROff + 1] * nx + qR[qROff + 2] * ny;
  const FR1 = (qR[qROff + 1] * uR + 0.5 * g * hR * hR) * nx + (qR[qROff + 1] * vR) * ny;
  const FR2 = (qR[qROff + 2] * uR) * nx + (qR[qROff + 2] * vR + 0.5 * g * hR * hR) * ny;
  const cL = Math.sqrt(g * Math.max(hL, 1e-14));
  const cR = Math.sqrt(g * Math.max(hR, 1e-14));
  const alpha = Math.max(Math.abs(uL * nx + vL * ny) + cL, Math.abs(uR * nx + vR * ny) + cR);
  fOut[fOutOff]     = 0.5 * (FL0 + FR0) - 0.5 * alpha * (qR[qROff]     - qL[qLOff]);
  fOut[fOutOff + 1] = 0.5 * (FL1 + FR1) - 0.5 * alpha * (qR[qROff + 1] - qL[qLOff + 1]);
  fOut[fOutOff + 2] = 0.5 * (FL2 + FR2) - 0.5 * alpha * (qR[qROff + 2] - qL[qLOff + 2]);
}

export function maxWavespeed2D(q, m, g) {
  // q is a flat array [h,hu,hv,h,hu,hv,...]
  let amax = 0;
  for (let off = 0; off < q.length; off += m) {
    const h = q[off];
    if (h <= 0) continue;
    const u = q[off + 1] / h, v = q[off + 2] / h;
    const c = Math.sqrt(g * h);
    const a = Math.max(Math.abs(u), Math.abs(v)) + c;
    if (a > amax) amax = a;
  }
  return amax;
}

export function maxWavespeed1Dflat(q, m, g) {
  let amax = 0;
  for (let off = 0; off < q.length; off += m) {
    const h = q[off];
    if (h <= 0) continue;
    const u = q[off + 1] / h;
    const c = Math.sqrt(g * h);
    const a = Math.abs(u) + c;
    if (a > amax) amax = a;
  }
  return amax;
}
