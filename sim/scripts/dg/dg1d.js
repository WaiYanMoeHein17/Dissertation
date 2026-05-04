// dg1d.js — 1D DG spatial operator for SWE on a periodic Cartesian mesh.
//
// Storage layout for the state vector q:
//   q is a flat Float64Array of length Nx * n * M
//   Element (i, ix, k) is at offset ((i * n) + ix) * M + k
//   where i = 0..Nx-1, ix = 0..n-1, k = 0..M-1, M = number of conserved vars (=2 for SWE).
//
// The spatial operator returns L(q) = -(2/dx) * dF/dxi + flux corrections
// using the strong-form nodal LGL DG with Rusanov surface flux.
//
// Optional face traces (qFaceL, qFaceR) override the use of instantaneous
// nodal endpoints. ADER-DG passes the time-averaged predictor traces here,
// while the volume term still uses q (which represents q^n).
//
// All inner loops avoid allocation: caller provides preallocated work buffers.

import { flux1D, rusanov1D } from "./swe.js";

export class Mesh1D {
  constructor(L, Nx, basis) {
    this.L = L; this.Nx = Nx;
    this.dx = L / Nx;
    this.basis = basis;
    this.n = basis.n;
    this.M = 2;          // SWE 1D: (h, hu)
    // Physical x-coordinates of every node, flat (Nx * n).
    this.xn = new Float64Array(Nx * basis.n);
    for (let i = 0; i < Nx; i++) {
      const xL = i * this.dx;
      for (let ix = 0; ix < basis.n; ix++) {
        this.xn[i * basis.n + ix] = xL + 0.5 * this.dx * (basis.x[ix] + 1.0);
      }
    }
  }
}

// Allocate scratch buffers for dgRhs1D — caller-owned, reusable across steps.
export function allocWork1D(mesh) {
  const { Nx, n, M } = mesh;
  return {
    Fvol  : new Float64Array(Nx * n * M),       // F(q) at every node
    qL    : new Float64Array(Nx * M),           // left  trace of each cell
    qR    : new Float64Array(Nx * M),           // right trace of each cell
    Fstar : new Float64Array(Nx * M),           // numerical flux at each face
    F_L_int: new Float64Array(Nx * M),          // F(qL_trace)
    F_R_int: new Float64Array(Nx * M),          // F(qR_trace)
  };
}

/**
 * Compute the DG semi-discrete RHS: dq/dt = L(q[, source]).
 *
 * @param {Float64Array} rhs        output, same shape as q (overwritten)
 * @param {Float64Array} q          input state, length Nx*n*M
 * @param {Mesh1D}       mesh
 * @param {object}       work       preallocated scratch (allocWork1D)
 * @param {number}       g          gravity
 * @param {Float64Array|null} qFaceL  optional left-trace state per cell  (Nx*M); null -> q[:,0,:]
 * @param {Float64Array|null} qFaceR  optional right-trace state per cell (Nx*M); null -> q[:,n-1,:]
 * @param {Float64Array|null} source  optional source contribution (same shape as q)
 */
export function dgRhs1D(rhs, q, mesh, work, g, qFaceL, qFaceR, source) {
  const { Nx, dx, n, M, basis } = mesh;
  const D = basis.D, w = basis.w;
  const { Fvol, qL, qR, Fstar, F_L_int, F_R_int } = work;

  // ---- Volume flux F(q) at every node ----
  for (let off = 0; off < Nx * n * M; off += M) flux1D(q, off, Fvol, off, g);

  // ---- -(2/dx) * D F  (nodal differentiation matrix applied per cell) ----
  const inv = 2.0 / dx;
  for (let i = 0; i < Nx; i++) {
    const cellBase = i * n * M;
    for (let ix = 0; ix < n; ix++) {
      let s0 = 0, s1 = 0;
      const Drow = D.subarray(ix * n, ix * n + n);
      for (let jx = 0; jx < n; jx++) {
        const fOff = cellBase + jx * M;
        const Dij = Drow[jx];
        s0 += Dij * Fvol[fOff];
        s1 += Dij * Fvol[fOff + 1];
      }
      const off = cellBase + ix * M;
      rhs[off]     = -inv * s0;
      rhs[off + 1] = -inv * s1;
    }
  }

  // ---- Face traces ----
  for (let i = 0; i < Nx; i++) {
    const lEnd = (i * n + 0)         * M;
    const rEnd = (i * n + (n - 1))   * M;
    if (qFaceL === null || qFaceL === undefined) {
      qL[i * M]     = q[lEnd];     qL[i * M + 1] = q[lEnd + 1];
    } else {
      qL[i * M]     = qFaceL[i * M];     qL[i * M + 1] = qFaceL[i * M + 1];
    }
    if (qFaceR === null || qFaceR === undefined) {
      qR[i * M]     = q[rEnd];     qR[i * M + 1] = q[rEnd + 1];
    } else {
      qR[i * M]     = qFaceR[i * M];     qR[i * M + 1] = qFaceR[i * M + 1];
    }
  }

  // F at the trace state, both endpoints
  for (let i = 0; i < Nx; i++) {
    flux1D(qL, i * M, F_L_int, i * M, g);
    flux1D(qR, i * M, F_R_int, i * M, g);
  }

  // ---- Numerical flux at each cell-face k (between cell k-1 and cell k, periodic) ----
  for (let k = 0; k < Nx; k++) {
    const kLeftCell = (k - 1 + Nx) % Nx;
    rusanov1D(qR, kLeftCell * M, qL, k * M, Fstar, k * M, g);
  }

  // ---- Strong-form correction at endpoints ----
  // q_t,0     += (2/dx)/w_0     * (F*_L - F_L)
  // q_t,n-1   += (2/dx)/w_{n-1} * (F_R - F*_R)
  const w0 = w[0], wn = w[n - 1];
  for (let i = 0; i < Nx; i++) {
    const left  = i;
    const right = (i + 1) % Nx;       // F* at right face of cell i = Fstar at face index right
    const offL = (i * n + 0)         * M;
    const offR = (i * n + (n - 1))   * M;
    rhs[offL]     += inv * (Fstar[left  * M]     - F_L_int[i * M])     / w0;
    rhs[offL + 1] += inv * (Fstar[left  * M + 1] - F_L_int[i * M + 1]) / w0;
    rhs[offR]     += inv * (F_R_int[i * M]     - Fstar[right * M])     / wn;
    rhs[offR + 1] += inv * (F_R_int[i * M + 1] - Fstar[right * M + 1]) / wn;
  }

  if (source) {
    for (let off = 0; off < rhs.length; off++) rhs[off] += source[off];
  }
}

// ---- Helpers: project a callable on the mesh, compute L2 error ----

export function projectField1D(mesh, fn, t = 0.0) {
  const { Nx, n, M, xn } = mesh;
  const out = new Float64Array(Nx * n * M);
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const x = xn[i * n + ix];
      const q = fn(x, t);                  // returns array of length M
      const off = (i * n + ix) * M;
      for (let k = 0; k < M; k++) out[off + k] = q[k];
    }
  }
  return out;
}

// L2 error using LGL quadrature (exact for polynomial degree 2p-1).
// Returns sqrt( int_{Omega} sum_k (q_k - qexact_k)^2 dx ).
export function l2Error1D(mesh, q, exactFn, t = 0.0) {
  const { Nx, n, M, xn, dx, basis } = mesh;
  const w = basis.w;
  let sum = 0;
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const x = xn[i * n + ix];
      const ex = exactFn(x, t);
      const off = (i * n + ix) * M;
      let local = 0;
      for (let k = 0; k < M; k++) {
        const e = q[off + k] - ex[k];
        local += e * e;
      }
      sum += local * w[ix] * 0.5 * dx;     // dx_phys/dxi = dx/2
    }
  }
  return Math.sqrt(sum);
}

// Component-wise max-norm error (h, hu separately) for diagnostics.
export function maxError1D(mesh, q, exactFn, t = 0.0) {
  const { Nx, n, M, xn } = mesh;
  const errs = new Float64Array(M);
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const x = xn[i * n + ix];
      const ex = exactFn(x, t);
      const off = (i * n + ix) * M;
      for (let k = 0; k < M; k++) {
        const e = Math.abs(q[off + k] - ex[k]);
        if (e > errs[k]) errs[k] = e;
      }
    }
  }
  return errs;
}
