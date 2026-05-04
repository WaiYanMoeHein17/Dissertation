// bench_linswe.mjs — Linearised SWE convergence benchmark for ADER-DG and RK-DG.
//
// Self-contained: uses basis.js for LGL nodes/weights/D matrix, and swe_lin.js
// for the linear flux + Rusanov + closed-form standing-wave reference. The DG
// operator and time steppers are inlined here so that the existing nonlinear
// SWE benchmark (bench.js) is untouched.
//
// Usage:  node sim/scripts/dg/bench_linswe.mjs
//
// Output: convergence tables (text) plus LaTeX-ready blocks for the paper.

import { makeBasis } from "./basis.js";
import {
  makeLinPhys1D, makeLinPhys2D,
  standingWave1D, standingWave2D, M1, M2,
} from "./swe_lin.js";

// =====================================================================
// Mesh + DG operator (1D)
// =====================================================================

class Mesh1D {
  constructor(L, Nx, basis) {
    this.L = L; this.Nx = Nx; this.dx = L / Nx;
    this.basis = basis; this.n = basis.n; this.M = M1;
    this.xn = new Float64Array(Nx * basis.n);
    for (let i = 0; i < Nx; i++) {
      const xL = i * this.dx;
      for (let ix = 0; ix < basis.n; ix++) {
        this.xn[i * basis.n + ix] = xL + 0.5 * this.dx * (basis.x[ix] + 1.0);
      }
    }
  }
}

function projectField1D(mesh, fn, t = 0.0) {
  const { Nx, n, M, xn } = mesh;
  const out = new Float64Array(Nx * n * M);
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const q = fn(xn[i * n + ix], t);
      const off = (i * n + ix) * M;
      for (let k = 0; k < M; k++) out[off + k] = q[k];
    }
  }
  return out;
}

function l2Error1D(mesh, q, exactFn, t = 0.0) {
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
      sum += local * w[ix] * 0.5 * dx;
    }
  }
  return Math.sqrt(sum);
}

function dgRhs1D(rhs, q, mesh, work, phys) {
  const { Nx, dx, n, M, basis } = mesh;
  const D = basis.D, w = basis.w;
  const { Fvol, qL, qR, Fstar, F_L_int, F_R_int } = work;

  for (let off = 0; off < Nx * n * M; off += M) phys.flux1D(q, off, Fvol, off);

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

  for (let i = 0; i < Nx; i++) {
    const lEnd = (i * n + 0)         * M;
    const rEnd = (i * n + (n - 1))   * M;
    qL[i * M]     = q[lEnd];     qL[i * M + 1] = q[lEnd + 1];
    qR[i * M]     = q[rEnd];     qR[i * M + 1] = q[rEnd + 1];
    phys.flux1D(qL, i * M, F_L_int, i * M);
    phys.flux1D(qR, i * M, F_R_int, i * M);
  }
  for (let k = 0; k < Nx; k++) {
    const kLeft = (k - 1 + Nx) % Nx;
    phys.rusanov1D(qR, kLeft * M, qL, k * M, Fstar, k * M);
  }
  for (let i = 0; i < Nx; i++) {
    const left  = i;
    const right = (i + 1) % Nx;
    const offL = (i * n + 0) * M;
    const offR = (i * n + (n - 1)) * M;
    rhs[offL]     += inv * (Fstar[left  * M]     - F_L_int[i * M])     / w[0];
    rhs[offL + 1] += inv * (Fstar[left  * M + 1] - F_L_int[i * M + 1]) / w[0];
    rhs[offR]     += inv * (F_R_int[i * M]     - Fstar[right * M])     / w[n - 1];
    rhs[offR + 1] += inv * (F_R_int[i * M + 1] - Fstar[right * M + 1]) / w[n - 1];
  }
}

function allocWork1D(mesh) {
  const { Nx, n, M } = mesh;
  return {
    Fvol  : new Float64Array(Nx * n * M),
    qL    : new Float64Array(Nx * M),
    qR    : new Float64Array(Nx * M),
    Fstar : new Float64Array(Nx * M),
    F_L_int: new Float64Array(Nx * M),
    F_R_int: new Float64Array(Nx * M),
  };
}

// =====================================================================
// RK-DG and ADER-DG steppers (1D)
// =====================================================================

function makeRKDG1D(mesh, phys) {
  const work = allocWork1D(mesh);
  const N = mesh.Nx * mesh.n * mesh.M;
  const rhs = new Float64Array(N);
  const u1 = new Float64Array(N);
  const u2 = new Float64Array(N);
  const stepper = mesh.basis.p === 1 ? "SSPRK2" : "SSPRK3";
  function step(q, dt /*, t */) {
    dgRhs1D(rhs, q, mesh, work, phys);
    if (stepper === "SSPRK2") {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      dgRhs1D(rhs, u1, mesh, work, phys);
      for (let i = 0; i < N; i++) q[i] = 0.5 * q[i] + 0.5 * (u1[i] + dt * rhs[i]);
    } else {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      dgRhs1D(rhs, u1, mesh, work, phys);
      for (let i = 0; i < N; i++) u2[i] = 0.75 * q[i] + 0.25 * (u1[i] + dt * rhs[i]);
      dgRhs1D(rhs, u2, mesh, work, phys);
      for (let i = 0; i < N; i++) q[i] = (q[i] + 2 * (u2[i] + dt * rhs[i])) / 3;
    }
  }
  return { step, name: `RK-DG (${stepper})` };
}

// Small linear-algebra helper (Gauss-Jordan inverse of a small dense matrix).
function invertInPlace(A, n) {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r * n + col]) > Math.abs(A[pivot * n + col])) pivot = r;
    }
    if (pivot !== col) {
      for (let c = 0; c < n; c++) {
        let t = A[col * n + c]; A[col * n + c] = A[pivot * n + c]; A[pivot * n + c] = t;
        t = I[col * n + c]; I[col * n + c] = I[pivot * n + c]; I[pivot * n + c] = t;
      }
    }
    const piv = A[col * n + col];
    for (let c = 0; c < n; c++) { A[col * n + c] /= piv; I[col * n + c] /= piv; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r * n + col];
      if (f === 0) continue;
      for (let c = 0; c < n; c++) {
        A[r * n + c] -= f * A[col * n + c];
        I[r * n + c] -= f * I[col * n + c];
      }
    }
  }
  return I;
}

function makeADERDG1D(mesh, phys, picardIters = -1) {
  const { Nx, n, M, dx, basis } = mesh;
  const T = n;
  const p = basis.p;
  // LGL on [0,1]: tau_l = (xi_l+1)/2, weights wt_l = w_l/2.
  const tau = new Float64Array(T);
  const wt  = new Float64Array(T);
  for (let i = 0; i < T; i++) { tau[i] = 0.5 * (basis.x[i] + 1); wt[i] = 0.5 * basis.w[i]; }
  // Time differentiation matrix on [0,1] = 2 * D on [-1,1].
  const Dtau = new Float64Array(T * T);
  for (let i = 0; i < T * T; i++) Dtau[i] = 2 * basis.D[i];
  const Dred = new Float64Array((T - 1) * (T - 1));
  const D0   = new Float64Array(T - 1);
  for (let m = 1; m < T; m++) {
    D0[m - 1] = Dtau[m * T + 0];
    for (let l = 1; l < T; l++) Dred[(m - 1) * (T - 1) + (l - 1)] = Dtau[m * T + l];
  }
  const Dred_inv = invertInPlace(Dred.slice(), T - 1);
  const nIter = picardIters < 0 ? p + 1 : picardIters;

  const Q     = new Float64Array(Nx * T * n * M);
  const Fpred = new Float64Array(Nx * T * n * M);
  const R     = new Float64Array(Nx * T * n * M);
  const RHSm  = new Float64Array(Nx * (T - 1) * n * M);
  const FbarV = new Float64Array(Nx * n * M);
  const qBarL = new Float64Array(Nx * M);
  const qBarR = new Float64Array(Nx * M);
  const FbarL_int = new Float64Array(Nx * M);
  const FbarR_int = new Float64Array(Nx * M);
  const FbarStar  = new Float64Array(Nx * M);
  const tmpF  = new Float64Array(M);
  const inv2dx = 2.0 / dx;
  const w = basis.w;

  function step(qIn, dt /*, tNow */) {
    // 1) Init Q[i, l, ix, :] = q^n[i, ix, :].
    for (let i = 0; i < Nx; i++) {
      for (let l = 0; l < T; l++) {
        for (let ix = 0; ix < n; ix++) {
          const dst = ((i * T + l) * n + ix) * M;
          const src = (i * n + ix) * M;
          for (let k = 0; k < M; k++) Q[dst + k] = qIn[src + k];
        }
      }
    }
    // 2) Picard iteration.
    for (let iter = 0; iter < nIter; iter++) {
      for (let off = 0; off < Nx * T * n * M; off += M) phys.flux1D(Q, off, Fpred, off);
      // R[i, l, ix, :] = -(2/dx) (D F)[i, l, ix, :]   for l = 1..T-1
      for (let i = 0; i < Nx; i++) {
        for (let l = 1; l < T; l++) {
          for (let ix = 0; ix < n; ix++) {
            let s0 = 0, s1 = 0;
            for (let jx = 0; jx < n; jx++) {
              const Dij = basis.D[ix * n + jx];
              const fOff = ((i * T + l) * n + jx) * M;
              s0 += Dij * Fpred[fOff];
              s1 += Dij * Fpred[fOff + 1];
            }
            const rOff = ((i * T + l) * n + ix) * M;
            R[rOff]     = -inv2dx * s0;
            R[rOff + 1] = -inv2dx * s1;
          }
        }
      }
      // RHSm[i, m-1, ix, :] = dt * R[i, m, ix, :] - Dτ[m, 0] * q^n[i, ix, :].
      for (let i = 0; i < Nx; i++) {
        for (let m = 1; m < T; m++) {
          const D0m = D0[m - 1];
          for (let ix = 0; ix < n; ix++) {
            const rOff = ((i * T + m) * n + ix) * M;
            const rhsOff = ((i * (T - 1) + (m - 1)) * n + ix) * M;
            const qnOff = (i * n + ix) * M;
            RHSm[rhsOff]     = dt * R[rOff]     - D0m * qIn[qnOff];
            RHSm[rhsOff + 1] = dt * R[rOff + 1] - D0m * qIn[qnOff + 1];
          }
        }
      }
      // Q[i, 1..T-1, ix, :] = Dred_inv @ RHSm.
      for (let i = 0; i < Nx; i++) {
        for (let ix = 0; ix < n; ix++) {
          for (let m = 0; m < T - 1; m++) {
            let s0 = 0, s1 = 0;
            for (let l = 0; l < T - 1; l++) {
              const Dij = Dred_inv[m * (T - 1) + l];
              const rhsOff = ((i * (T - 1) + l) * n + ix) * M;
              s0 += Dij * RHSm[rhsOff];
              s1 += Dij * RHSm[rhsOff + 1];
            }
            const qOff = ((i * T + (m + 1)) * n + ix) * M;
            Q[qOff]     = s0;
            Q[qOff + 1] = s1;
          }
        }
      }
    }
    // 3) Time averages.
    for (let off = 0; off < Nx * T * n * M; off += M) phys.flux1D(Q, off, Fpred, off);
    for (let i = 0; i < Nx; i++) {
      for (let ix = 0; ix < n; ix++) {
        let s0 = 0, s1 = 0;
        for (let l = 0; l < T; l++) {
          const fOff = ((i * T + l) * n + ix) * M;
          s0 += wt[l] * Fpred[fOff];
          s1 += wt[l] * Fpred[fOff + 1];
        }
        const off = (i * n + ix) * M;
        FbarV[off]     = s0;
        FbarV[off + 1] = s1;
      }
    }
    for (let i = 0; i < Nx; i++) {
      let qL0 = 0, qL1 = 0, qR0 = 0, qR1 = 0;
      let FL0 = 0, FL1 = 0, FR0 = 0, FR1 = 0;
      for (let l = 0; l < T; l++) {
        const lOff = ((i * T + l) * n + 0)         * M;
        const rOff = ((i * T + l) * n + (n - 1))   * M;
        qL0 += wt[l] * Q[lOff];      qL1 += wt[l] * Q[lOff + 1];
        qR0 += wt[l] * Q[rOff];      qR1 += wt[l] * Q[rOff + 1];
        FL0 += wt[l] * Fpred[lOff];  FL1 += wt[l] * Fpred[lOff + 1];
        FR0 += wt[l] * Fpred[rOff];  FR1 += wt[l] * Fpred[rOff + 1];
      }
      qBarL[i * M]     = qL0; qBarL[i * M + 1] = qL1;
      qBarR[i * M]     = qR0; qBarR[i * M + 1] = qR1;
      FbarL_int[i * M] = FL0; FbarL_int[i * M + 1] = FL1;
      FbarR_int[i * M] = FR0; FbarR_int[i * M + 1] = FR1;
    }
    for (let k = 0; k < Nx; k++) {
      const kLeft = (k - 1 + Nx) % Nx;
      let s0 = 0, s1 = 0;
      for (let l = 0; l < T; l++) {
        const lLOff = ((kLeft * T + l) * n + (n - 1)) * M;
        const lROff = ((k     * T + l) * n + 0)       * M;
        phys.rusanov1D(Q, lLOff, Q, lROff, tmpF, 0);
        s0 += wt[l] * tmpF[0];
        s1 += wt[l] * tmpF[1];
      }
      FbarStar[k * M]     = s0;
      FbarStar[k * M + 1] = s1;
    }
    // 4) Corrector update.
    for (let i = 0; i < Nx; i++) {
      const cellBase = i * n * M;
      for (let ix = 0; ix < n; ix++) {
        let s0 = 0, s1 = 0;
        for (let jx = 0; jx < n; jx++) {
          const fOff = cellBase + jx * M;
          const Dij = basis.D[ix * n + jx];
          s0 += Dij * FbarV[fOff];
          s1 += Dij * FbarV[fOff + 1];
        }
        const off = cellBase + ix * M;
        let r0 = -inv2dx * s0;
        let r1 = -inv2dx * s1;
        if (ix === 0) {
          const left = i;
          r0 += inv2dx * (FbarStar[left * M]     - FbarL_int[i * M])     / w[0];
          r1 += inv2dx * (FbarStar[left * M + 1] - FbarL_int[i * M + 1]) / w[0];
        }
        if (ix === n - 1) {
          const right = (i + 1) % Nx;
          r0 += inv2dx * (FbarR_int[i * M]     - FbarStar[right * M])     / w[n - 1];
          r1 += inv2dx * (FbarR_int[i * M + 1] - FbarStar[right * M + 1]) / w[n - 1];
        }
        qIn[off]     += dt * r0;
        qIn[off + 1] += dt * r1;
      }
    }
  }
  return { step, name: `ADER-DG (P${p}, ${nIter} Picard sweeps)` };
}

// =====================================================================
// 2D mesh + DG operator + steppers (parallel structure to 1D)
// =====================================================================

class Mesh2D {
  constructor(L, Nx, Ny, basis) {
    this.L = L; this.Nx = Nx; this.Ny = Ny;
    this.dx = L / Nx; this.dy = L / Ny;
    this.basis = basis; this.n = basis.n; this.M = M2;
    const n = basis.n;
    this.xn = new Float64Array(Nx * Ny * n * n);
    this.yn = new Float64Array(Nx * Ny * n * n);
    for (let iy = 0; iy < Ny; iy++) {
      const yL = iy * this.dy;
      for (let ix = 0; ix < Nx; ix++) {
        const xL = ix * this.dx;
        for (let jy = 0; jy < n; jy++) {
          const y = yL + 0.5 * this.dy * (basis.x[jy] + 1.0);
          for (let jx = 0; jx < n; jx++) {
            const x = xL + 0.5 * this.dx * (basis.x[jx] + 1.0);
            const idx = ((iy * Nx + ix) * n + jy) * n + jx;
            this.xn[idx] = x;
            this.yn[idx] = y;
          }
        }
      }
    }
  }
  elemBase(ix, iy) { return ((iy * this.Nx + ix) * this.n * this.n) * this.M; }
}

function projectField2D(mesh, fn, t = 0.0) {
  const { Nx, Ny, n, M, xn, yn } = mesh;
  const out = new Float64Array(Nx * Ny * n * n * M);
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      for (let jy = 0; jy < n; jy++) {
        for (let jx = 0; jx < n; jx++) {
          const idx = ((iy * Nx + ix) * n + jy) * n + jx;
          const q = fn(xn[idx], yn[idx], t);
          const off = idx * M;
          for (let k = 0; k < M; k++) out[off + k] = q[k];
        }
      }
    }
  }
  return out;
}

function l2Error2D(mesh, q, exactFn, t = 0.0) {
  const { Nx, Ny, n, M, xn, yn, dx, dy, basis } = mesh;
  const w = basis.w;
  let sum = 0;
  const jw = dx * dy * 0.25;
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      for (let jy = 0; jy < n; jy++) {
        for (let jx = 0; jx < n; jx++) {
          const idx = ((iy * Nx + ix) * n + jy) * n + jx;
          const ex = exactFn(xn[idx], yn[idx], t);
          const off = idx * M;
          let local = 0;
          for (let k = 0; k < M; k++) {
            const e = q[off + k] - ex[k];
            local += e * e;
          }
          sum += local * w[jx] * w[jy] * jw;
        }
      }
    }
  }
  return Math.sqrt(sum);
}

function dgRhs2D(rhs, q, mesh, work, phys) {
  const { Nx, Ny, n, M, dx, dy, basis } = mesh;
  const D = basis.D, w = basis.w;
  const { Fxv, Fyv, qW, qE, qS, qN, FstarX, FstarY, FxW, FxE, FyS, FyN } = work;

  for (let off = 0; off < Nx * Ny * n * n * M; off += M) {
    phys.fluxX2D(q, off, Fxv, off);
    phys.fluxY2D(q, off, Fyv, off);
  }
  const invX = 2.0 / dx, invY = 2.0 / dy;
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = ((iy * Nx + ix) * n * n) * M;
      for (let jy = 0; jy < n; jy++) {
        const rowBase = eb + jy * n * M;
        for (let jx = 0; jx < n; jx++) {
          let s0 = 0, s1 = 0, s2 = 0;
          for (let kx = 0; kx < n; kx++) {
            const Dj = D[jx * n + kx];
            const fOff = rowBase + kx * M;
            s0 += Dj * Fxv[fOff];
            s1 += Dj * Fxv[fOff + 1];
            s2 += Dj * Fxv[fOff + 2];
          }
          const off = rowBase + jx * M;
          rhs[off]     = -invX * s0;
          rhs[off + 1] = -invX * s1;
          rhs[off + 2] = -invX * s2;
        }
      }
      for (let jx = 0; jx < n; jx++) {
        for (let jy = 0; jy < n; jy++) {
          let s0 = 0, s1 = 0, s2 = 0;
          for (let ky = 0; ky < n; ky++) {
            const Dj = D[jy * n + ky];
            const fOff = eb + (ky * n + jx) * M;
            s0 += Dj * Fyv[fOff];
            s1 += Dj * Fyv[fOff + 1];
            s2 += Dj * Fyv[fOff + 2];
          }
          const off = eb + (jy * n + jx) * M;
          rhs[off]     -= invY * s0;
          rhs[off + 1] -= invY * s1;
          rhs[off + 2] -= invY * s2;
        }
      }
    }
  }
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = mesh.elemBase(ix, iy);
      for (let j = 0; j < n; j++) {
        const off = ((iy * Nx + ix) * n + j) * M;
        const wOff = eb + (j * n + 0) * M;
        const eOff = eb + (j * n + (n - 1)) * M;
        const sOff = eb + (0 * n + j) * M;
        const nOff = eb + ((n - 1) * n + j) * M;
        qW[off]   = q[wOff]; qW[off+1] = q[wOff+1]; qW[off+2] = q[wOff+2];
        qE[off]   = q[eOff]; qE[off+1] = q[eOff+1]; qE[off+2] = q[eOff+2];
        qS[off]   = q[sOff]; qS[off+1] = q[sOff+1]; qS[off+2] = q[sOff+2];
        qN[off]   = q[nOff]; qN[off+1] = q[nOff+1]; qN[off+2] = q[nOff+2];
        phys.fluxX2D(qW, off, FxW, off);
        phys.fluxX2D(qE, off, FxE, off);
        phys.fluxY2D(qS, off, FyS, off);
        phys.fluxY2D(qN, off, FyN, off);
      }
    }
  }
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const ixL = (ix - 1 + Nx) % Nx;
      for (let j = 0; j < n; j++) {
        const offL = ((iy * Nx + ixL) * n + j) * M;
        const offR = ((iy * Nx + ix)  * n + j) * M;
        phys.rusanov2D(qE, offL, qW, offR, FstarX, offR, 1, 0);
      }
    }
  }
  for (let iy = 0; iy < Ny; iy++) {
    const iyB = (iy - 1 + Ny) % Ny;
    for (let ix = 0; ix < Nx; ix++) {
      for (let j = 0; j < n; j++) {
        const offB = ((iyB * Nx + ix) * n + j) * M;
        const offT = ((iy  * Nx + ix) * n + j) * M;
        phys.rusanov2D(qN, offB, qS, offT, FstarY, offT, 0, 1);
      }
    }
  }
  const w0 = w[0], wn = w[n - 1];
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = mesh.elemBase(ix, iy);
      const ixR = (ix + 1) % Nx;
      const iyT = (iy + 1) % Ny;
      for (let j = 0; j < n; j++) {
        const offW = ((iy * Nx + ix) * n + j) * M;
        const offE = ((iy * Nx + ixR) * n + j) * M;
        const offS = ((iy * Nx + ix) * n + j) * M;
        const offN = ((iyT * Nx + ix) * n + j) * M;
        const wNode = eb + (j * n + 0) * M;
        rhs[wNode]     += invX * (FstarX[offW]     - FxW[offW])     / w0;
        rhs[wNode + 1] += invX * (FstarX[offW + 1] - FxW[offW + 1]) / w0;
        rhs[wNode + 2] += invX * (FstarX[offW + 2] - FxW[offW + 2]) / w0;
        const eNode = eb + (j * n + (n - 1)) * M;
        rhs[eNode]     += invX * (FxE[offW]     - FstarX[offE])     / wn;
        rhs[eNode + 1] += invX * (FxE[offW + 1] - FstarX[offE + 1]) / wn;
        rhs[eNode + 2] += invX * (FxE[offW + 2] - FstarX[offE + 2]) / wn;
        const sNode = eb + (0 * n + j) * M;
        rhs[sNode]     += invY * (FstarY[offS]     - FyS[offS])     / w0;
        rhs[sNode + 1] += invY * (FstarY[offS + 1] - FyS[offS + 1]) / w0;
        rhs[sNode + 2] += invY * (FstarY[offS + 2] - FyS[offS + 2]) / w0;
        const nNode = eb + ((n - 1) * n + j) * M;
        rhs[nNode]     += invY * (FyN[offS]     - FstarY[offN])     / wn;
        rhs[nNode + 1] += invY * (FyN[offS + 1] - FstarY[offN + 1]) / wn;
        rhs[nNode + 2] += invY * (FyN[offS + 2] - FstarY[offN + 2]) / wn;
      }
    }
  }
}

function allocWork2D(mesh) {
  const { Nx, Ny, n, M } = mesh;
  return {
    Fxv : new Float64Array(Nx * Ny * n * n * M),
    Fyv : new Float64Array(Nx * Ny * n * n * M),
    qW  : new Float64Array(Nx * Ny * n * M),
    qE  : new Float64Array(Nx * Ny * n * M),
    qS  : new Float64Array(Nx * Ny * n * M),
    qN  : new Float64Array(Nx * Ny * n * M),
    FstarX : new Float64Array(Nx * Ny * n * M),
    FstarY : new Float64Array(Nx * Ny * n * M),
    FxW : new Float64Array(Nx * Ny * n * M),
    FxE : new Float64Array(Nx * Ny * n * M),
    FyS : new Float64Array(Nx * Ny * n * M),
    FyN : new Float64Array(Nx * Ny * n * M),
  };
}

function makeRKDG2D(mesh, phys) {
  const work = allocWork2D(mesh);
  const N = mesh.Nx * mesh.Ny * mesh.n * mesh.n * mesh.M;
  const rhs = new Float64Array(N);
  const u1 = new Float64Array(N);
  const u2 = new Float64Array(N);
  const stepper = mesh.basis.p === 1 ? "SSPRK2" : "SSPRK3";
  function step(q, dt) {
    dgRhs2D(rhs, q, mesh, work, phys);
    if (stepper === "SSPRK2") {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      dgRhs2D(rhs, u1, mesh, work, phys);
      for (let i = 0; i < N; i++) q[i] = 0.5 * q[i] + 0.5 * (u1[i] + dt * rhs[i]);
    } else {
      for (let i = 0; i < N; i++) u1[i] = q[i] + dt * rhs[i];
      dgRhs2D(rhs, u1, mesh, work, phys);
      for (let i = 0; i < N; i++) u2[i] = 0.75 * q[i] + 0.25 * (u1[i] + dt * rhs[i]);
      dgRhs2D(rhs, u2, mesh, work, phys);
      for (let i = 0; i < N; i++) q[i] = (q[i] + 2 * (u2[i] + dt * rhs[i])) / 3;
    }
  }
  return { step, name: `RK-DG (${stepper})` };
}

function makeADERDG2D(mesh, phys, picardIters = -1) {
  const { Nx, Ny, n, M, dx, dy, basis } = mesh;
  const T = n;
  const p = basis.p;
  const tau = new Float64Array(T);
  const wt  = new Float64Array(T);
  for (let i = 0; i < T; i++) { tau[i] = 0.5 * (basis.x[i] + 1); wt[i] = 0.5 * basis.w[i]; }
  const Dtau = new Float64Array(T * T);
  for (let i = 0; i < T * T; i++) Dtau[i] = 2 * basis.D[i];
  const Dred = new Float64Array((T - 1) * (T - 1));
  const D0   = new Float64Array(T - 1);
  for (let m = 1; m < T; m++) {
    D0[m - 1] = Dtau[m * T + 0];
    for (let l = 1; l < T; l++) Dred[(m - 1) * (T - 1) + (l - 1)] = Dtau[m * T + l];
  }
  const Dred_inv = invertInPlace(Dred.slice(), T - 1);
  const nIter = picardIters < 0 ? p + 1 : picardIters;
  const Ne = Nx * Ny;
  const inv2dx = 2.0 / dx, inv2dy = 2.0 / dy;
  const w = basis.w;

  const Q     = new Float64Array(Ne * T * n * n * M);
  const Fxv   = new Float64Array(Ne * T * n * n * M);
  const Fyv   = new Float64Array(Ne * T * n * n * M);
  const R     = new Float64Array(Ne * T * n * n * M);
  const RHSm  = new Float64Array(Ne * (T - 1) * n * n * M);
  const FbarVx = new Float64Array(Ne * n * n * M);
  const FbarVy = new Float64Array(Ne * n * n * M);
  const FxBarW = new Float64Array(Ne * n * M);
  const FxBarE = new Float64Array(Ne * n * M);
  const FyBarS = new Float64Array(Ne * n * M);
  const FyBarN = new Float64Array(Ne * n * M);
  const FstarX = new Float64Array(Ne * n * M);
  const FstarY = new Float64Array(Ne * n * M);
  const tmpF  = new Float64Array(M);

  function step(qIn, dt) {
    for (let e = 0; e < Ne; e++) {
      for (let l = 0; l < T; l++) {
        const dstE = ((e * T + l) * n * n) * M;
        const srcE = e * n * n * M;
        Q.set(qIn.subarray(srcE, srcE + n * n * M), dstE);
      }
    }
    for (let iter = 0; iter < nIter; iter++) {
      for (let off = 0; off < Ne * T * n * n * M; off += M) {
        phys.fluxX2D(Q, off, Fxv, off);
        phys.fluxY2D(Q, off, Fyv, off);
      }
      for (let e = 0; e < Ne; e++) {
        for (let l = 1; l < T; l++) {
          const eb = ((e * T + l) * n * n) * M;
          for (let jy = 0; jy < n; jy++) {
            for (let jx = 0; jx < n; jx++) {
              let s0 = 0, s1 = 0, s2 = 0;
              for (let kx = 0; kx < n; kx++) {
                const Dij = basis.D[jx * n + kx];
                const fOff = eb + (jy * n + kx) * M;
                s0 += Dij * Fxv[fOff]; s1 += Dij * Fxv[fOff + 1]; s2 += Dij * Fxv[fOff + 2];
              }
              s0 *= -inv2dx; s1 *= -inv2dx; s2 *= -inv2dx;
              let t0 = 0, t1 = 0, t2 = 0;
              for (let ky = 0; ky < n; ky++) {
                const Dij = basis.D[jy * n + ky];
                const fOff = eb + (ky * n + jx) * M;
                t0 += Dij * Fyv[fOff]; t1 += Dij * Fyv[fOff + 1]; t2 += Dij * Fyv[fOff + 2];
              }
              s0 -= inv2dy * t0; s1 -= inv2dy * t1; s2 -= inv2dy * t2;
              const rOff = eb + (jy * n + jx) * M;
              R[rOff] = s0; R[rOff + 1] = s1; R[rOff + 2] = s2;
            }
          }
        }
      }
      for (let e = 0; e < Ne; e++) {
        for (let m = 1; m < T; m++) {
          const D0m = D0[m - 1];
          const eRBase = ((e * T + m) * n * n) * M;
          const eRhsBase = ((e * (T - 1) + (m - 1)) * n * n) * M;
          const qnBase = e * n * n * M;
          for (let p2 = 0; p2 < n * n; p2++) {
            const off = p2 * M;
            for (let k = 0; k < M; k++) {
              RHSm[eRhsBase + off + k] = dt * R[eRBase + off + k] - D0m * qIn[qnBase + off + k];
            }
          }
        }
      }
      for (let e = 0; e < Ne; e++) {
        for (let p2 = 0; p2 < n * n; p2++) {
          for (let mm = 0; mm < T - 1; mm++) {
            let s0 = 0, s1 = 0, s2 = 0;
            for (let l = 0; l < T - 1; l++) {
              const Dij = Dred_inv[mm * (T - 1) + l];
              const rhsOff = ((e * (T - 1) + l) * n * n + p2) * M;
              s0 += Dij * RHSm[rhsOff]; s1 += Dij * RHSm[rhsOff + 1]; s2 += Dij * RHSm[rhsOff + 2];
            }
            const qOff = ((e * T + (mm + 1)) * n * n + p2) * M;
            Q[qOff] = s0; Q[qOff + 1] = s1; Q[qOff + 2] = s2;
          }
        }
      }
    }
    for (let off = 0; off < Ne * T * n * n * M; off += M) {
      phys.fluxX2D(Q, off, Fxv, off);
      phys.fluxY2D(Q, off, Fyv, off);
    }
    FbarVx.fill(0); FbarVy.fill(0);
    for (let e = 0; e < Ne; e++) {
      for (let p2 = 0; p2 < n * n; p2++) {
        let sx0 = 0, sx1 = 0, sx2 = 0, sy0 = 0, sy1 = 0, sy2 = 0;
        for (let l = 0; l < T; l++) {
          const off = ((e * T + l) * n * n + p2) * M;
          const wl = wt[l];
          sx0 += wl * Fxv[off]; sx1 += wl * Fxv[off + 1]; sx2 += wl * Fxv[off + 2];
          sy0 += wl * Fyv[off]; sy1 += wl * Fyv[off + 1]; sy2 += wl * Fyv[off + 2];
        }
        const off = (e * n * n + p2) * M;
        FbarVx[off]     = sx0; FbarVx[off + 1] = sx1; FbarVx[off + 2] = sx2;
        FbarVy[off]     = sy0; FbarVy[off + 1] = sy1; FbarVy[off + 2] = sy2;
      }
    }
    FxBarW.fill(0); FxBarE.fill(0); FyBarS.fill(0); FyBarN.fill(0);
    for (let e = 0; e < Ne; e++) {
      for (let j = 0; j < n; j++) {
        let FW0 = 0, FW1 = 0, FW2 = 0, FE0 = 0, FE1 = 0, FE2 = 0;
        let FS0 = 0, FS1 = 0, FS2 = 0, FN0 = 0, FN1 = 0, FN2 = 0;
        for (let l = 0; l < T; l++) {
          const wl = wt[l];
          const eb = ((e * T + l) * n * n) * M;
          const oW = eb + (j * n + 0) * M;
          const oE = eb + (j * n + (n - 1)) * M;
          const oS = eb + (0 * n + j) * M;
          const oN = eb + ((n - 1) * n + j) * M;
          FW0 += wl * Fxv[oW]; FW1 += wl * Fxv[oW + 1]; FW2 += wl * Fxv[oW + 2];
          FE0 += wl * Fxv[oE]; FE1 += wl * Fxv[oE + 1]; FE2 += wl * Fxv[oE + 2];
          FS0 += wl * Fyv[oS]; FS1 += wl * Fyv[oS + 1]; FS2 += wl * Fyv[oS + 2];
          FN0 += wl * Fyv[oN]; FN1 += wl * Fyv[oN + 1]; FN2 += wl * Fyv[oN + 2];
        }
        const off = (e * n + j) * M;
        FxBarW[off] = FW0; FxBarW[off + 1] = FW1; FxBarW[off + 2] = FW2;
        FxBarE[off] = FE0; FxBarE[off + 1] = FE1; FxBarE[off + 2] = FE2;
        FyBarS[off] = FS0; FyBarS[off + 1] = FS1; FyBarS[off + 2] = FS2;
        FyBarN[off] = FN0; FyBarN[off + 1] = FN1; FyBarN[off + 2] = FN2;
      }
    }
    for (let iy = 0; iy < Ny; iy++) {
      for (let ix = 0; ix < Nx; ix++) {
        const ixL = (ix - 1 + Nx) % Nx;
        const eL = iy * Nx + ixL;
        const eR = iy * Nx + ix;
        for (let j = 0; j < n; j++) {
          let s0 = 0, s1 = 0, s2 = 0;
          for (let l = 0; l < T; l++) {
            const wl = wt[l];
            const oL = ((eL * T + l) * n * n + j * n + (n - 1)) * M;
            const oR = ((eR * T + l) * n * n + j * n + 0)       * M;
            phys.rusanov2D(Q, oL, Q, oR, tmpF, 0, 1, 0);
            s0 += wl * tmpF[0]; s1 += wl * tmpF[1]; s2 += wl * tmpF[2];
          }
          const off = (eR * n + j) * M;
          FstarX[off] = s0; FstarX[off + 1] = s1; FstarX[off + 2] = s2;
        }
      }
    }
    for (let iy = 0; iy < Ny; iy++) {
      const iyB = (iy - 1 + Ny) % Ny;
      for (let ix = 0; ix < Nx; ix++) {
        const eB = iyB * Nx + ix;
        const eT = iy  * Nx + ix;
        for (let j = 0; j < n; j++) {
          let s0 = 0, s1 = 0, s2 = 0;
          for (let l = 0; l < T; l++) {
            const wl = wt[l];
            const oB = ((eB * T + l) * n * n + (n - 1) * n + j) * M;
            const oTt = ((eT * T + l) * n * n + 0       * n + j) * M;
            phys.rusanov2D(Q, oB, Q, oTt, tmpF, 0, 0, 1);
            s0 += wl * tmpF[0]; s1 += wl * tmpF[1]; s2 += wl * tmpF[2];
          }
          const off = (eT * n + j) * M;
          FstarY[off] = s0; FstarY[off + 1] = s1; FstarY[off + 2] = s2;
        }
      }
    }
    for (let iy = 0; iy < Ny; iy++) {
      for (let ix = 0; ix < Nx; ix++) {
        const e = iy * Nx + ix;
        const eb = e * n * n * M;
        const ixR = (ix + 1) % Nx;
        const iyT = (iy + 1) % Ny;
        for (let jy = 0; jy < n; jy++) {
          for (let jx = 0; jx < n; jx++) {
            let sx0 = 0, sx1 = 0, sx2 = 0, sy0 = 0, sy1 = 0, sy2 = 0;
            for (let kx = 0; kx < n; kx++) {
              const Dij = basis.D[jx * n + kx];
              const fOff = eb + (jy * n + kx) * M;
              sx0 += Dij * FbarVx[fOff]; sx1 += Dij * FbarVx[fOff + 1]; sx2 += Dij * FbarVx[fOff + 2];
            }
            for (let ky = 0; ky < n; ky++) {
              const Dij = basis.D[jy * n + ky];
              const fOff = eb + (ky * n + jx) * M;
              sy0 += Dij * FbarVy[fOff]; sy1 += Dij * FbarVy[fOff + 1]; sy2 += Dij * FbarVy[fOff + 2];
            }
            const off = eb + (jy * n + jx) * M;
            qIn[off]     += dt * (-inv2dx * sx0 - inv2dy * sy0);
            qIn[off + 1] += dt * (-inv2dx * sx1 - inv2dy * sy1);
            qIn[off + 2] += dt * (-inv2dx * sx2 - inv2dy * sy2);
          }
        }
        const eR = iy  * Nx + ixR;
        const eN = iyT * Nx + ix;
        for (let j = 0; j < n; j++) {
          const offW = (e  * n + j) * M;
          const offE = (eR * n + j) * M;
          const offS = (e  * n + j) * M;
          const offN = (eN * n + j) * M;
          const wNode = eb + (j * n + 0) * M;
          qIn[wNode]     += dt * inv2dx * (FstarX[offW]     - FxBarW[offW])     / w[0];
          qIn[wNode + 1] += dt * inv2dx * (FstarX[offW + 1] - FxBarW[offW + 1]) / w[0];
          qIn[wNode + 2] += dt * inv2dx * (FstarX[offW + 2] - FxBarW[offW + 2]) / w[0];
          const eNode = eb + (j * n + (n - 1)) * M;
          qIn[eNode]     += dt * inv2dx * (FxBarE[offW]     - FstarX[offE])     / w[n - 1];
          qIn[eNode + 1] += dt * inv2dx * (FxBarE[offW + 1] - FstarX[offE + 1]) / w[n - 1];
          qIn[eNode + 2] += dt * inv2dx * (FxBarE[offW + 2] - FstarX[offE + 2]) / w[n - 1];
          const sNode = eb + (0 * n + j) * M;
          qIn[sNode]     += dt * inv2dy * (FstarY[offS]     - FyBarS[offS])     / w[0];
          qIn[sNode + 1] += dt * inv2dy * (FstarY[offS + 1] - FyBarS[offS + 1]) / w[0];
          qIn[sNode + 2] += dt * inv2dy * (FstarY[offS + 2] - FyBarS[offS + 2]) / w[0];
          const nNode = eb + ((n - 1) * n + j) * M;
          qIn[nNode]     += dt * inv2dy * (FyBarN[offS]     - FstarY[offN])     / w[n - 1];
          qIn[nNode + 1] += dt * inv2dy * (FyBarN[offS + 1] - FstarY[offN + 1]) / w[n - 1];
          qIn[nNode + 2] += dt * inv2dy * (FyBarN[offS + 2] - FstarY[offN + 2]) / w[n - 1];
        }
      }
    }
  }
  return { step, name: `ADER-DG (P${p}, ${nIter} Picard sweeps)` };
}

// =====================================================================
// Convergence drivers
// =====================================================================

function dtFor(p, dx, c0, CFL, scheme, schemeOrder) {
  let dt = CFL * dx / ((2 * p + 1) * c0);
  if (scheme === "rk" && p + 1 > schemeOrder) {
    const exponent = (p + 1) / schemeOrder;
    dt *= Math.pow(dx, exponent - 1);
  }
  return dt;
}

function rkOrder(p) { return p === 1 ? 2 : 3; }

function runConv1D({ p, NList, scheme, T, CFL, A, He, g, L }) {
  const basis = makeBasis(p);
  const phys = makeLinPhys1D({ g, He });
  const sw = standingWave1D({ L, A, g, He });
  const dxs = [], errs = [], runtimes = [], steps = [];
  for (const Nx of NList) {
    const mesh = new Mesh1D(L, Nx, basis);
    const q = projectField1D(mesh, sw.exact, 0.0);
    const stepper = scheme === "rk" ? makeRKDG1D(mesh, phys) : makeADERDG1D(mesh, phys);
    const order = scheme === "rk" ? rkOrder(p) : (p + 1);
    const t0 = performance.now();
    let t = 0, nSteps = 0;
    while (t < T - 1e-15) {
      let dt = dtFor(p, mesh.dx, phys.c0, CFL, scheme, order);
      if (t + dt > T) dt = T - t;
      stepper.step(q, dt, t);
      t += dt;
      nSteps++;
    }
    const wall = performance.now() - t0;
    dxs.push(mesh.dx);
    errs.push(l2Error1D(mesh, q, sw.exact, T));
    runtimes.push(wall);
    steps.push(nSteps);
  }
  const rates = [];
  for (let i = 1; i < dxs.length; i++) rates.push(Math.log(errs[i - 1] / errs[i]) / Math.log(dxs[i - 1] / dxs[i]));
  return { Ns: NList.slice(), dxs, errs, rates, runtimes, steps };
}

function runConv2D({ p, NList, scheme, T, CFL, A, He, g, L }) {
  const basis = makeBasis(p);
  const phys = makeLinPhys2D({ g, He });
  const sw = standingWave2D({ L, A, g, He });
  const dxs = [], errs = [], runtimes = [], steps = [];
  for (const N of NList) {
    const mesh = new Mesh2D(L, N, N, basis);
    const q = projectField2D(mesh, sw.exact, 0.0);
    const stepper = scheme === "rk" ? makeRKDG2D(mesh, phys) : makeADERDG2D(mesh, phys);
    const order = scheme === "rk" ? rkOrder(p) : (p + 1);
    const t0 = performance.now();
    let t = 0, nSteps = 0;
    while (t < T - 1e-15) {
      let dt = dtFor(p, mesh.dx, phys.c0, CFL, scheme, order);
      if (t + dt > T) dt = T - t;
      stepper.step(q, dt, t);
      t += dt;
      nSteps++;
    }
    const wall = performance.now() - t0;
    dxs.push(mesh.dx);
    errs.push(l2Error2D(mesh, q, sw.exact, T));
    runtimes.push(wall);
    steps.push(nSteps);
  }
  const rates = [];
  for (let i = 1; i < dxs.length; i++) rates.push(Math.log(errs[i - 1] / errs[i]) / Math.log(dxs[i - 1] / dxs[i]));
  return { Ns: NList.slice(), dxs, errs, rates, runtimes, steps };
}

// =====================================================================
// Reporting helpers
// =====================================================================

function pad(s, n) { s = String(s); while (s.length < n) s = " " + s; return s; }
function sci(x, d = 2) { return Number(x).toExponential(d); }

function txtTable(label, res) {
  console.log(`\n--- ${label} ---`);
  console.log("  N      dx          err        rate     steps   wall(ms)");
  for (let i = 0; i < res.Ns.length; i++) {
    const r = i === 0 ? "  -   " : pad(res.rates[i - 1].toFixed(3), 6);
    console.log(`  ${pad(res.Ns[i], 4)}   ${sci(res.dxs[i], 2)}   ${sci(res.errs[i], 2)}   ${r}   ${pad(res.steps[i], 5)}   ${pad(res.runtimes[i].toFixed(1), 7)}`);
  }
}

function texCell(x, d = 1) {
  if (!Number.isFinite(x)) return "—";
  const s = Number(x).toExponential(d);
  const m = s.match(/^(-?\d+(?:\.\d+)?)e([+-]?\d+)$/);
  if (!m) return s;
  let mant = m[1];
  let exp = parseInt(m[2], 10);
  return `$${mant}\\!\\times\\!10^{${exp}}$`;
}

function texTable1D(scheme, results) {
  const lines = [];
  lines.push("\\begin{tabular}{ccccccc}");
  lines.push("\\toprule");
  lines.push("Method order & $N=32$ & $N=64$ & $N=128$ & $N=256$ & rate (last) \\\\");
  lines.push("\\midrule");
  for (const { order, errs, rates } of results) {
    const cells = errs.map(e => texCell(e));
    const r = rates.length ? rates[rates.length - 1].toFixed(2) : "—";
    lines.push(`${order} & ${cells.join(" & ")} & ${r} \\\\`);
  }
  lines.push("\\bottomrule");
  lines.push("\\end{tabular}");
  return lines.join("\n");
}

// =====================================================================
// Main
// =====================================================================

(function main() {
  const params = { L: 1.0, A: 0.1, g: 1.0, He: 1.0, CFL: 0.25 };
  const T = 0.5;       // final time
  const NList1 = [32, 64, 128, 256];
  const NList2_lo = [16, 32, 64, 128];   // for method orders 2 and 3
  const NList2_hi = [8, 16, 32, 64];     // for method orders 4 and 5 (avoid long 2D runs)

  const psExt = [1, 2, 3, 4];

  console.log("=== Linearised SWE convergence (CPU/Float64 reference) ===");
  console.log(`Parameters: L=${params.L}, A=${params.A}, g=${params.g}, H_e=${params.He}, T=${T}, CFL=${params.CFL}`);
  console.log(`(c0 = sqrt(g H_e) = ${Math.sqrt(params.g * params.He)})`);

  const results1d = { rk: [], ader: [] };
  const results2d = { rk: [], ader: [] };

  for (const p of psExt) {
    const order = p + 1;
    const res_rk = runConv1D({ p, NList: NList1, scheme: "rk", T, ...params });
    const res_ad = runConv1D({ p, NList: NList1, scheme: "ader", T, ...params });
    txtTable(`1D method order ${order}  RK-DG`, res_rk);
    txtTable(`1D method order ${order}  ADER-DG`, res_ad);
    results1d.rk.push({ order, ...res_rk });
    results1d.ader.push({ order, ...res_ad });
  }
  for (const p of psExt) {
    const order = p + 1;
    const NList2 = p <= 2 ? NList2_lo : NList2_hi;
    const res_rk = runConv2D({ p, NList: NList2, scheme: "rk", T, ...params });
    const res_ad = runConv2D({ p, NList: NList2, scheme: "ader", T, ...params });
    txtTable(`2D method order ${order}  RK-DG`, res_rk);
    txtTable(`2D method order ${order}  ADER-DG`, res_ad);
    results2d.rk.push({ order, NList: NList2, ...res_rk });
    results2d.ader.push({ order, NList: NList2, ...res_ad });
  }

  console.log("\n\n=========== LaTeX BLOCKS ===========\n");
  console.log("% ---- 1D ADER-DG ----");
  console.log(texTable1D("ADER-DG", results1d.ader));
  console.log("\n% ---- 1D RK-DG ----");
  console.log(texTable1D("RK-DG", results1d.rk));

  // 2D table — N grid varies per row to keep run time tractable for orders 4–5.
  function texTable2D(scheme, results) {
    const lines = [];
    lines.push("\\begin{tabular}{ccccccc}");
    lines.push("\\toprule");
    lines.push("Method order & $N$ values & error sequence & rate (last) \\\\");
    lines.push("\\midrule");
    for (const { order, NList, errs, rates } of results) {
      const N_str = NList.join(",\\,");
      const errCells = errs.map(e => texCell(e)).join(", ");
      const r = rates.length ? rates[rates.length - 1].toFixed(2) : "—";
      lines.push(`${order} & ${N_str} & ${errCells} & ${r} \\\\`);
    }
    lines.push("\\bottomrule");
    lines.push("\\end{tabular}");
    return lines.join("\n");
  }
  console.log("\n% ---- 2D ADER-DG ----");
  console.log(texTable2D("ADER-DG", results2d.ader));
  console.log("\n% ---- 2D RK-DG ----");
  console.log(texTable2D("RK-DG", results2d.rk));
})();
