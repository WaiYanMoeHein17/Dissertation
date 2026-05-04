// aderdg.js — ADER-DG with local space-time predictor (Dumbser-style).
//
// Algorithm (per time step, per cell, independent across cells):
//
//   PREDICTOR: solve a local space-time Galerkin problem on the cell × [0, dt]
//   using LGL nodes in time (n_t = p + 1 nodes including τ = 0 and τ = 1).
//   With tau_l in [0, 1] and Lagrange basis ψ_l, the discrete equation is
//
//       sum_l Dτ[m, l] Q[l, ix, k] = dt * R(Q)[m, ix, k]      for m = 1..p
//       Q[0, ix, k] = q^n[ix, k]                              (initial condition)
//
//   where Dτ is the time differentiation matrix on [0, 1] (= 2 × D on [-1, 1])
//   and R(Q) is the spatial residual -(2/dx) ∂_xi F(Q) + S(x, t_l).
//   Iterating Picard sweeps on Q yields a predictor of order p + 1 in time.
//
//   CORRECTOR: a single explicit DG update using TIME-AVERAGED quantities
//
//       q^{n+1} = q^n + dt * [ -(2/dx)(D F̄)_ix + boundary( F̄^*, F̄^trace ) + S̄ ]
//
//   where bars are time averages computed with the LGL weights wτ.
//
// The corrector is (p+1)-th order accurate in time when the predictor has
// converged; with p Picard sweeps + 1 fixed-point sweep, this is achieved
// for all p we report.

import { fluxX2D, fluxY2D, flux1D, rusanov1D, rusanov2D } from "./swe.js";
import { lglNodesWeights } from "./basis.js";
import { dgRhs1D, allocWork1D } from "./dg1d.js";
import { dgRhs2D, allocWork2D } from "./dg2d.js";
import { buildSourceField1D, buildSourceField2D } from "./mms.js";

// ----- Small linear algebra: invert a small (p x p) dense matrix in place.
// Gauss-Jordan elimination with partial pivoting. p is at most ~6 for our use.
function invertInPlace(A, n) {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let col = 0; col < n; col++) {
    // pivot
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
    if (piv === 0) throw new Error("singular reduced time-derivative matrix");
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

// LGL nodes/weights/Diff matrix on [0, 1] (mapping τ = (xi + 1)/2 from [-1,1]).
function lglOn01(p) {
  const { x: xi, w: wxi } = lglNodesWeights(p);
  const n = p + 1;
  const tau = new Float64Array(n);
  const wt  = new Float64Array(n);
  for (let i = 0; i < n; i++) { tau[i] = 0.5 * (xi[i] + 1); wt[i] = 0.5 * wxi[i]; }
  return { tau, wt };
}

// Build time differentiation matrix Dτ (n_t x n_t) on [0,1] from the basis.D matrix on [-1,1].
function buildTimeDiffMatrix(basis) {
  const n = basis.n;
  const Dt = new Float64Array(n * n);
  // D^τ = 2 * D_xi   (since dτ/dξ = 1/2, so d/dτ = 2 d/dξ)
  for (let i = 0; i < n * n; i++) Dt[i] = 2 * basis.D[i];
  return Dt;
}

// Reduced (p x p) submatrix Dτ[1..n-1, 1..n-1] and column Dτ[1..n-1, 0].
function buildReduced(Dtau, n) {
  const p = n - 1;
  const Dred = new Float64Array(p * p);
  const D0   = new Float64Array(p);
  for (let m = 1; m < n; m++) {
    D0[m - 1] = Dtau[m * n + 0];
    for (let l = 1; l < n; l++) {
      Dred[(m - 1) * p + (l - 1)] = Dtau[m * n + l];
    }
  }
  return { Dred, D0 };
}

// =============================================================
// 1D ADER-DG
// =============================================================
export function makeADERDG1D(mesh, g, mmsSource = null, picardIters = -1) {
  const { Nx, n, M, dx, basis } = mesh;
  const T = n;                       // n_t = p + 1 time nodes
  const p = basis.p;
  const { tau, wt } = lglOn01(p);
  const Dtau = buildTimeDiffMatrix(basis);
  const { Dred, D0 } = buildReduced(Dtau, T);
  const Dred_inv = invertInPlace(Dred.slice(), T - 1);
  const nIter = picardIters < 0 ? p + 1 : picardIters;

  // Storage
  const Q     = new Float64Array(Nx * T * n * M);     // predictor at (cell, time, space)
  const Fpred = new Float64Array(Nx * T * n * M);     // F(Q) at all (cell, time, space)
  const R     = new Float64Array(Nx * T * n * M);     // residual at (cell, time>0, space) only m>=1 used
  const RHSm  = new Float64Array(Nx * (T - 1) * n * M);
  const FbarV = new Float64Array(Nx * n * M);         // time-averaged volume flux F̄
  const qBarL = new Float64Array(Nx * M);             // time-avg trace at xi=-1
  const qBarR = new Float64Array(Nx * M);             // time-avg trace at xi=+1
  const FbarL_int = new Float64Array(Nx * M);         // time-avg F at left  trace = sum w^τ F(Q[ix=0, l])
  const FbarR_int = new Float64Array(Nx * M);
  const FbarStar  = new Float64Array(Nx * M);         // time-avg Rusanov flux at each face
  const Sbar  = new Float64Array(Nx * n * M);         // time-avg source

  // Pre-source per time node, allocated once.
  const Stime = mmsSource ? new Float64Array(Nx * T * n * M) : null;

  function step(qIn, dt, tNow) {
    // ---- 1) Build source at every (l, x) ----
    if (mmsSource) {
      for (let l = 0; l < T; l++) {
        const tl = tNow + dt * tau[l];
        const slice = buildSourceField1D(mesh, mmsSource, tl);
        Stime.set(slice, l * Nx * n * M);    // store as [time-major, cell, space, M]
      }
    }

    // ---- 2) Initialise predictor: Q[i, l, ix, :] = q^n[i, ix, :] for all l ----
    for (let i = 0; i < Nx; i++) {
      for (let l = 0; l < T; l++) {
        for (let ix = 0; ix < n; ix++) {
          const dst = ((i * T + l) * n + ix) * M;
          const src = (i * n + ix) * M;
          for (let k = 0; k < M; k++) Q[dst + k] = qIn[src + k];
        }
      }
    }

    // ---- 3) Picard iteration ----
    const inv2dx = 2.0 / dx;
    for (let iter = 0; iter < nIter; iter++) {
      // 3a) F(Q) at all nodes
      for (let off = 0; off < Nx * T * n * M; off += M) flux1D(Q, off, Fpred, off, g);

      // 3b) Compute R[i, l, ix, :] = -(2/dx) (D F[i, l, :, :])[ix, :] + S[i, l, ix, :]   for l = 1..T-1
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
            if (mmsSource) {
              const sOff = ((l * Nx + i) * n + ix) * M;
              R[rOff]     += Stime[sOff];
              R[rOff + 1] += Stime[sOff + 1];
            }
          }
        }
      }

      // 3c) Form RHSm[i, m-1, ix, :] = dt * R[i, m, ix, :] - Dτ[m, 0] * q^n[i, ix, :]
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

      // 3d) Solve Q[i, 1..T-1, ix, :] = Dred_inv @ RHSm[i, :, ix, :]
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

    // ---- 4) Time-average ----
    // F(Q) was last updated for the latest Q; recompute to be safe (cheap).
    for (let off = 0; off < Nx * T * n * M; off += M) flux1D(Q, off, Fpred, off, g);

    // F̄_vol
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

    // q̄ at trace, F̄ at trace (using F(Q) at endpoint nodes)
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

    // F̄^* at every face k — average of pointwise-in-time Rusanov fluxes
    // (more accurate than Rusanov(q̄_L, q̄_R) for nonlinear problems).
    const tmpL = new Float64Array(M), tmpR = new Float64Array(M), tmpF = new Float64Array(M);
    for (let k = 0; k < Nx; k++) {
      const kLeft = (k - 1 + Nx) % Nx;
      let s0 = 0, s1 = 0;
      for (let l = 0; l < T; l++) {
        const lLOff = ((kLeft * T + l) * n + (n - 1)) * M;
        const lROff = ((k     * T + l) * n + 0)       * M;
        rusanov1D(Q, lLOff, Q, lROff, tmpF, 0, g);
        s0 += wt[l] * tmpF[0];
        s1 += wt[l] * tmpF[1];
      }
      FbarStar[k * M]     = s0;
      FbarStar[k * M + 1] = s1;
    }

    // S̄
    if (mmsSource) {
      Sbar.fill(0);
      for (let i = 0; i < Nx; i++) {
        for (let ix = 0; ix < n; ix++) {
          let s0 = 0, s1 = 0;
          for (let l = 0; l < T; l++) {
            const sOff = ((l * Nx + i) * n + ix) * M;
            s0 += wt[l] * Stime[sOff];
            s1 += wt[l] * Stime[sOff + 1];
          }
          const off = (i * n + ix) * M;
          Sbar[off]     = s0;
          Sbar[off + 1] = s1;
        }
      }
    } else {
      Sbar.fill(0);
    }

    // ---- 5) Corrector update ----
    // q^{n+1}_i = q^n_i + dt * [ -(2/dx)(D F̄_vol)_ix + boundary corrections + S̄_ix ]
    const w = basis.w;
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
        let r0 = -inv2dx * s0 + Sbar[off];
        let r1 = -inv2dx * s1 + Sbar[off + 1];
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

    return tNow + dt;
  }
  return { step, name: `ADER-DG (P${p}, ${nIter} Picard sweeps)`, mesh };
}

// =============================================================
// 2D ADER-DG
// =============================================================
export function makeADERDG2D(mesh, g, mmsSource = null, picardIters = -1) {
  const { Nx, Ny, n, M, dx, dy, basis } = mesh;
  const T = n;
  const p = basis.p;
  const { tau, wt } = lglOn01(p);
  const Dtau = buildTimeDiffMatrix(basis);
  const { Dred, D0 } = buildReduced(Dtau, T);
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
  const qBarW = new Float64Array(Ne * n * M);
  const qBarE = new Float64Array(Ne * n * M);
  const qBarS = new Float64Array(Ne * n * M);
  const qBarN = new Float64Array(Ne * n * M);
  const FxBarW = new Float64Array(Ne * n * M);
  const FxBarE = new Float64Array(Ne * n * M);
  const FyBarS = new Float64Array(Ne * n * M);
  const FyBarN = new Float64Array(Ne * n * M);
  const FstarX = new Float64Array(Ne * n * M);
  const FstarY = new Float64Array(Ne * n * M);
  const Sbar  = new Float64Array(Ne * n * n * M);
  const Stime = mmsSource ? new Float64Array(T * Ne * n * n * M) : null;
  const tmpF  = new Float64Array(M);

  function step(qIn, dt, tNow) {
    // 1) Source at each time level
    if (mmsSource) {
      for (let l = 0; l < T; l++) {
        const tl = tNow + dt * tau[l];
        const slice = buildSourceField2D(mesh, mmsSource, tl);
        Stime.set(slice, l * Ne * n * n * M);
      }
    }

    // 2) Init predictor
    for (let e = 0; e < Ne; e++) {
      for (let l = 0; l < T; l++) {
        const dstE = ((e * T + l) * n * n) * M;
        const srcE = e * n * n * M;
        Q.set(qIn.subarray(srcE, srcE + n * n * M), dstE);
      }
    }

    // 3) Picard iterations
    for (let iter = 0; iter < nIter; iter++) {
      // F(Q)
      for (let off = 0; off < Ne * T * n * n * M; off += M) {
        fluxX2D(Q, off, Fxv, off, g);
        fluxY2D(Q, off, Fyv, off, g);
      }
      // R[e, l, jy, jx, :]
      for (let e = 0; e < Ne; e++) {
        for (let l = 1; l < T; l++) {
          const eb = ((e * T + l) * n * n) * M;
          for (let jy = 0; jy < n; jy++) {
            for (let jx = 0; jx < n; jx++) {
              let s0 = 0, s1 = 0, s2 = 0;
              // d/dxi (Fx)
              for (let kx = 0; kx < n; kx++) {
                const Dij = basis.D[jx * n + kx];
                const fOff = eb + (jy * n + kx) * M;
                s0 += Dij * Fxv[fOff];
                s1 += Dij * Fxv[fOff + 1];
                s2 += Dij * Fxv[fOff + 2];
              }
              s0 *= -inv2dx; s1 *= -inv2dx; s2 *= -inv2dx;
              // d/deta (Fy)
              let t0 = 0, t1 = 0, t2 = 0;
              for (let ky = 0; ky < n; ky++) {
                const Dij = basis.D[jy * n + ky];
                const fOff = eb + (ky * n + jx) * M;
                t0 += Dij * Fyv[fOff];
                t1 += Dij * Fyv[fOff + 1];
                t2 += Dij * Fyv[fOff + 2];
              }
              s0 -= inv2dy * t0; s1 -= inv2dy * t1; s2 -= inv2dy * t2;
              const rOff = eb + (jy * n + jx) * M;
              R[rOff]     = s0;
              R[rOff + 1] = s1;
              R[rOff + 2] = s2;
              if (mmsSource) {
                const sOff = ((l * Ne + e) * n * n + jy * n + jx) * M;
                R[rOff]     += Stime[sOff];
                R[rOff + 1] += Stime[sOff + 1];
                R[rOff + 2] += Stime[sOff + 2];
              }
            }
          }
        }
      }
      // RHSm
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
      // Solve Q[e, 1..T-1, jy, jx, :] = Dred_inv @ RHSm
      for (let e = 0; e < Ne; e++) {
        for (let p2 = 0; p2 < n * n; p2++) {
          for (let mm = 0; mm < T - 1; mm++) {
            let s0 = 0, s1 = 0, s2 = 0;
            for (let l = 0; l < T - 1; l++) {
              const Dij = Dred_inv[mm * (T - 1) + l];
              const rhsOff = ((e * (T - 1) + l) * n * n + p2) * M;
              s0 += Dij * RHSm[rhsOff];
              s1 += Dij * RHSm[rhsOff + 1];
              s2 += Dij * RHSm[rhsOff + 2];
            }
            const qOff = ((e * T + (mm + 1)) * n * n + p2) * M;
            Q[qOff]     = s0;
            Q[qOff + 1] = s1;
            Q[qOff + 2] = s2;
          }
        }
      }
    }

    // 4) Time averages
    for (let off = 0; off < Ne * T * n * n * M; off += M) {
      fluxX2D(Q, off, Fxv, off, g);
      fluxY2D(Q, off, Fyv, off, g);
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

    // Trace averages on the four sides of every cell
    qBarW.fill(0); qBarE.fill(0); qBarS.fill(0); qBarN.fill(0);
    FxBarW.fill(0); FxBarE.fill(0); FyBarS.fill(0); FyBarN.fill(0);
    for (let e = 0; e < Ne; e++) {
      for (let j = 0; j < n; j++) {
        // West: jx=0, jy=j ; East: jx=n-1; South: jy=0, jx=j; North: jy=n-1
        let qW0 = 0, qW1 = 0, qW2 = 0, qE0 = 0, qE1 = 0, qE2 = 0;
        let qS0 = 0, qS1 = 0, qS2 = 0, qN0 = 0, qN1 = 0, qN2 = 0;
        let FW0 = 0, FW1 = 0, FW2 = 0, FE0 = 0, FE1 = 0, FE2 = 0;
        let FS0 = 0, FS1 = 0, FS2 = 0, FN0 = 0, FN1 = 0, FN2 = 0;
        for (let l = 0; l < T; l++) {
          const wl = wt[l];
          const eb = ((e * T + l) * n * n) * M;
          const oW = eb + (j * n + 0) * M;
          const oE = eb + (j * n + (n - 1)) * M;
          const oS = eb + (0 * n + j) * M;
          const oN = eb + ((n - 1) * n + j) * M;
          qW0 += wl * Q[oW]; qW1 += wl * Q[oW + 1]; qW2 += wl * Q[oW + 2];
          qE0 += wl * Q[oE]; qE1 += wl * Q[oE + 1]; qE2 += wl * Q[oE + 2];
          qS0 += wl * Q[oS]; qS1 += wl * Q[oS + 1]; qS2 += wl * Q[oS + 2];
          qN0 += wl * Q[oN]; qN1 += wl * Q[oN + 1]; qN2 += wl * Q[oN + 2];
          FW0 += wl * Fxv[oW]; FW1 += wl * Fxv[oW + 1]; FW2 += wl * Fxv[oW + 2];
          FE0 += wl * Fxv[oE]; FE1 += wl * Fxv[oE + 1]; FE2 += wl * Fxv[oE + 2];
          FS0 += wl * Fyv[oS]; FS1 += wl * Fyv[oS + 1]; FS2 += wl * Fyv[oS + 2];
          FN0 += wl * Fyv[oN]; FN1 += wl * Fyv[oN + 1]; FN2 += wl * Fyv[oN + 2];
        }
        const off = (e * n + j) * M;
        qBarW[off] = qW0; qBarW[off + 1] = qW1; qBarW[off + 2] = qW2;
        qBarE[off] = qE0; qBarE[off + 1] = qE1; qBarE[off + 2] = qE2;
        qBarS[off] = qS0; qBarS[off + 1] = qS1; qBarS[off + 2] = qS2;
        qBarN[off] = qN0; qBarN[off + 1] = qN1; qBarN[off + 2] = qN2;
        FxBarW[off] = FW0; FxBarW[off + 1] = FW1; FxBarW[off + 2] = FW2;
        FxBarE[off] = FE0; FxBarE[off + 1] = FE1; FxBarE[off + 2] = FE2;
        FyBarS[off] = FS0; FyBarS[off + 1] = FS1; FyBarS[off + 2] = FS2;
        FyBarN[off] = FN0; FyBarN[off + 1] = FN1; FyBarN[off + 2] = FN2;
      }
    }

    // F̄^* at faces (time-average of pointwise Rusanov flux).
    // E-W faces: face between cell (ix-1, iy) and (ix, iy)
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
            rusanov2D(Q, oL, Q, oR, tmpF, 0, 1, 0, g);
            s0 += wl * tmpF[0]; s1 += wl * tmpF[1]; s2 += wl * tmpF[2];
          }
          const off = (eR * n + j) * M;
          FstarX[off] = s0; FstarX[off + 1] = s1; FstarX[off + 2] = s2;
        }
      }
    }
    // N-S faces
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
            rusanov2D(Q, oB, Q, oTt, tmpF, 0, 0, 1, g);
            s0 += wl * tmpF[0]; s1 += wl * tmpF[1]; s2 += wl * tmpF[2];
          }
          const off = (eT * n + j) * M;
          FstarY[off] = s0; FstarY[off + 1] = s1; FstarY[off + 2] = s2;
        }
      }
    }

    // S̄
    Sbar.fill(0);
    if (mmsSource) {
      for (let e = 0; e < Ne; e++) {
        for (let p2 = 0; p2 < n * n; p2++) {
          let s0 = 0, s1 = 0, s2 = 0;
          for (let l = 0; l < T; l++) {
            const off = ((l * Ne + e) * n * n + p2) * M;
            const wl = wt[l];
            s0 += wl * Stime[off]; s1 += wl * Stime[off + 1]; s2 += wl * Stime[off + 2];
          }
          const off = (e * n * n + p2) * M;
          Sbar[off] = s0; Sbar[off + 1] = s1; Sbar[off + 2] = s2;
        }
      }
    }

    // 5) Corrector update
    for (let iy = 0; iy < Ny; iy++) {
      for (let ix = 0; ix < Nx; ix++) {
        const e = iy * Nx + ix;
        const eb = e * n * n * M;
        const ixR = (ix + 1) % Nx;
        const iyT = (iy + 1) % Ny;
        // Volume contribution
        for (let jy = 0; jy < n; jy++) {
          for (let jx = 0; jx < n; jx++) {
            let sx0 = 0, sx1 = 0, sx2 = 0, sy0 = 0, sy1 = 0, sy2 = 0;
            for (let kx = 0; kx < n; kx++) {
              const Dij = basis.D[jx * n + kx];
              const fOff = eb + (jy * n + kx) * M;
              sx0 += Dij * FbarVx[fOff];
              sx1 += Dij * FbarVx[fOff + 1];
              sx2 += Dij * FbarVx[fOff + 2];
            }
            for (let ky = 0; ky < n; ky++) {
              const Dij = basis.D[jy * n + ky];
              const fOff = eb + (ky * n + jx) * M;
              sy0 += Dij * FbarVy[fOff];
              sy1 += Dij * FbarVy[fOff + 1];
              sy2 += Dij * FbarVy[fOff + 2];
            }
            const off = eb + (jy * n + jx) * M;
            const sOff = (e * n * n + jy * n + jx) * M;
            qIn[off]     += dt * (-inv2dx * sx0 - inv2dy * sy0 + Sbar[sOff]);
            qIn[off + 1] += dt * (-inv2dx * sx1 - inv2dy * sy1 + Sbar[sOff + 1]);
            qIn[off + 2] += dt * (-inv2dx * sx2 - inv2dy * sy2 + Sbar[sOff + 2]);
          }
        }
        // Boundary corrections (faces of this cell)
        const eR = iy  * Nx + ixR;
        const eN = iyT * Nx + ix;
        for (let j = 0; j < n; j++) {
          const offW = (e  * n + j) * M;     // west face state of this cell == stored index
          const offE = (eR * n + j) * M;     // west face state of right neighbour == east face of this cell (FstarX index)
          const offS = (e  * n + j) * M;
          const offN = (eN * n + j) * M;
          // West edge: jx=0
          const wNode = eb + (j * n + 0) * M;
          qIn[wNode]     += dt * inv2dx * (FstarX[offW]     - FxBarW[offW])     / w[0];
          qIn[wNode + 1] += dt * inv2dx * (FstarX[offW + 1] - FxBarW[offW + 1]) / w[0];
          qIn[wNode + 2] += dt * inv2dx * (FstarX[offW + 2] - FxBarW[offW + 2]) / w[0];
          // East edge: jx=n-1
          const eNode = eb + (j * n + (n - 1)) * M;
          qIn[eNode]     += dt * inv2dx * (FxBarE[offW]     - FstarX[offE])     / w[n - 1];
          qIn[eNode + 1] += dt * inv2dx * (FxBarE[offW + 1] - FstarX[offE + 1]) / w[n - 1];
          qIn[eNode + 2] += dt * inv2dx * (FxBarE[offW + 2] - FstarX[offE + 2]) / w[n - 1];
          // South edge: jy=0
          const sNode = eb + (0 * n + j) * M;
          qIn[sNode]     += dt * inv2dy * (FstarY[offS]     - FyBarS[offS])     / w[0];
          qIn[sNode + 1] += dt * inv2dy * (FstarY[offS + 1] - FyBarS[offS + 1]) / w[0];
          qIn[sNode + 2] += dt * inv2dy * (FstarY[offS + 2] - FyBarS[offS + 2]) / w[0];
          // North edge: jy=n-1
          const nNode = eb + ((n - 1) * n + j) * M;
          qIn[nNode]     += dt * inv2dy * (FyBarN[offS]     - FstarY[offN])     / w[n - 1];
          qIn[nNode + 1] += dt * inv2dy * (FyBarN[offS + 1] - FstarY[offN + 1]) / w[n - 1];
          qIn[nNode + 2] += dt * inv2dy * (FyBarN[offS + 2] - FstarY[offN + 2]) / w[n - 1];
        }
      }
    }

    return tNow + dt;
  }
  return { step, name: `ADER-DG (P${p}, ${nIter} Picard sweeps)`, mesh };
}
