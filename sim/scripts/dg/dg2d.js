// dg2d.js — 2D DG spatial operator for SWE on periodic Cartesian quad mesh.
//
// Storage layout for the state vector q (tensor-product nodal LGL):
//   q is a flat Float64Array of length Nx * Ny * n * n * M
//   Indexing: ((((iy * Nx) + ix) * n + jy) * n + jx) * M + k
//             where (ix, iy) = element index, (jx, jy) = node index in element,
//             k = variable index (M = 3 for SWE 2D: h, hu, hv)
//
// All face-trace arrays have shape (Ny, Nx, n, M) — i.e., per-face-line of nodes
// per element along the corresponding face.

import { fluxX2D, fluxY2D, rusanov2D } from "./swe.js";

export class Mesh2D {
  constructor(Lx, Ly, Nx, Ny, basis) {
    this.Lx = Lx; this.Ly = Ly; this.Nx = Nx; this.Ny = Ny;
    this.dx = Lx / Nx; this.dy = Ly / Ny;
    this.basis = basis; this.n = basis.n; this.M = 3;
    const n = basis.n;
    // Node coordinates per element, flat (Ny*Nx*n*n)
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

export function allocWork2D(mesh) {
  const { Nx, Ny, n, M } = mesh;
  return {
    Fxv : new Float64Array(Nx * Ny * n * n * M),
    Fyv : new Float64Array(Nx * Ny * n * n * M),
    qW  : new Float64Array(Nx * Ny * n * M),    // west face traces  (xi=-1)
    qE  : new Float64Array(Nx * Ny * n * M),    // east face traces  (xi=+1)
    qS  : new Float64Array(Nx * Ny * n * M),    // south face traces (eta=-1)
    qN  : new Float64Array(Nx * Ny * n * M),    // north face traces (eta=+1)
    FstarX : new Float64Array(Nx * Ny * n * M), // numerical flux at vertical faces
    FstarY : new Float64Array(Nx * Ny * n * M), // numerical flux at horizontal faces
    FxW : new Float64Array(Nx * Ny * n * M),
    FxE : new Float64Array(Nx * Ny * n * M),
    FyS : new Float64Array(Nx * Ny * n * M),
    FyN : new Float64Array(Nx * Ny * n * M),
  };
}

/**
 * 2D DG semi-discrete RHS.  Mirrors dgRhs1D in structure.
 *
 * Optional face traces qW/qE/qS/qN override the instantaneous nodal endpoints;
 * pass them from the ADER-DG predictor.  Each is shape (Ny, Nx, n, M).
 */
export function dgRhs2D(rhs, q, mesh, work, g,
                        qW_in, qE_in, qS_in, qN_in, source) {
  const { Nx, Ny, n, M, dx, dy, basis } = mesh;
  const D = basis.D, w = basis.w;
  const { Fxv, Fyv, qW, qE, qS, qN, FstarX, FstarY, FxW, FxE, FyS, FyN } = work;

  // ---- Volume fluxes Fx, Fy at every node ----
  for (let off = 0; off < Nx * Ny * n * n * M; off += M) {
    fluxX2D(q, off, Fxv, off, g);
    fluxY2D(q, off, Fyv, off, g);
  }

  // ---- -(2/dx) d/dxi Fx  -  (2/dy) d/deta Fy ----
  const invX = 2.0 / dx, invY = 2.0 / dy;
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = ((iy * Nx + ix) * n * n) * M;
      // d/dxi: derivative along inner index jx
      for (let jy = 0; jy < n; jy++) {
        const rowBase = eb + jy * n * M;
        for (let jx = 0; jx < n; jx++) {
          let s0 = 0, s1 = 0, s2 = 0;
          // sum over kx: D[jx,kx] * Fx[kx]
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
      // d/deta: derivative along outer index jy
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

  // ---- Face traces ----
  // qW[(iy*Nx+ix)*n + jy] : value at (xi=-1, eta=basis.x[jy]) of cell (ix,iy)
  // qE: xi=+1; qS: eta=-1; qN: eta=+1
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = mesh.elemBase(ix, iy);
      for (let j = 0; j < n; j++) {
        const off = ((iy * Nx + ix) * n + j) * M;
        // West:  jx=0,    jy=j     -> node index j*n + 0
        const wOff = eb + (j * n + 0) * M;
        const eOff = eb + (j * n + (n - 1)) * M;
        // South: jy=0,    jx=j
        const sOff = eb + (0 * n + j) * M;
        const nOff = eb + ((n - 1) * n + j) * M;
        if (qW_in == null) { qW[off]   = q[wOff]; qW[off+1] = q[wOff+1]; qW[off+2] = q[wOff+2]; }
        else                { qW[off]   = qW_in[off]; qW[off+1] = qW_in[off+1]; qW[off+2] = qW_in[off+2]; }
        if (qE_in == null) { qE[off]   = q[eOff]; qE[off+1] = q[eOff+1]; qE[off+2] = q[eOff+2]; }
        else                { qE[off]   = qE_in[off]; qE[off+1] = qE_in[off+1]; qE[off+2] = qE_in[off+2]; }
        if (qS_in == null) { qS[off]   = q[sOff]; qS[off+1] = q[sOff+1]; qS[off+2] = q[sOff+2]; }
        else                { qS[off]   = qS_in[off]; qS[off+1] = qS_in[off+1]; qS[off+2] = qS_in[off+2]; }
        if (qN_in == null) { qN[off]   = q[nOff]; qN[off+1] = q[nOff+1]; qN[off+2] = q[nOff+2]; }
        else                { qN[off]   = qN_in[off]; qN[off+1] = qN_in[off+1]; qN[off+2] = qN_in[off+2]; }
        // F at trace
        fluxX2D(qW, off, FxW, off, g);
        fluxX2D(qE, off, FxE, off, g);
        fluxY2D(qS, off, FyS, off, g);
        fluxY2D(qN, off, FyN, off, g);
      }
    }
  }

  // ---- East-West Riemann problems (face k between cell (ix-1, iy) and (ix, iy)) ----
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const ixL = (ix - 1 + Nx) % Nx;
      for (let j = 0; j < n; j++) {
        const offL = ((iy * Nx + ixL) * n + j) * M;   // east trace of left neighbour
        const offR = ((iy * Nx + ix)  * n + j) * M;   // west trace of cell
        rusanov2D(qE, offL, qW, offR, FstarX, offR, 1, 0, g);
      }
    }
  }
  // ---- North-South Riemann problems (face between (ix, iy-1) and (ix, iy)) ----
  for (let iy = 0; iy < Ny; iy++) {
    const iyB = (iy - 1 + Ny) % Ny;
    for (let ix = 0; ix < Nx; ix++) {
      for (let j = 0; j < n; j++) {
        const offB = ((iyB * Nx + ix) * n + j) * M;   // north trace of bottom neighbour
        const offT = ((iy  * Nx + ix) * n + j) * M;   // south trace of cell
        rusanov2D(qN, offB, qS, offT, FstarY, offT, 0, 1, g);
      }
    }
  }

  // ---- Strong-form corrections on the four sides of every cell ----
  const w0 = w[0], wn = w[n - 1];
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      const eb = mesh.elemBase(ix, iy);
      const ixR = (ix + 1) % Nx;
      const iyT = (iy + 1) % Ny;
      for (let j = 0; j < n; j++) {
        const offW = ((iy * Nx + ix) * n + j) * M;     // west of (ix,iy)
        const offE = ((iy * Nx + ixR) * n + j) * M;    // west of right neighbour = east of (ix,iy)
        const offS = ((iy * Nx + ix) * n + j) * M;     // south of (ix,iy)
        const offN = ((iyT * Nx + ix) * n + j) * M;    // south of top neighbour = north of (ix,iy)

        // West edge: jx=0, jy=j  ; rhs += (2/dx)/w0 * (F* - F_internal)
        const wNode = eb + (j * n + 0) * M;
        rhs[wNode]     += invX * (FstarX[offW]     - FxW[offW])     / w0;
        rhs[wNode + 1] += invX * (FstarX[offW + 1] - FxW[offW + 1]) / w0;
        rhs[wNode + 2] += invX * (FstarX[offW + 2] - FxW[offW + 2]) / w0;
        // East edge: jx=n-1, jy=j  ; rhs += (2/dx)/wn * (F_internal - F*)
        const eNode = eb + (j * n + (n - 1)) * M;
        rhs[eNode]     += invX * (FxE[offW]     - FstarX[offE])     / wn;
        rhs[eNode + 1] += invX * (FxE[offW + 1] - FstarX[offE + 1]) / wn;
        rhs[eNode + 2] += invX * (FxE[offW + 2] - FstarX[offE + 2]) / wn;
        // South edge: jy=0, jx=j ; rhs += (2/dy)/w0 * (F* - F_internal)
        const sNode = eb + (0 * n + j) * M;
        rhs[sNode]     += invY * (FstarY[offS]     - FyS[offS])     / w0;
        rhs[sNode + 1] += invY * (FstarY[offS + 1] - FyS[offS + 1]) / w0;
        rhs[sNode + 2] += invY * (FstarY[offS + 2] - FyS[offS + 2]) / w0;
        // North edge: jy=n-1, jx=j ; rhs += (2/dy)/wn * (F_internal - F*)
        const nNode = eb + ((n - 1) * n + j) * M;
        rhs[nNode]     += invY * (FyN[offS]     - FstarY[offN])     / wn;
        rhs[nNode + 1] += invY * (FyN[offS + 1] - FstarY[offN + 1]) / wn;
        rhs[nNode + 2] += invY * (FyN[offS + 2] - FstarY[offN + 2]) / wn;
      }
    }
  }

  if (source) {
    for (let off = 0; off < rhs.length; off++) rhs[off] += source[off];
  }
}

// ---- Field projection and L2 error ----

export function projectField2D(mesh, fn, t = 0.0) {
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

export function l2Error2D(mesh, q, exactFn, t = 0.0) {
  const { Nx, Ny, n, M, xn, yn, dx, dy, basis } = mesh;
  const w = basis.w;
  let sum = 0;
  const jw = dx * dy * 0.25;     // jacobian dx_phys/dxi * dy_phys/deta
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
