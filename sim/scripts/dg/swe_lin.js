// swe_lin.js — Linearised shallow water equations.
//
// Linearised about the rest state (h, u, v) = (H_e, 0, 0):
//   1D:  h_t + H_e u_x        = 0
//        u_t + g   h_x        = 0
//   2D:  h_t + H_e (u_x+v_y)  = 0
//        u_t + g   h_x        = 0
//        v_t + g   h_y        = 0
//
// We carry conservative-style state q = (h, u[, v]) — already in primitive
// form for the linear system; the linear flux is constant-coefficient so
// we treat it that way. M = 2 in 1D, 3 in 2D, matching the nonlinear code.
//
// Closed-form standing wave (1D):
//   h(x,t) = A cos(2π x/L) cos(ω t),   ω = (2π/L) √(g H_e)
//   u(x,t) = (A √(g/H_e)) sin(2π x/L) sin(ω t)
// (1D wave: ∂_tt h = g H_e ∂_xx h; the velocity follows from h_t = -H_e u_x.)
//
// 2D standing wave (separable on a square periodic domain):
//   h(x,y,t) = A cos(2π x/L) cos(2π y/L) cos(ω t),   ω = (2π/L) √(2 g H_e)
//   u(x,y,t) = A √(g/H_e) (1/√2) sin(2π x/L) cos(2π y/L) sin(ω t)
//   v(x,y,t) = A √(g/H_e) (1/√2) cos(2π x/L) sin(2π y/L) sin(ω t)
// (Verifies the linear SWE system on the doubly-periodic torus.)

export const M1 = 2;
export const M2 = 3;

// ---------- 1D ----------

export function makeLinPhys1D({ g = 1.0, He = 1.0 } = {}) {
  const c0 = Math.sqrt(g * He);

  function flux1D(qIn, qInOff, fOut, fOutOff /*, gParam */) {
    // F = (H_e u, g h) for q = (h, u).
    fOut[fOutOff]     = He * qIn[qInOff + 1];
    fOut[fOutOff + 1] = g  * qIn[qInOff];
  }

  function rusanov1D(qL, qLOff, qR, qROff, fOut, fOutOff /*, gParam */) {
    const FL0 = He * qL[qLOff + 1];
    const FL1 = g  * qL[qLOff];
    const FR0 = He * qR[qROff + 1];
    const FR1 = g  * qR[qROff];
    fOut[fOutOff]     = 0.5 * (FL0 + FR0) - 0.5 * c0 * (qR[qROff]     - qL[qLOff]);
    fOut[fOutOff + 1] = 0.5 * (FL1 + FR1) - 0.5 * c0 * (qR[qROff + 1] - qL[qLOff + 1]);
  }

  function maxWavespeed1Dflat(/* q, M, gParam */) { return c0; }

  return { flux1D, rusanov1D, maxWavespeed1Dflat, c0, g, He };
}

// ---------- 2D ----------

export function makeLinPhys2D({ g = 1.0, He = 1.0 } = {}) {
  const c0 = Math.sqrt(g * He);

  function fluxX2D(qIn, qOff, F, FOff /*, gParam */) {
    F[FOff]     = He * qIn[qOff + 1];
    F[FOff + 1] = g  * qIn[qOff];
    F[FOff + 2] = 0;
  }
  function fluxY2D(qIn, qOff, F, FOff /*, gParam */) {
    F[FOff]     = He * qIn[qOff + 2];
    F[FOff + 1] = 0;
    F[FOff + 2] = g  * qIn[qOff];
  }
  function rusanov2D(qL, qLOff, qR, qROff, fOut, fOutOff, nx, ny /*, gParam */) {
    // Linear flux dotted with the face normal:
    //   F·n = (H_e (u nx + v ny), g h nx, g h ny)
    const FL0 = He * (qL[qLOff + 1] * nx + qL[qLOff + 2] * ny);
    const FL1 = g  *  qL[qLOff]     * nx;
    const FL2 = g  *  qL[qLOff]     * ny;
    const FR0 = He * (qR[qROff + 1] * nx + qR[qROff + 2] * ny);
    const FR1 = g  *  qR[qROff]     * nx;
    const FR2 = g  *  qR[qROff]     * ny;
    fOut[fOutOff]     = 0.5 * (FL0 + FR0) - 0.5 * c0 * (qR[qROff]     - qL[qLOff]);
    fOut[fOutOff + 1] = 0.5 * (FL1 + FR1) - 0.5 * c0 * (qR[qROff + 1] - qL[qLOff + 1]);
    fOut[fOutOff + 2] = 0.5 * (FL2 + FR2) - 0.5 * c0 * (qR[qROff + 2] - qL[qLOff + 2]);
  }
  function maxWavespeed2D(/* q, M, gParam */) { return c0; }
  return { fluxX2D, fluxY2D, rusanov2D, maxWavespeed2D, c0, g, He };
}

// ---------- Closed-form standing-wave reference ----------

export function standingWave1D({ L = 1.0, A = 0.1, g = 1.0, He = 1.0 } = {}) {
  const k = 2 * Math.PI / L;
  const omega = k * Math.sqrt(g * He);
  const u_amp = A * Math.sqrt(g / He);
  function exact(x, t) {
    const cx = Math.cos(k * x);
    const sx = Math.sin(k * x);
    const ct = Math.cos(omega * t);
    const st = Math.sin(omega * t);
    return [A * cx * ct, u_amp * sx * st];
  }
  return { exact, omega, period: 2 * Math.PI / omega, params: { L, A, g, He, k } };
}

export function standingWave2D({ L = 1.0, A = 0.1, g = 1.0, He = 1.0 } = {}) {
  const k = 2 * Math.PI / L;
  const omega = k * Math.sqrt(2 * g * He);
  const uv_amp = A * Math.sqrt(g / He) / Math.sqrt(2);
  function exact(x, y, t) {
    const cx = Math.cos(k * x), sx = Math.sin(k * x);
    const cy = Math.cos(k * y), sy = Math.sin(k * y);
    const ct = Math.cos(omega * t);
    const st = Math.sin(omega * t);
    return [
      A * cx * cy * ct,
      uv_amp * sx * cy * st,
      uv_amp * cx * sy * st,
    ];
  }
  return { exact, omega, period: 2 * Math.PI / omega, params: { L, A, g, He, k } };
}
