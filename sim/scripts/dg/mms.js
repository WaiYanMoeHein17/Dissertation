// mms.js — Manufactured solutions for SWE convergence verification.
//
// Add the analytic source S = q_t + div F(q) to the SWE so the chosen ansatz
// is an exact solution. L2 error is measured against q_exact.
//
// 1D ansatz (periodic, doubly-smooth, fully nonlinear):
//   h(x,t) = H0 + a sin(k x - omega t)
//   u(x,t) = U0 + b cos(k x - omega t)        with k = 2 pi / L
//
// 2D ansatz (separable + travelling):
//   h(x,y,t) = H0 + a sin(kx - wt) sin(ky)
//   u(x,y,t) = U0 + b cos(kx - wt) sin(ky)
//   v(x,y,t) = V0 + b sin(kx - wt) cos(ky)

export function makeMMS1D({ L = 1.0, H0 = 1.0, U0 = 0.3, a = 0.1, b = 0.1, omega = 1.5, g = 1.0 } = {}) {
  const k = 2 * Math.PI / L;
  function exact(x, t) {
    const s = Math.sin(k * x - omega * t);
    const c = Math.cos(k * x - omega * t);
    const h = H0 + a * s;
    const u = U0 + b * c;
    return [h, h * u];
  }
  function source(x, t) {
    const s = Math.sin(k * x - omega * t);
    const c = Math.cos(k * x - omega * t);
    const h  = H0 + a * s;
    const u  = U0 + b * c;
    const h_t = -a * omega * c;
    const u_t =  b * omega * s;
    const h_x =  a * k * c;
    const u_x = -b * k * s;
    const S0 = h_t + h_x * u + h * u_x;
    const hu_t = h_t * u + h * u_t;
    const hu2_x = h_x * u * u + 2 * h * u * u_x;
    const S1 = hu_t + hu2_x + g * h * h_x;
    return [S0, S1];
  }
  return { exact, source, params: { L, H0, U0, a, b, omega, g, k } };
}

export function makeMMS2D({ L = 1.0, H0 = 1.0, U0 = 0.2, V0 = 0.1, a = 0.1, b = 0.05, omega = 1.5, g = 1.0 } = {}) {
  const k = 2 * Math.PI / L;
  function exact(x, y, t) {
    const sx = Math.sin(k * x - omega * t), cx = Math.cos(k * x - omega * t);
    const sy = Math.sin(k * y),             cy = Math.cos(k * y);
    const h = H0 + a * sx * sy;
    const u = U0 + b * cx * sy;
    const v = V0 + b * sx * cy;
    return [h, h * u, h * v];
  }
  function source(x, y, t) {
    const sx = Math.sin(k * x - omega * t), cx = Math.cos(k * x - omega * t);
    const sy = Math.sin(k * y),             cy = Math.cos(k * y);
    const h = H0 + a * sx * sy;
    const u = U0 + b * cx * sy;
    const v = V0 + b * sx * cy;
    // first derivatives
    const h_t = -a * omega * cx * sy;
    const u_t =  b * omega * sx * sy;
    const v_t = -b * omega * cx * cy;
    const h_x =  a * k * cx * sy;
    const h_y =  a * k * sx * cy;
    const u_x = -b * k * sx * sy;
    const u_y =  b * k * cx * cy;
    const v_x =  b * k * cx * cy;
    const v_y = -b * k * sx * sy;
    // continuity
    const S0 = h_t + (h_x * u + h * u_x) + (h_y * v + h * v_y);
    // x-momentum
    const hu_t  = h_t * u + h * u_t;
    const hu2_x = h_x * u * u + 2 * h * u * u_x;
    const huv_y = h_y * u * v + h * (u_y * v + u * v_y);
    const S1 = hu_t + hu2_x + g * h * h_x + huv_y;
    // y-momentum
    const hv_t  = h_t * v + h * v_t;
    const huv_x = h_x * u * v + h * (u_x * v + u * v_x);
    const hv2_y = h_y * v * v + 2 * h * v * v_y;
    const S2 = hv_t + huv_x + hv2_y + g * h * h_y;
    return [S0, S1, S2];
  }
  return { exact, source, params: { L, H0, U0, V0, a, b, omega, g, k } };
}

// Build the source array for the entire mesh at time t.
export function buildSourceField1D(mesh, sourceFn, t) {
  const { Nx, n, M, xn } = mesh;
  const arr = new Float64Array(Nx * n * M);
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const s = sourceFn(xn[i * n + ix], t);
      const off = (i * n + ix) * M;
      for (let k = 0; k < M; k++) arr[off + k] = s[k];
    }
  }
  return arr;
}

export function buildSourceField2D(mesh, sourceFn, t) {
  const { Nx, Ny, n, M, xn, yn } = mesh;
  const arr = new Float64Array(Nx * Ny * n * n * M);
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      for (let jy = 0; jy < n; jy++) {
        for (let jx = 0; jx < n; jx++) {
          const idx = ((iy * Nx + ix) * n + jy) * n + jx;
          const s = sourceFn(xn[idx], yn[idx], t);
          const off = idx * M;
          for (let k = 0; k < M; k++) arr[off + k] = s[k];
        }
      }
    }
  }
  return arr;
}
