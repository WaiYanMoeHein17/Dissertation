// basis.js — Legendre-Gauss-Lobatto (LGL) nodal DG basis on [-1, 1].
//
// LGL collocation gives a diagonal mass matrix on the reference element,
// which keeps both RK-DG and ADER-DG linear-algebra-free (no factorisation
// of the mass matrix). Quadrature accuracy of LGL on n = p+1 nodes is 2p-1.
//
// Exports:
//   lglNodesWeights(p)            -> { x: Float64Array, w: Float64Array }
//   legendreEval(n, x)            -> number              (P_n at scalar x)
//   diffMatrix(x)                 -> Float64Array(n*n)   (row-major)
//   makeBasis(p)                  -> Basis object

// Evaluate (P_n(x), P_n'(x)) using the standard three-term recurrences.
function legendreAndDerivative(n, x) {
  if (n === 0) return [1, 0];
  if (n === 1) return [x, 1];
  let P0 = 1, P1 = x;
  let dP0 = 0, dP1 = 1;
  for (let k = 1; k < n; k++) {
    const P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1);
    // (1-x^2) P_{k+1}'(x) = (k+1) (P_k(x) - x P_{k+1}(x))
    // Equivalent recurrence: P_{k+1}'(x) = (2k+1) P_k(x) + P_{k-1}'(x)
    const dP2 = (2 * k + 1) * P1 + dP0;
    P0 = P1; P1 = P2;
    dP0 = dP1; dP1 = dP2;
  }
  return [P1, dP1];
}

// Newton iteration on P_p'(x) = 0 to find an interior LGL node, starting
// from a good initial guess (Chebyshev-Lobatto points).  Uses the identity
//     P_p''(x) (1 - x^2) = 2 x P_p'(x) - p(p+1) P_p(x)
// as the second derivative.
function newtonInteriorNode(p, x0) {
  let x = x0;
  for (let it = 0; it < 50; it++) {
    const [P, dP] = legendreAndDerivative(p, x);
    if (Math.abs(1 - x * x) < 1e-15) break;
    const ddP = (2 * x * dP - p * (p + 1) * P) / (1 - x * x);
    const dx = dP / ddP;
    x -= dx;
    if (Math.abs(dx) < 1e-15) break;
  }
  return x;
}

export function lglNodesWeights(p) {
  const n = p + 1;
  if (n === 2) return { x: new Float64Array([-1, 1]), w: new Float64Array([1, 1]) };
  const x = new Float64Array(n);
  x[0] = -1; x[n - 1] = 1;
  // Interior nodes: zeros of P_p'(x). Use Chebyshev-Lobatto starting guess and Newton.
  // The interior LGL nodes interlace with the Chebyshev-Lobatto points.
  for (let i = 1; i < n - 1; i++) {
    const x0 = -Math.cos(Math.PI * i / p);
    x[i] = newtonInteriorNode(p, x0);
  }
  // Weights: w_i = 2 / ( p (p+1) [P_p(x_i)]^2 )
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const Pp = legendreEval(p, x[i]);
    w[i] = 2 / (p * (p + 1) * Pp * Pp);
  }
  return { x, w };
}

export function legendreEval(n, x) {
  if (n === 0) return 1;
  if (n === 1) return x;
  let P0 = 1, P1 = x;
  for (let k = 1; k < n; k++) {
    const P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1);
    P0 = P1; P1 = P2;
  }
  return P1;
}

// Differentiation matrix for Lagrange basis on nodes x.
// D[i*n + j] = ell_j'(x_i). Barycentric formula.
export function diffMatrix(x) {
  const n = x.length;
  const wbar = new Float64Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) if (i !== j) wbar[i] /= (x[i] - x[j]);
  }
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    let diag = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const v = (wbar[j] / wbar[i]) / (x[i] - x[j]);
      D[i * n + j] = v;
      diag -= v;
    }
    D[i * n + i] = diag;
  }
  return D;
}

// Lagrange interpolation matrix L (m x n): L[i*n+j] = ell_j(xi[i]).
export function interpMatrix(xNodes, xi) {
  const n = xNodes.length, m = xi.length;
  const L = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let num = 1, den = 1;
      for (let k = 0; k < n; k++) {
        if (k === j) continue;
        num *= (xi[i] - xNodes[k]);
        den *= (xNodes[j] - xNodes[k]);
      }
      L[i * n + j] = num / den;
    }
  }
  return L;
}

export function makeBasis(p) {
  const { x, w } = lglNodesWeights(p);
  const D = diffMatrix(x);
  return { p, n: p + 1, x, w, D };
}
