// simulation_shaders.js

// =====================================================================
// ADER-DG support
// =====================================================================
//
// The DG layout groups (p+1) consecutive texels into one element, with each
// texel holding the value at one Legendre-Gauss-Lobatto (LGL) node of that
// element. The LGL differentiation matrices below are baked in as
// compile-time GLSL constants by the JS emitter `RDShaderDGPreamble(p)`,
// which exposes:
//
//   * `Dij`           — entries of the (p+1)×(p+1) D matrix on [-1, 1]
//   * `LGL_NODES_PER_ELEM` — int constant = p+1
//   * `dgddxi(node, u0, ...)`     — du/dxi at `node` from the p+1 element values
//   * `dgSampleElement(tex, tc)`  — read all p+1 element values and identify
//                                    the current node index inside the element
//
// The emitter is parameterised on `p` so the same code path serves orders 3
// (p=3, 4 nodes) and 4 (p=4, 5 nodes). The DG element layout itself (texture
// sizing) is handled in main.js (setSizes); these shader helpers operate
// purely on the texture once it has the right shape.

// Closed-form LGL nodes on [-1, 1] for the orders we actually emit.
// (Trivial cases p=0 and p=1 are not used by RDShaderDGPreamble — orders 1
// and 2 in the paper's convention degenerate and are handled by FE/Mid.)
const _LGL_X = {
  2: [-1, 0, 1],
  3: [-1, -1 / Math.sqrt(5), 1 / Math.sqrt(5), 1],
  4: [-1, -Math.sqrt(3 / 7), 0, Math.sqrt(3 / 7), 1],
};

// Pre-computed LGL differentiation matrix D where (D u)_i = sum_j D_ij u_j
// approximates du/dxi at node i from the element-local nodal values u_j on
// the reference interval [-1, 1]. Computed via the standard barycentric
// Lagrange formula.
function _lglDiffMatrix(p) {
  const x = _LGL_X[p];
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

// Second-derivative matrix on [-1, 1]: D^(2) = D · D. Approximates
// d²u/dxi² at node i. Used for diffusion (u_xx) terms via the chain rule
//   d²u/dx² = (2/h)² · d²u/dxi² = (4/h²) · (D^(2) · u)
// where h is the physical element width.
function _lglSecondDiffMatrix(p) {
  const D = _lglDiffMatrix(p);
  const n = p + 1;
  const DD = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += D[i * n + k] * D[k * n + j];
      DD[i * n + j] = s;
    }
  }
  return DD;
}

function _glslFloat(x) {
  if (Number.isInteger(x)) return x.toFixed(1);
  // GLSL accepts standard scientific notation; full Float64 precision keeps
  // the GPU constants bit-identical to a CPU reference using the same matrix.
  return x.toExponential(15);
}

/**
 * Emit the GLSL preamble that backs an ADER-DG shader at the given DG
 * polynomial degree (p = 3 or 4). Returns a string suitable to inject into
 * any fragment shader that needs DG-aware spatial derivatives.
 *
 * The preamble is independent of the rest of the simulation's RHS pipeline:
 * it can sit alongside the existing finite-difference machinery so DG and
 * FD shaders coexist on the same texture (when DG is active, the texture
 * is sized to (p+1) texels per element by setSizes()).
 *
 * @param {2|3|4} p
 */
export function RDShaderDGPreamble(p) {
  if (p !== 2 && p !== 3 && p !== 4) {
    throw new Error(`RDShaderDGPreamble: unsupported DG order ${p} (2, 3 or 4 only)`);
  }
  const n = p + 1;
  const D  = _lglDiffMatrix(p);
  const DD = _lglSecondDiffMatrix(p);
  let dConsts = "";
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      dConsts += `const float D${i}${j} = ${_glslFloat(D[i * n + j])};\n`;
      dConsts += `const float DD${i}${j} = ${_glslFloat(DD[i * n + j])};\n`;
    }
  }

  // Build the ddxi function body: a row-select dispatch on nodeIdx.
  let ddxiBody = "  ";
  for (let i = 0; i < n; i++) {
    if (i > 0) ddxiBody += "  else ";
    ddxiBody += `if (node == ${i}) return `;
    const terms = [];
    for (let j = 0; j < n; j++) terms.push(`D${i}${j}*u${j}`);
    ddxiBody += terms.join(" + ") + ";\n";
  }
  ddxiBody += `  return vec4(0.0);\n`;

  // Build the ddxi2 (second-derivative) function body — same dispatch on
  // nodeIdx, but multiplies through the D^(2) row.
  let ddxi2Body = "  ";
  for (let i = 0; i < n; i++) {
    if (i > 0) ddxi2Body += "  else ";
    ddxi2Body += `if (node == ${i}) return `;
    const terms = [];
    for (let j = 0; j < n; j++) terms.push(`DD${i}${j}*u${j}`);
    ddxi2Body += terms.join(" + ") + ";\n";
  }
  ddxi2Body += `  return vec4(0.0);\n`;

  // Per-element sampler: extract the (p+1) element-local x-row values plus
  // the current node's value, its index within the element, and the
  // y-direction neighbour rows for the 4th-order central FD stencil that
  // the hybrid 1D-DG-x + FD-y layout uses for y-derivatives. In 1D mode
  // (texSize.y == 1) the y-neighbours all wrap back to self via the texture
  // wrap mode and the FD-4 stencil evaluates to zero.
  const sampleBody = [];
  for (let j = 0; j < n; j++) {
    sampleBody.push(
      `  s.u${j} = texture2D(tex, vec2((float(base + ${j}) + 0.5) * texelW, tc.y));`,
    );
  }

  return `
const int LGL_NODES_PER_ELEM = ${n};
${dConsts}

vec4 dgddxi(int node, ${Array.from({ length: n }, (_, i) => `vec4 u${i}`).join(", ")}) {
${ddxiBody}}

vec4 dgddxi2(int node, ${Array.from({ length: n }, (_, i) => `vec4 u${i}`).join(", ")}) {
${ddxi2Body}}

struct DGElemSamples {
  ${Array.from({ length: n }, (_, i) => `vec4 u${i};`).join(" ")}
  vec4 self;
  vec4 yT;  vec4 yB;  vec4 yTT; vec4 yBB;
  int  nodeIdx;
};

DGElemSamples dgSampleElement(sampler2D tex, vec2 tc) {
  ivec2 texSize = textureSize(tex, 0);
  float texelW = 1.0 / float(texSize.x);
  float texelH = 1.0 / float(texSize.y);
  int globalIdx = int(floor(tc.x * float(texSize.x)));
  int elemIdx   = globalIdx / LGL_NODES_PER_ELEM;
  int nodeIdx   = globalIdx - elemIdx * LGL_NODES_PER_ELEM;
  int base      = elemIdx * LGL_NODES_PER_ELEM;
  DGElemSamples s;
${sampleBody.join("\n")}
  s.self    = texture2D(tex, tc);
  s.yT      = texture2D(tex, tc + vec2(0.0, +texelH));
  s.yB      = texture2D(tex, tc + vec2(0.0, -texelH));
  s.yTT     = texture2D(tex, tc + vec2(0.0, +2.0 * texelH));
  s.yBB     = texture2D(tex, tc + vec2(0.0, -2.0 * texelH));
  s.nodeIdx = nodeIdx;
  return s;
}
`;
}

// =====================================================================
// ADER-DG predictor + Picard + corrector shader generators.
//
// Pass sequence per timestep (order 3, 5 passes; orders 4 has 7):
//   1. predictor init at GL2 node c1: q1_init = u^n + c1·dt·F(u^n)
//   2. predictor init at GL2 node c2: q2_init = u^n + c2·dt·F(u^n)
//   3. Picard refinement, row 1:      q1 = u^n + dt·(a11·F(q1_init) + a12·F(q2_init))
//   4. Picard refinement, row 2:      q2 = u^n + dt·(a21·F(q1_init) + a22·F(q2_init))
//   5. corrector:                     u^{n+1} = u^n + dt/2·(F̄(q1) + F̄(q2))
//                                              with local Lax-Friedrichs (Rusanov)
//                                              flux at element boundaries.
//
// The corresponding MRT-fused versions (RDShaderDG*MRT) collapse the two
// init passes into one and the two Picard rows into one each, halving the
// dispatch count.
//
// The element-local spatial RHS is supplied by the caller via the
// customRhsBody argument (the parser adapter buildDGComputeRHSBody
// in main.js); the default body, used for the linear-advection sandbox,
// is dgComputeRHS(s) = -dgFluxSpeed · u_x via the LGL D matrix.
// =====================================================================

const _ADER3_C1 = (3 - Math.sqrt(3)) / 6;
const _ADER3_C2 = (3 + Math.sqrt(3)) / 6;
const _ADER3_A11 = 0.25;
const _ADER3_A12 = _ADER3_C1 - 0.25;
const _ADER3_A21 = _ADER3_C2 - 0.25;
const _ADER3_A22 = 0.25;

// Common boilerplate every ADER-DG shader needs: precision, varyings,
// shared uniforms, the LGL preamble, and the dgComputeRHS hook.
//
// dgComputeRHS contract: the single function the ADER pipeline calls to
// compute the element-local spatial RHS at one LGL node, given the (p+1)
// element-local nodal values `s.u0`, …, `s.uP`, the current node index
// `s.nodeIdx`, and the current node's value `s.self`. It returns a vec4 =
// (RHS_R, RHS_G, RHS_B, RHS_A) with one entry per species (RGBA-packed).
//
// The default implementation hardcodes linear advection F(u) = a·u (with
// a = dgFluxSpeed), used by the sandbox convergence tests. Production
// callers inject their own body via customRhsBody — see
// buildDGComputeRHSBody in main.js for the parser-driven version.
function _aderHeader(p, customRhsBody) {
  // Optional kineticParamUniformsGLSL: declarations like
  //   "uniform float a; uniform float b; ..." — emitted by main.js so the
  //   user's reaction expressions can reference kineticParams uniforms.
  const kineticUniforms = customRhsBody && customRhsBody.kineticUniformsGLSL
    ? customRhsBody.kineticUniformsGLSL
    : "";
  const bodyText = customRhsBody && customRhsBody.body
    ? customRhsBody.body
    : `  vec4 du_dxi = dgddxi(s.nodeIdx, ${
        Array.from({ length: p + 1 }, (_, i) => `s.u${i}`).join(", ")
      });
  vec4 du_dx = (2.0 / u_dxElement) * du_dxi;
  return -dgFluxSpeed * du_dx;`;
  return [
    "precision highp float;",
    "precision highp sampler2D;",
    "varying vec2 textureCoords;",
    "uniform float dt;",
    "uniform float dx;",
    "uniform float dy;",
    "uniform float L;",
    "uniform float L_x;",
    "uniform float L_y;",
    "uniform float t;",
    "uniform float u_dxElement;",
    "uniform float dgFluxSpeed;        // advection wave speed (uniform a)",
    kineticUniforms,
    RDShaderDGPreamble(p),
    `// Element-local spatial-RHS hook. The body is supplied by main.js via`,
    `// is provided by the parser adapter (parseShaderString output) when`,
    `// customRhsBody is passed in; otherwise it falls back to linear`,
    `// advection F = -a·u_x.`,
    `vec4 dgComputeRHS(DGElemSamples s) {`,
    bodyText,
    `}`,
    ``,
    `// Backwards-compatibility alias: existing call sites used dgFluxLocal.`,
    `vec4 dgFluxLocal(DGElemSamples s) { return dgComputeRHS(s); }`,
  ].join("\n");
}

/**
 * Predictor-init shader: q = u^n + c·Δt·F(u^n) at one GL2 quadrature node.
 * The caller sets the `u_c` uniform to c1 (~0.2113) or c2 (~0.7887).
 */
export function RDShaderDGPredictorInit(p, customRhsBody) {
  return `
${_aderHeader(p, customRhsBody)}
uniform sampler2D textureSource;   // u^n
uniform float u_c;                 // GL2 node (c1 or c2)
void main() {
  DGElemSamples s = dgSampleElement(textureSource, textureCoords);
  vec4 F = dgFluxLocal(s);
  gl_FragColor = s.self + u_c * dt * F;
}
`;
}

/**
 * Picard-refinement shader: q_out = u^n + Δt·(a_α·F(q_α) + a_β·F(q_β)).
 * Inputs: textureSource = u^n, textureSource1 = q_α, textureSource2 = q_β.
 * The 2×2 weight matrix entries are passed as `u_aAlpha`, `u_aBeta`.
 */
export function RDShaderDGPicard(p, customRhsBody) {
  return `
${_aderHeader(p, customRhsBody)}
uniform sampler2D textureSource;   // u^n
uniform sampler2D textureSource1;  // q_alpha
uniform sampler2D textureSource2;  // q_beta
uniform float u_aAlpha;
uniform float u_aBeta;
void main() {
  DGElemSamples sa = dgSampleElement(textureSource1, textureCoords);
  DGElemSamples sb = dgSampleElement(textureSource2, textureCoords);
  vec4 F_a = dgFluxLocal(sa);
  vec4 F_b = dgFluxLocal(sb);
  vec4 u_n = texture2D(textureSource, textureCoords);
  gl_FragColor = u_n + dt * (u_aAlpha * F_a + u_aBeta * F_b);
}
`;
}

/**
 * Corrector shader: u^{n+1} = u^n + Δt·corrector RHS.
 * Volume term: time-averaged flux divergence using GL2 quadrature points
 * (predictor states q1, q2 in textureSource1, textureSource2). At element
 * boundary nodes (local index 0 = left, p = right), the volume term receives
 * a strong-form lifting correction using the local Lax-Friedrichs (Rusanov)
 * flux computed from the boundary trace of this element and its neighbour.
 *
 * Numerical flux: F̂ = ½(F_L + F_R) + ½α·(u_R − u_L) in the F̃ = −F_physical
 * convention used throughout dgInternalFlux / lfFlux. The sign on the
 * dissipation term flips relative to the standard convention because of
 * that flip; see lfFlux for the derivation.
 */
export function RDShaderDGCorrector(p, customRhsBody) {
  const n = p + 1;
  // Read the boundary traces of the LEFT-NEIGHBOUR element when current
  // node sits at the left edge of an element, and similarly the
  // right-neighbour element when current node sits at the right edge.
  // Texel layout: element k owns texels [k*n, k*n + n). Neighbour boundary
  // is the texel one element away in the corresponding direction.
  return `
${_aderHeader(p, customRhsBody)}
uniform sampler2D textureSource;   // u^n
uniform sampler2D textureSource1;  // q1 (predictor at GL2 node c1)
uniform sampler2D textureSource2;  // q2 (predictor at GL2 node c2)
uniform bool dgPeriodicX;          // x-direction BC mode for the lifting term
uniform bool dgUseDirichletX;      // constant-Dirichlet ghost-cell trace
uniform vec4 dgDirichletXValue;    // fixed value used at both x-edges

// Read the right-most node of the element to the LEFT of the current one,
// or the left-most node of the element to the RIGHT. BC priority at the
// domain edge: periodic wrap > constant-Dirichlet ghost > Neumann mirror.
// The Neumann fallback (neighbour = self) produces zero LF flux jump and
// thus zero boundary lifting. The Dirichlet branch returns a constant
// ghost-cell value, which the LF flux then compares against the interior
// trace — this lets a fixed boundary value drive a proper outward flux.
// Robin and x/t-dependent Dirichlet remain Neumann-like.
vec4 dgNeighbourLeftEdge(sampler2D tex) {
  ivec2 texSize = textureSize(tex, 0);
  float texelW = 1.0 / float(texSize.x);
  int globalIdx = int(floor(textureCoords.x * float(texSize.x)));
  int leftNbr   = globalIdx - 1;
  if (leftNbr < 0) {
    if (dgPeriodicX) {
      leftNbr += texSize.x;
    } else if (dgUseDirichletX) {
      return dgDirichletXValue;
    } else {
      return texture2D(tex, textureCoords);
    }
  }
  return texture2D(tex, vec2((float(leftNbr) + 0.5) * texelW, textureCoords.y));
}
vec4 dgNeighbourRightEdge(sampler2D tex) {
  ivec2 texSize = textureSize(tex, 0);
  float texelW = 1.0 / float(texSize.x);
  int globalIdx = int(floor(textureCoords.x * float(texSize.x)));
  int rightNbr  = globalIdx + 1;
  if (rightNbr >= texSize.x) {
    if (dgPeriodicX) {
      rightNbr -= texSize.x;
    } else if (dgUseDirichletX) {
      return dgDirichletXValue;
    } else {
      return texture2D(tex, textureCoords);
    }
  }
  return texture2D(tex, vec2((float(rightNbr) + 0.5) * texelW, textureCoords.y));
}
// Element-interface flux selector. The corrector's lifting term needs both
// the per-state internal flux F̃(U) and a numerical flux F̃̂(U_L, U_R). For
// dgFluxKind == 0 (linear advection) we keep the original convention
// F̃(U) = -dgFluxSpeed·U; for dgFluxKind == 1 (2D SWE in primitive form
// h=R, u=G, v=B) we use the conservative-form fluxes
//   F̃_h = -(h+H_e)·u,  F̃_u = -(g·h + u²/2),  F̃_v = -u·v.
// The max characteristic for SWE is |u| + sqrt(g·(h+H_e)); for linear
// advection it is |dgFluxSpeed|. The Rusanov dissipation uses the larger
// of the two trace estimates so the lifting is upwind-stable in both
// directions across the element interface.
uniform int dgFluxKind;
uniform float u_dgG;
uniform float u_dgHe;
vec4 dgInternalFlux(vec4 U) {
  if (dgFluxKind == 1) {
    float h = U.r, uVel = U.g, vVel = U.b;
    return vec4(
      -(h + u_dgHe) * uVel,
      -(u_dgG * h + 0.5 * uVel * uVel),
      -uVel * vVel,
      0.0
    );
  }
  return -dgFluxSpeed * U;
}
float dgMaxWaveSpeed(vec4 U) {
  if (dgFluxKind == 1) {
    float h = U.r, uVel = U.g;
    return abs(uVel) + sqrt(u_dgG * max(h + u_dgHe, 1e-6));
  }
  return abs(dgFluxSpeed);
}
vec4 lfFlux(vec4 uL, vec4 uR) {
  vec4 fL = dgInternalFlux(uL);
  vec4 fR = dgInternalFlux(uR);
  float alpha = max(dgMaxWaveSpeed(uL), dgMaxWaveSpeed(uR));
  // Dissipation has a + sign because dgInternalFlux emits F̃(U) = −F_physical:
  // the central-flux average is ½(F̃_L+F̃_R) and the upwind correction picks
  // up an extra minus from the F̃↔F flip, giving + ½ α (U_R − U_L). Sanity
  // check: for linear advection at speed a > 0 with U_L = 1, U_R = 0 this
  // formula yields F̃̂ = −1 = −a·u_L, the correct upwind trace
  // (F̂_standard = +a·u_L = +1).
  return 0.5 * (fL + fR) + 0.5 * alpha * (uR - uL);
}

void main() {
  DGElemSamples s1 = dgSampleElement(textureSource1, textureCoords);
  DGElemSamples s2 = dgSampleElement(textureSource2, textureCoords);
  vec4 F1 = dgFluxLocal(s1);
  vec4 F2 = dgFluxLocal(s2);

  // Volume contribution: 2-point GL quadrature average of F at the GL2 nodes.
  vec4 volume = 0.5 * (F1 + F2);

  // Boundary lifting: only fires at the left or right boundary node of an
  // element. nodeIdx == 0 is the left boundary; nodeIdx == p is the right.
  // Use the predictor-averaged trace at this boundary (½(q1+q2) at this node)
  // and at the co-located twin in the neighbouring element.
  vec4 lifting = vec4(0.0);
  if (s1.nodeIdx == 0) {
    // Trace from this element's left edge:
    vec4 uHere     = 0.5 * (s1.self + s2.self);
    // Trace from neighbour's right edge (its last node):
    vec4 uNbr      = 0.5 * (
      dgNeighbourLeftEdge(textureSource1) +
      dgNeighbourLeftEdge(textureSource2)
    );
    vec4 fHere     = dgInternalFlux(uHere);
    vec4 fStar     = lfFlux(uNbr, uHere);
    // Strong-form lifting at left boundary: + (F_internal - F̂) * (2/h)/w_0,
    // where w_0 is the LGL endpoint weight. For p=3, w_0 = 1/6 → 2/h/w_0 = 12/h.
    // (For p=4, w_0 = 1/10 → 2/h/w_0 = 20/h.) Encoded as LIFT_LEFT below.
    lifting += LIFT_LEFT * (fHere - fStar);
  } else if (s1.nodeIdx == LGL_NODES_PER_ELEM - 1) {
    vec4 uHere = 0.5 * (s1.self + s2.self);
    vec4 uNbr  = 0.5 * (
      dgNeighbourRightEdge(textureSource1) +
      dgNeighbourRightEdge(textureSource2)
    );
    vec4 fHere = dgInternalFlux(uHere);
    vec4 fStar = lfFlux(uHere, uNbr);
    lifting -= LIFT_RIGHT * (fHere - fStar);
  }

  vec4 u_n = texture2D(textureSource, textureCoords);
  gl_FragColor = u_n + dt * (volume + lifting);
}
`.replace(/LIFT_LEFT/g,  _liftCoeff(p, 0))
   .replace(/LIFT_RIGHT/g, _liftCoeff(p, p));
}

// LGL endpoint quadrature weight w_endpoint = 2 / (p (p+1)). The lifting
// coefficient (2/h) / w_endpoint = (p (p+1)) / h. We multiply by 1/h once
// (i.e., we leave the 1/h to be applied via u_dxElement) and bake the
// p(p+1) factor into the GLSL string. That gives a clean compile-time
// constant that uses u_dxElement at runtime.
function _liftCoeff(p, /* edgeIdx */ _) {
  const factor = p * (p + 1);  // 6 for p=2, 12 for p=3
  return `(${factor.toFixed(1)} / u_dxElement)`;
}

// =====================================================================
// ADER-DG MRT-fused shaders.
//
// These shaders write to multiple render targets simultaneously:
//   * RDShaderDGPredictorInitMRT(p): one pass producing both q1_init and
//     q2_init (instead of two separate passes at c1 and c2). Saves one
//     framebuffer rebind + one F-evaluation.
//   * RDShaderDGPicardMRT(p): one pass producing both Picard rows
//     (q_out_row1 and q_out_row2) from the same q_alpha, q_beta inputs.
//     Saves one framebuffer rebind plus one du/dx evaluation per node.
//
// Three.js compiles these in GLSL3 dialect (set via the material's
// `glslVersion: THREE.GLSL3` flag). With GLSL3 we declare the MRT
// outputs explicitly via `layout(location = N) out vec4 ...`.
// `varying` and `texture2D` are auto-rewritten by three.js's preamble.
//
// Pass count after MRT fusion:
//   Order 3: 5 → 3 (init MRT, Picard MRT, corrector)
//   Order 4: 7 → 4 (init MRT, Picard sweep 1 MRT, Picard sweep 2 MRT, corrector)
// =====================================================================

const _ADER3_FMT = (x) => Number.isInteger(x) ? x.toFixed(1) : x.toExponential(15);

/**
 * Fused predictor-init: one pass writes both q1_init = u^n + c1·Δt·F and
 * q2_init = u^n + c2·Δt·F. Output[0] = q1_init, output[1] = q2_init.
 */
export function RDShaderDGPredictorInitMRT(p, customRhsBody) {
  return `
${_aderHeader(p, customRhsBody)}
uniform sampler2D textureSource;
layout(location = 0) out vec4 q1Init;
layout(location = 1) out vec4 q2Init;

const float MRT_C1 = ${_ADER3_FMT(_ADER3_C1)};
const float MRT_C2 = ${_ADER3_FMT(_ADER3_C2)};

void main() {
  DGElemSamples s = dgSampleElement(textureSource, textureCoords);
  vec4 F = dgFluxLocal(s);
  q1Init = s.self + MRT_C1 * dt * F;
  q2Init = s.self + MRT_C2 * dt * F;
}
`;
}

/**
 * Fused Picard refinement: one pass evaluates F at each of two inputs and
 * uses both rows of the 2×2 Picard weight matrix to produce two outputs.
 * Output[0] = u^n + Δt·(a11·F_α + a12·F_β); Output[1] = ditto with row 2.
 */
export function RDShaderDGPicardMRT(p, customRhsBody) {
  return `
${_aderHeader(p, customRhsBody)}
uniform sampler2D textureSource;     // u^n
uniform sampler2D textureSource1;    // q_alpha
uniform sampler2D textureSource2;    // q_beta
layout(location = 0) out vec4 q1Out;
layout(location = 1) out vec4 q2Out;

const float MRT_A11 = ${_ADER3_FMT(_ADER3_A11)};
const float MRT_A12 = ${_ADER3_FMT(_ADER3_A12)};
const float MRT_A21 = ${_ADER3_FMT(_ADER3_A21)};
const float MRT_A22 = ${_ADER3_FMT(_ADER3_A22)};

void main() {
  DGElemSamples sa = dgSampleElement(textureSource1, textureCoords);
  DGElemSamples sb = dgSampleElement(textureSource2, textureCoords);
  vec4 F_a = dgFluxLocal(sa);
  vec4 F_b = dgFluxLocal(sb);
  vec4 u_n = texture2D(textureSource, textureCoords);
  q1Out = u_n + dt * (MRT_A11 * F_a + MRT_A12 * F_b);
  q2Out = u_n + dt * (MRT_A21 * F_a + MRT_A22 * F_b);
}
`;
}

/**
 * Generates the top part of a shader for reaction-diffusion simulation based on the given timestepping scheme.
 * @param {string} type - The timestepping scheme to generate the shader for.
 * @returns {string} The generated shader code.
 */
export function RDShaderTop(type) {
  let numInputs = 0;
  switch (type) {
    case "FE":
      numInputs = 2;
      break;
    case "AB2":
      numInputs = 2;
      break;
    case "Mid1":
      numInputs = 1;
      break;
    case "Mid2":
      numInputs = 2;
      break;
    case "RK41":
      numInputs = 1;
      break;
    case "RK42":
      numInputs = 2;
      break;
    case "RK43":
      numInputs = 3;
      break;
    case "RK44":
      numInputs = 4;
      break;
  }
  let parts = [];
  parts[0] =
    "precision highp float; precision highp sampler2D; varying vec2 textureCoords;";
  parts[1] = "uniform sampler2D textureSource;";
  parts[2] = "uniform sampler2D textureSource1;";
  parts[3] = "uniform sampler2D textureSource2;";
  parts[4] = "uniform sampler2D textureSource3;";
  return (
    parts.slice(0, numInputs + 1).join("\n") +
    `
    uniform float dt;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform float seed;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;

    AUXILIARY_GLSL_FUNS

    const float ALPHA = 0.147;
    const float INV_ALPHA = 1.0 / ALPHA;
    const float BETA = 2.0 / (pi * ALPHA);
    float erfinv(float pERF) {
      float yERF;
      if (pERF == -1.0) {
        yERF = log(1.0 - (-0.99)*(-0.99));
      } else {
        yERF = log(1.0 - pERF*pERF);
      }
      float zERF = BETA + 0.5 * yERF;
      return sqrt(sqrt(zERF*zERF - yERF * INV_ALPHA) - zERF) * sign(pERF);
    }

    void computeRHS(sampler2D textureSource, vec4 uvwqIn, vec4 uvwqLIn, vec4 uvwqRIn, vec4 uvwqTIn, vec4 uvwqBIn, vec4 uvwqLLIn, vec4 uvwqRRIn, vec4 uvwqTTIn, vec4 uvwqBBIn, out highp vec4 result) {

        ivec2 texSize = textureSize(textureSource,0);
        float step_x = 1.0 / float(texSize.x);
        float step_y = 1.0 / float(texSize.y);
        float x = textureCoords.x * L_x + MINX;
        float y = textureCoords.y * L_y + MINY;
        float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
        float exterior = 1.0 - interior;
        vec2 dSquared = 1.0/vec2(dx*dx, dy*dy);
        vec2 textureCoordsL = textureCoords + vec2(-step_x, 0.0);
        vec2 textureCoordsLL = textureCoordsL + vec2(-step_x, 0.0);
        vec2 textureCoordsR = textureCoords + vec2(+step_x, 0.0);
        vec2 textureCoordsRR = textureCoordsR + vec2(+step_x, 0.0);
        vec2 textureCoordsT = textureCoords + vec2(0.0, +step_y);
        vec2 textureCoordsTT = textureCoordsT + vec2(0.0, +step_y);
        vec2 textureCoordsB = textureCoords + vec2(0.0, -step_y);
        vec2 textureCoordsBB = textureCoordsB + vec2(0.0, -step_y);

        vec4 uvwq = uvwqIn;
        vec4 uvwqL = uvwqLIn;
        vec4 uvwqLL = uvwqLLIn;
        vec4 uvwqR = uvwqRIn;
        vec4 uvwqRR = uvwqRRIn;
        vec4 uvwqT = uvwqTIn;
        vec4 uvwqTT = uvwqTTIn;
        vec4 uvwqB = uvwqBIn;
        vec4 uvwqBB = uvwqBBIn;
    `
  );
}

/**
 * Generates shader code based on the timestepping scheme.
 * @param {string} type - The type of timestepping scheme ("FE", "AB2", "Mid1", "Mid2", "RK41", "RK42", "RK43", "RK44").
 * @returns {string} - The generated shader code.
 */
export function RDShaderMain(type) {
  let update = {};
  update.FE = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwq;`;
  update.AB2 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS1);
    uvwq = texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource1, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS2);
    RHS = 1.5 * RHS1 - 0.5 * RHS2;
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + texture2D(textureSource, textureCoords);`;
  update.Mid1 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.Mid2 = `uvwqLast = texture2D(textureSource, textureCoords);
    uvwq = uvwqLast + 0.5*dt*texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwqLast;`;
  update.RK41 = `uvwq = texture2D(textureSource, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK42 = `uvwq = texture2D(textureSource, textureCoords) + 0.5*dt*texture2D(textureSource1, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource1, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource1, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource1, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource1, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource1, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource1, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource1, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource1, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK43 = `uvwq = texture2D(textureSource, textureCoords) + 0.5*dt*texture2D(textureSource2, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + 0.5*dt*texture2D(textureSource2, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + 0.5*dt*texture2D(textureSource2, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + 0.5*dt*texture2D(textureSource2, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + 0.5*dt*texture2D(textureSource2, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + 0.5*dt*texture2D(textureSource2, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + 0.5*dt*texture2D(textureSource2, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + 0.5*dt*texture2D(textureSource2, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + 0.5*dt*texture2D(textureSource2, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS);
    vec4 timescales = TIMESCALES;
    updated = RHS;`;
  update.RK44 = `uvwqLast = texture2D(textureSource, textureCoords);
    uvwq = uvwqLast + dt*texture2D(textureSource3, textureCoords);
    uvwqL = texture2D(textureSource, textureCoordsL) + dt*texture2D(textureSource3, textureCoordsL);
    uvwqR = texture2D(textureSource, textureCoordsR) + dt*texture2D(textureSource3, textureCoordsR);
    uvwqT = texture2D(textureSource, textureCoordsT) + dt*texture2D(textureSource3, textureCoordsT);
    uvwqB = texture2D(textureSource, textureCoordsB) + dt*texture2D(textureSource3, textureCoordsB);
    uvwqLL = texture2D(textureSource, textureCoordsLL) + dt*texture2D(textureSource3, textureCoordsLL);
    uvwqRR = texture2D(textureSource, textureCoordsRR) + dt*texture2D(textureSource3, textureCoordsRR);
    uvwqTT = texture2D(textureSource, textureCoordsTT) + dt*texture2D(textureSource3, textureCoordsTT);
    uvwqBB = texture2D(textureSource, textureCoordsBB) + dt*texture2D(textureSource3, textureCoordsBB);
    computeRHS(textureSource, uvwq, uvwqL, uvwqR, uvwqT, uvwqB, uvwqLL, uvwqRR, uvwqTT, uvwqBB, RHS1);
    RHS = (texture2D(textureSource1, textureCoords) + 2.0*(texture2D(textureSource2, textureCoords) + texture2D(textureSource3, textureCoords)) + RHS1) / 6.0;
    vec4 timescales = TIMESCALES;
    updated = dt * RHS / timescales + uvwqLast;`;
  return (
    `
  void main()
  {
      ivec2 texSize = textureSize(textureSource,0);
      float step_x = 1.0 / float(texSize.x);
      float step_y = 1.0 / float(texSize.y);
      float x = textureCoords.x * L_x + MINX;
      float y = textureCoords.y * L_y + MINY;
      float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
      float exterior = 1.0 - interior;

      vec2 textureCoordsL = textureCoords + vec2(-step_x, 0.0);
      vec2 textureCoordsLL = textureCoordsL + vec2(-step_x, 0.0);
      vec2 textureCoordsR = textureCoords + vec2(+step_x, 0.0);
      vec2 textureCoordsRR = textureCoordsR + vec2(+step_x, 0.0);
      vec2 textureCoordsT = textureCoords + vec2(0.0, +step_y);
      vec2 textureCoordsTT = textureCoordsT + vec2(0.0, +step_y);
      vec2 textureCoordsB = textureCoords + vec2(0.0, -step_y);
      vec2 textureCoordsBB = textureCoordsB + vec2(0.0, -step_y);
      
      vec4 RHS;
      vec4 RHS1;
      vec4 RHS2;
      vec4 updated;
      vec4 uvwq;
      vec4 uvwqL;
      vec4 uvwqLL;
      vec4 uvwqR;
      vec4 uvwqRR;
      vec4 uvwqT;
      vec4 uvwqTT;
      vec4 uvwqB;
      vec4 uvwqBB;
      vec4 uvwqLast;
  ` + update[type]
  );
}

/**
 * Returns the shader code for a reaction-diffusion simulation with periodic boundary conditions.
 * @returns {string} The shader code.
 */
export function RDShaderPeriodic() {
  return ``;
}

/**
 * Generates shader code for specifying the values of ghost cells in the x-direction.
 * @param {string} [LR] - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @returns {string} The shader code for setting the species of ghost cells in the x-direction.
 */
export function RDShaderGhostX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        uvwqL.SPECIES = GHOSTSPECIES;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        uvwqR.SPECIES = GHOSTSPECIES;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Generates shader code for specifying the values of ghost cells in the y-direction.
 * @param {string} [TB] - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @returns {string} The shader code for setting the species of ghost cells in the y-direction.
 */
export function RDShaderGhostY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0){
        uvwqT.SPECIES = GHOSTSPECIES;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        uvwqB.SPECIES = GHOSTSPECIES;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns a string containing the Robin boundary condition shader code in the x-direction.
 * @param {string} [LR] - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @returns {string} The Robin boundary condition shader code.
 */
export function RDShaderRobinX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        uvwqL.SPECIES = 2.0 * (dx * robinRHSSPECIESL) + uvwqR.SPECIES;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        uvwqR.SPECIES = 2.0 * (dx * robinRHSSPECIESR) + uvwqL.SPECIES;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Returns a string containing the Robin boundary condition shader code in the y-direction.
 * @param {string} [TB] - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @returns {string} The Robin boundary condition shader code.
 */
export function RDShaderRobinY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0){
        uvwqT.SPECIES = 2.0 * (dy * robinRHSSPECIEST) + uvwqB.SPECIES;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        uvwqB.SPECIES = 2.0 * (dy * robinRHSSPECIESB) + uvwqT.SPECIES;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Generates a Robin boundary condition shader for a custom domain in the x-direction.
 * @param {string} TB - Determines whether to apply the condition at the left ("L"), right ("R"), or both ("LR"). If undefined, returns both.
 * @param {string} fun - A function that defines the custom domain.
 * @returns {string} The generated shader code.
 */
export function RDShaderRobinCustomDomainX(LR, fun) {
  const L = `
    if (float(indicatorFunL) <= 0.0 || textureCoords.x - 2.0*step_x < 0.0) {
      if (float(indicatorFunR) <= 0.0) {
        uvwqL.SPECIES = dx * robinRHSSPECIESL + uvwq.SPECIES;
      } else {
        uvwqL.SPECIES = 2.0 * (dx * robinRHSSPECIESL) + uvwqR.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunL/,
      fun.replaceAll(/\bx\b/g, "(x-1.25*dx)").replaceAll(/\buvwq\./g, "uvwqL."),
    )
    .replace(
      /indicatorFunR/,
      fun.replaceAll(/\bx\b/g, "(x+1.25*dx)").replaceAll(/\buvwq\./g, "uvwqR."),
    );
  const R = `
    if (float(indicatorFunR) <= 0.0 || textureCoords.x + 2.0*step_x > 1.0) {
      if (float(indicatorFunL) <= 0.0) {
        uvwqR.SPECIES = dx * robinRHSSPECIESR + uvwq.SPECIES;
      } else {
        uvwqR.SPECIES = 2.0 * (dx * robinRHSSPECIESR) + uvwqL.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunR/,
      fun.replaceAll(/\bx\b/g, "(x+1.25*dx)").replaceAll(/\buvwq\./g, "uvwqR."),
    )
    .replace(
      /indicatorFunL/,
      fun.replaceAll(/\bx\b/g, "(x-1.25*dx)").replaceAll(/\buvwq\./g, "uvwqL."),
    );
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Generates a Robin boundary condition shader for a custom domain in the y-direction.
 * @param {string} TB - Determines whether to apply the condition at the top ("T"), bottom ("B"), or both ("TB"). If undefined, returns both.
 * @param {string} fun - A function that defines the custom domain.
 * @returns {string} The generated shader code.
 */
export function RDShaderRobinCustomDomainY(TB, fun) {
  const T = `
    if (float(indicatorFunT) <= 0.0 || textureCoords.y + 2.0*step_y > 1.0){
      if (float(indicatorFunB) <= 0.0) {
        uvwqT.SPECIES = dy * robinRHSSPECIEST + uvwq.SPECIES;
      } else {
        uvwqT.SPECIES = 2.0 * (dy * robinRHSSPECIEST) + uvwqB.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunT/,
      fun.replaceAll(/\by\b/g, "(y+1.25*dy)").replaceAll(/\buvwq\./g, "uvwqT."),
    )
    .replace(
      /indicatorFunB/,
      fun.replaceAll(/\by\b/g, "(y-1.25*dy)").replaceAll(/\buvwq\./g, "uvwqB."),
    );
  const B = `
    if (float(indicatorFunB) <= 0.0 || textureCoords.y - 2.0*step_y < 0.0) {
      if (float(indicatorFunT) <= 0.0) {
        uvwqB.SPECIES = dy * robinRHSSPECIESB + uvwq.SPECIES;
      } else {
        uvwqB.SPECIES = 2.0 * (dy * robinRHSSPECIESB) + uvwqT.SPECIES;
      }
    }
    `
    .replace(
      /indicatorFunB/,
      fun.replaceAll(/\by\b/g, "(y-1.25*dy)").replaceAll(/\buvwq\./g, "uvwqB."),
    )
    .replace(
      /indicatorFunT/,
      fun.replaceAll(/\by\b/g, "(y+1.25*dy)").replaceAll(/\buvwq\./g, "uvwqT."),
    );
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns the shader code for computing advection before boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderAdvectionPreBC() {
  return `
    vec4 uvwqX = (uvwqR - uvwqL) / (2.0*dx);
    vec4 uvwqY = (uvwqT - uvwqB) / (2.0*dy);
    vec4 uvwqXF = (uvwqR - uvwq) / dx;
    vec4 uvwqYF = (uvwqT - uvwq) / dy;
    vec4 uvwqXB = (uvwq - uvwqL) / dx;
    vec4 uvwqYB = (uvwq - uvwqB) / dy;
    vec4 uvwqXFXF = (4.0*uvwqR - 3.0*uvwq - uvwqRR) / (2.0*dx);
    vec4 uvwqYFYF = (4.0*uvwqT - 3.0*uvwq - uvwqTT) / (2.0*dy);
    vec4 uvwqXBXB = (3.0*uvwq - 4.0*uvwqL + uvwqLL) / (2.0*dx);
    vec4 uvwqYBYB = (3.0*uvwq - 4.0*uvwqB + uvwqBB) / (2.0*dy);
    `;
}

/**
 * Returns the shader code for computing advection after boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderAdvectionPostBC() {
  return `
    uvwqX = (uvwqR - uvwqL) / (2.0*dx);
    uvwqY = (uvwqT - uvwqB) / (2.0*dy);
    uvwqXF = (uvwqR - uvwq) / dx;
    uvwqYF = (uvwqT - uvwq) / dy;
    uvwqXB = (uvwq - uvwqL) / dx;
    uvwqYB = (uvwq - uvwqB) / dy;
    uvwqXFXF = (4.0*uvwqR - 3.0*uvwq - uvwqRR) / (2.0*dx);
    uvwqYFYF = (4.0*uvwqT - 3.0*uvwq - uvwqTT) / (2.0*dy);
    uvwqXBXB = (3.0*uvwq - 4.0*uvwqL + uvwqLL) / (2.0*dx);
    uvwqYBYB = (3.0*uvwq - 4.0*uvwqB + uvwqBB) / (2.0*dy);
    `;
}

/**
 * Returns the shader code for computing diffusion before boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderDiffusionPreBC() {
  return `
    vec4 uvwqXX = (uvwqR - 2.0*uvwq + uvwqL) / (dx*dx);
    vec4 uvwqYY = (uvwqT - 2.0*uvwq + uvwqB) / (dy*dy);
    `;
}

/**
 * Returns the shader code for computing diffusion after boundary conditions have been applied.
 * @returns {string} The shader code.
 */
export function RDShaderDiffusionPostBC() {
  return `
    uvwqXX = (uvwqR - 2.0*uvwq + uvwqL) / (dx*dx);
    uvwqYY = (uvwqT - 2.0*uvwq + uvwqB) / (dy*dy);
    `;
}

/**
 * Generates a shader for updating a reaction-diffusion system without cross diffusion.
 * @param {number} [numSpecies=4] - The number of species. Defaults to 4.
 * @returns {string} - The shader code for the update.
 */
export function RDShaderUpdateNormal(numSpecies) {
  if (numSpecies == undefined) numSpecies = 4;
  let shader = "";
  shader += `
  float LDuuU = 0.5*((Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r)) / dx) / dx +  0.5*((Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r)) / dy) / dy;
  float du = LDuuU + UFUN;
  `;
  if (numSpecies > 1) {
    shader += `
    float LDvvV = 0.5*((Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g)) / dx) / dx +  0.5*((Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g)) / dy) / dy;
    float dv = LDvvV + VFUN;
    `;
  }
  if (numSpecies > 2) {
    shader += `
    float LDwwW = 0.5*((Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b)) / dx) / dx +  0.5*((Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b)) / dy) / dy;
    float dw = LDwwW + WFUN;
    `;
  }
  if (numSpecies > 3) {
    shader += `
    float LDqqQ = 0.5*((Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a)) / dx) / dx +  0.5*((Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a)) / dy) / dy;
    float dq = LDqqQ + QFUN;
    `;
  }
  // Add the final line of the shader.
  switch (numSpecies) {
    case 1:
      shader += `result = vec4(du,0.0,0.0,0.0);`;
      break;
    case 2:
      shader += `result = vec4(du,dv,0.0,0.0);`;
      break;
    case 3:
      shader += `result = vec4(du,dv,dw,0.0);`;
      break;
    case 4:
      shader += `result = vec4(du,dv,dw,dq);`;
      break;
  }
  return (
    shader +
    `
    }`
  );
}

/**
 * Generates a shader for updating a reaction-diffusion system with cross diffusion.
 * @param {number} [numSpecies=4] - The number of species in the system.
 * @returns {string} The generated shader code.
 */
export function RDShaderUpdateCross(numSpecies) {
  if (numSpecies == undefined) numSpecies = 4;
  let shader = "";
  shader +=
    [
      `vec2 LDuuU = vec2(Duux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DuuxR*(uvwqR.r - uvwq.r) + DuuxL*(uvwqL.r - uvwq.r), Duuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DuuyT*(uvwqT.r - uvwq.r) + DuuyB*(uvwqB.r - uvwq.r));`,
      `vec2 LDuvV = vec2(Duvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DuvxR*(uvwqR.g - uvwq.g) + DuvxL*(uvwqL.g - uvwq.g), Duvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DuvyT*(uvwqT.g - uvwq.g) + DuvyB*(uvwqB.g - uvwq.g));`,
      `vec2 LDuwW = vec2(Duwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DuwxR*(uvwqR.b - uvwq.b) + DuwxL*(uvwqL.b - uvwq.b), Duwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DuwyT*(uvwqT.b - uvwq.b) + DuwyB*(uvwqB.b - uvwq.b));`,
      `vec2 LDuqQ = vec2(Duqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DuqxR*(uvwqR.a - uvwq.a) + DuqxL*(uvwqL.a - uvwq.a), Duqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DuqyT*(uvwqT.a - uvwq.a) + DuqyB*(uvwqB.a - uvwq.a));`,
    ]
      .slice(0, numSpecies)
      .join("\n") +
    `\nfloat du = 0.5*dot(dSquared,` +
    [`LDuuU`, `LDuvV`, `LDuwW`, `LDuqQ`].slice(0, numSpecies).join(" + ") +
    `) + UFUN;\n`;
  // If there is more than one species, add the second species.
  if (numSpecies > 1) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDvuU = vec2(Dvux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DvuxR*(uvwqR.r - uvwq.r) + DvuxL*(uvwqL.r - uvwq.r), Dvuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DvuyT*(uvwqT.r - uvwq.r) + DvuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDvvV = vec2(Dvvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DvvxR*(uvwqR.g - uvwq.g) + DvvxL*(uvwqL.g - uvwq.g), Dvvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DvvyT*(uvwqT.g - uvwq.g) + DvvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDvwW = vec2(Dvwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DvwxR*(uvwqR.b - uvwq.b) + DvwxL*(uvwqL.b - uvwq.b), Dvwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DvwyT*(uvwqT.b - uvwq.b) + DvwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDvqQ = vec2(Dvqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DvqxR*(uvwqR.a - uvwq.a) + DvqxL*(uvwqL.a - uvwq.a), Dvqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DvqyT*(uvwqT.a - uvwq.a) + DvqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dv = 0.5*dot(dSquared,` +
      [`LDvuU`, `LDvvV`, `LDvwW`, `LDvqQ`].slice(0, numSpecies).join(" + ") +
      `) + VFUN;\n`;
  }
  // If there are more than two species, add the third species.
  if (numSpecies > 2) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDwuU = vec2(Dwux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DwuxR*(uvwqR.r - uvwq.r) + DwuxL*(uvwqL.r - uvwq.r), Dwuy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DwuyT*(uvwqT.r - uvwq.r) + DwuyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDwvV = vec2(Dwvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DwvxR*(uvwqR.g - uvwq.g) + DwvxL*(uvwqL.g - uvwq.g), Dwvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DwvyT*(uvwqT.g - uvwq.g) + DwvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDwwW = vec2(Dwwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DwwxR*(uvwqR.b - uvwq.b) + DwwxL*(uvwqL.b - uvwq.b), Dwwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DwwyT*(uvwqT.b - uvwq.b) + DwwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDwqQ = vec2(Dwqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DwqxR*(uvwqR.a - uvwq.a) + DwqxL*(uvwqL.a - uvwq.a), Dwqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DwqyT*(uvwqT.a - uvwq.a) + DwqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dw = 0.5*dot(dSquared,` +
      [`LDwuU`, `LDwvV`, `LDwwW`, `LDwqQ`].slice(0, numSpecies).join(" + ") +
      `) + WFUN;\n`;
  }
  // If there are more than three species, add the fourth species.
  if (numSpecies > 3) {
    // Compute the cross-diffusion terms.
    shader +=
      [
        `vec2 LDquU = vec2(Dqux*(uvwqR.r + uvwqL.r - 2.0*uvwq.r) + DquxR*(uvwqR.r - uvwq.r) + DquxL*(uvwqL.r - uvwq.r), Dquy*(uvwqT.r + uvwqB.r - 2.0*uvwq.r) + DquyT*(uvwqT.r - uvwq.r) + DquyB*(uvwqB.r - uvwq.r));`,
        `vec2 LDqvV = vec2(Dqvx*(uvwqR.g + uvwqL.g - 2.0*uvwq.g) + DqvxR*(uvwqR.g - uvwq.g) + DqvxL*(uvwqL.g - uvwq.g), Dqvy*(uvwqT.g + uvwqB.g - 2.0*uvwq.g) + DqvyT*(uvwqT.g - uvwq.g) + DqvyB*(uvwqB.g - uvwq.g));`,
        `vec2 LDqwW = vec2(Dqwx*(uvwqR.b + uvwqL.b - 2.0*uvwq.b) + DqwxR*(uvwqR.b - uvwq.b) + DqwxL*(uvwqL.b - uvwq.b), Dqwy*(uvwqT.b + uvwqB.b - 2.0*uvwq.b) + DqwyT*(uvwqT.b - uvwq.b) + DqwyB*(uvwqB.b - uvwq.b));`,
        `vec2 LDqqQ = vec2(Dqqx*(uvwqR.a + uvwqL.a - 2.0*uvwq.a) + DqqxR*(uvwqR.a - uvwq.a) + DqqxL*(uvwqL.a - uvwq.a), Dqqy*(uvwqT.a + uvwqB.a - 2.0*uvwq.a) + DqqyT*(uvwqT.a - uvwq.a) + DqqyB*(uvwqB.a - uvwq.a));`,
      ]
        .slice(0, numSpecies)
        .join("\n") +
      `\nfloat dq = 0.5*dot(dSquared,` +
      [`LDquU`, `LDqvV`, `LDqwW`, `LDqqQ`].slice(0, numSpecies).join(" + ") +
      `) + QFUN;\n`;
  }
  // Add the final line of the shader.
  switch (numSpecies) {
    case 1:
      shader += `result = vec4(du,0.0,0.0,0.0);`;
      break;
    case 2:
      shader += `result = vec4(du,dv,0.0,0.0);`;
      break;
    case 3:
      shader += `result = vec4(du,dv,dw,0.0);`;
      break;
    case 4:
      shader += `result = vec4(du,dv,dw,dq);`;
      break;
  }
  return (
    shader +
    `
    }`
  );
}

/**
 * Returns the shader code for updating the algebraic species in a reaction-diffusion simulation.
 * @returns {string} The shader code for updating the algebraic species.
 */
export function RDShaderAlgebraicSpecies() {
  return `
    updated.SPECIES = RHS.SPECIES / timescales.SPECIES;
    `;
}

/**
 * Returns the shader code for applying Dirichlet boundary conditions in the x-direction.
 * @param {string} [LR] - Optional argument to specify whether to return the shader code for the left boundary ("L"), right boundary ("R"), or both boundaries (undefined).
 * @returns {string} The shader code for applying Dirichlet boundary conditions in the x-direction.
 */
export function RDShaderDirichletX(LR) {
  const L = `
    if (textureCoords.x - step_x < 0.0) {
        updated.SPECIES = dirichletRHSSPECIESL;
    }
    `;
  const R = `
    if (textureCoords.x + step_x > 1.0) {
        updated.SPECIES = dirichletRHSSPECIESR;
    }
    `;
  if (LR == undefined) return L + R;
  if (LR == "L") return L;
  if (LR == "R") return R;
  return "";
}

/**
 * Returns the shader code for applying Dirichlet boundary conditions in the y-direction.
 * @param {string} [LR] - Optional argument to specify whether to return the shader code for the top boundary ("T"), bottom boundary ("B"), or both boundaries (undefined).
 * @returns {string} The shader code for applying Dirichlet boundary conditions in the y-direction.
 */
export function RDShaderDirichletY(TB) {
  const T = `
    if (textureCoords.y + step_y > 1.0) {
        updated.SPECIES = dirichletRHSSPECIEST;
    }
    `;
  const B = `
    if (textureCoords.y - step_y < 0.0) {
        updated.SPECIES = dirichletRHSSPECIESB;
    }
    `;
  if (TB == undefined) return T + B;
  if (TB == "T") return T;
  if (TB == "B") return B;
  return "";
}

/**
 * Returns a shader fragment that updates the SPECIES based on an indicator function.
 * @returns {string} The shader function as a string.
 */
export function RDShaderDirichletIndicatorFun() {
  return `
    if (float(indicatorFun) <= 0.0) {
        updated.SPECIES = `;
}

/**
 * Returns the final part of shader code for a reaction-diffusion simulation.
 * @returns {string} The shader code.
 */
export function RDShaderBot() {
  return ` 
    gl_FragColor = updated;
}`;
}

/**
 * Returns the top part of shader code for enforcing Dirichlet boundary conditions.
 * @returns {string} The shader code.
 */
export function RDShaderEnforceDirichletTop() {
  return `precision highp float;
    varying vec2 textureCoords;
    uniform sampler2D textureSource;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;

    AUXILIARY_GLSL_FUNS

    const float ALPHA = 0.147;
    const float INV_ALPHA = 1.0 / ALPHA;
    const float BETA = 2.0 / (pi * ALPHA);
    float erfinv(float pERF) {
      float yERF;
      if (pERF == -1.0) {
        yERF = log(1.0 - (-0.99)*(-0.99));
      } else {
        yERF = log(1.0 - pERF*pERF);
      }
      float zERF = BETA + 0.5 * yERF;
      return sqrt(sqrt(zERF*zERF - yERF * INV_ALPHA) - zERF) * sign(pERF);
    }
    
    void main()
    {
        ivec2 texSize = textureSize(textureSource,0);
        float step_x = 1.0 / float(texSize.x);
        float step_y = 1.0 / float(texSize.y);
        float x = textureCoords.x * L_x + MINX;
        float y = textureCoords.y * L_y + MINY;
        float interior = float(textureCoords.x > 0.75*step_x && textureCoords.x < 1.0 - 0.75*step_x && textureCoords.y > 0.5*step_y && textureCoords.y < 1.0 - 0.75*step_y);
        float exterior = 1.0 - interior;

        vec4 uvwq = texture2D(textureSource, textureCoords);
        gl_FragColor = uvwq;
    `;
}

/**
 * Generates shader code for clamping species values to the edge of a texture in a given direction.
 * @param {string} direction - The direction in which to clamp the species values. Can include "H" for horizontal and/or "V" for vertical.
 * @returns {string} The generated GLSL code.
 */
export function clampSpeciesToEdgeShader(direction) {
  let out = "";
  if (direction.includes("H")) {
    out += `
    if (textureCoords.x - step_x < 0.0) {
      uvwqL.SPECIES = uvwq.SPECIES;
    }
    if (textureCoords.x + step_x > 1.0) {
      uvwqR.SPECIES = uvwq.SPECIES;
    }`;
  }
  if (direction.includes("V")) {
    out += `
    if (textureCoords.y + step_y > 1.0) {
      uvwqT.SPECIES = uvwq.SPECIES;
    }
    if (textureCoords.y - step_y < 0.0) {
      uvwqB.SPECIES = uvwq.SPECIES;
    }`;
  }
  return out;
}
