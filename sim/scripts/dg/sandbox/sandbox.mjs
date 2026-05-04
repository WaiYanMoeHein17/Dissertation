// sandbox.mjs — GPU/CPU equivalence test for the ADER-DG predictor.
//
// Builds a 1D DG-layout texture (16 elements × 4 LGL nodes), runs the full
// p=3 predictor on the GPU, and verifies it pointwise against an identical
// CPU computation using the same LGL D matrix.
//
// Predictor pipeline (no MRT, four sequential passes):
//   1. q1_init = u^n + c1·Δt·F(u^n)              [init at GL2 node c1]
//   2. q2_init = u^n + c2·Δt·F(u^n)              [init at GL2 node c2]
//   3. q1     = u^n + Δt·(a11·F(q1_init) + a12·F(q2_init))   [Picard, row 1]
//   4. q2     = u^n + Δt·(a21·F(q1_init) + a22·F(q2_init))   [Picard, row 2]
//
// Test PDE: linear advection u_t + a·u_x = 0 on x ∈ [0,1] periodic. The
// element-local F(u) = -a · u_x is computed via the LGL differentiation
// matrix; no inter-element flux is applied (this is the predictor stage).
//
// p = 3 here is chosen to exercise the same shader-generator parameter as
// the production ADER4 path (which uses 4 LGL nodes/element). The p = 2
// case used by ADER3 in production is verified instead by _preamble_check.mjs
// against the basis.js reference, since the shader generator is parametric
// on p and a successful parameterised verification at p ∈ {2, 3} covers
// both production paths.

import { makeBasis } from "../basis.js";

// =====================================================================
// Problem set-up
// =====================================================================

const P = 3;                                  // DG polynomial degree
const N_NODES = P + 1;                        // 4 nodes per element
const N_ELEM = 16;
const N_X = N_ELEM * N_NODES;                 // 64 texels in x
const N_Y = 1;
const L = 1.0;
const DX_ELEMENT = L / N_ELEM;                // 0.0625
const DT = 0.001;
const ADVECTION_SPEED = 1.0;

const C1 = (3 - Math.sqrt(3)) / 6;            // 0.21132...
const C2 = (3 + Math.sqrt(3)) / 6;            // 0.78867...

// Picard 2×2 weight matrix (Eq. 12 in the paper):
//   a11 = 1/4,    a12 = c1 - 1/4,
//   a21 = c2-1/4, a22 = 1/4.
const A11 = 0.25;
const A12 = C1 - 0.25;
const A21 = C2 - 0.25;
const A22 = 0.25;

const basis = makeBasis(P);                   // { x, w, D, n: 4, p: 3 }

// Per-texel physical x: element-local LGL node mapped to [x_left, x_left+dx_elem].
function physicalXAtTexel(j) {
  const e = Math.floor(j / N_NODES);
  const i = j - e * N_NODES;
  const xLeft = e * DX_ELEMENT;
  return xLeft + 0.5 * DX_ELEMENT * (basis.x[i] + 1.0);
}

// Initial condition: u(x, 0) = sin(2 pi x).
function initialState() {
  const u = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) u[j] = Math.sin(2 * Math.PI * physicalXAtTexel(j));
  return u;
}

// =====================================================================
// CPU reference for each pass
// =====================================================================

// Compute F = -a·u_x at every node using the LGL D matrix (no inter-element
// coupling). Returns a fresh Float32Array.
function elementLocalFlux(u) {
  const D = basis.D;
  const n = N_NODES;
  const out = new Float32Array(N_X);
  const inv = 2.0 / DX_ELEMENT;
  for (let e = 0; e < N_ELEM; e++) {
    for (let i = 0; i < n; i++) {
      let du_dxi = 0;
      for (let k = 0; k < n; k++) du_dxi += D[i * n + k] * u[e * n + k];
      out[e * n + i] = -ADVECTION_SPEED * inv * du_dxi;
    }
  }
  return out;
}

// CPU reference for the full order-3 predictor.
function cpuPredictor(u) {
  const F_un = elementLocalFlux(u);
  const q1Init = new Float32Array(N_X);
  const q2Init = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) {
    q1Init[j] = u[j] + C1 * DT * F_un[j];
    q2Init[j] = u[j] + C2 * DT * F_un[j];
  }
  const F_q1 = elementLocalFlux(q1Init);
  const F_q2 = elementLocalFlux(q2Init);
  const q1 = new Float32Array(N_X);
  const q2 = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) {
    q1[j] = u[j] + DT * (A11 * F_q1[j] + A12 * F_q2[j]);
    q2[j] = u[j] + DT * (A21 * F_q1[j] + A22 * F_q2[j]);
  }
  return { q1Init, q2Init, q1, q2 };
}

// =====================================================================
// WebGL2 helpers
// =====================================================================

function setupGL() {
  const glCanvas = document.createElement("canvas");
  glCanvas.width = N_X;
  glCanvas.height = N_Y;
  const gl = glCanvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");
  if (!gl.getExtension("EXT_color_buffer_float")) {
    throw new Error("EXT_color_buffer_float is required (RGBA32F render targets).");
  }
  return gl;
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const msg = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Shader compile error:\n" + msg + "\n--- source ---\n" + src);
  }
  return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const msg = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error("Program link error:\n" + msg);
  }
  return prog;
}

// Create a 1D-style RGBA32F texture; if `dataR` is provided, pack it into the R channel.
function makeRGBA32FTexture(gl, dataR /* optional Float32Array length N_X */) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  let rgba = null;
  if (dataR) {
    rgba = new Float32Array(N_X * N_Y * 4);
    for (let j = 0; j < N_X; j++) rgba[j * 4] = dataR[j];
  }
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, N_X, N_Y, 0, gl.RGBA, gl.FLOAT, rgba);
  return tex;
}

function makeFBO(gl, attachTex) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, attachTex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("Framebuffer incomplete (RGBA32F render target).");
  }
  return fbo;
}

function readbackR(gl, fbo) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const px = new Float32Array(N_X * N_Y * 4);
  gl.readPixels(0, 0, N_X, N_Y, gl.RGBA, gl.FLOAT, px);
  const out = new Float32Array(N_X);
  for (let j = 0; j < N_X; j++) out[j] = px[j * 4];
  return out;
}

// =====================================================================
// Shader sources
// =====================================================================

const VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_texCoord;
void main() {
  v_texCoord = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// LGL D matrix as compile-time literals + helper that returns du/dxi at the
// current node from a 4-vector of node values. Shared by both shaders below.
function dgPreamble() {
  const D = basis.D;
  const fmt = (x) => Number.isInteger(x) ? x.toFixed(1) : x.toExponential(15);
  let dConsts = "";
  for (let i = 0; i < N_NODES; i++) {
    for (let j = 0; j < N_NODES; j++) {
      dConsts += `const float D${i}${j} = ${fmt(D[i * N_NODES + j])};\n`;
    }
  }
  return `
const float C1 = ${fmt(C1)};
const float C2 = ${fmt(C2)};
const int N_NODES = ${N_NODES};

${dConsts}

// Compute du/dxi at node `node` from a 4-vector u of element-local LGL values.
vec4 ddxi(int node, vec4 u0, vec4 u1, vec4 u2, vec4 u3) {
  if (node == 0)      return D00*u0 + D01*u1 + D02*u2 + D03*u3;
  else if (node == 1) return D10*u0 + D11*u1 + D12*u2 + D13*u3;
  else if (node == 2) return D20*u0 + D21*u1 + D22*u2 + D23*u3;
  else                return D30*u0 + D31*u1 + D32*u2 + D33*u3;
}

struct ElemSamples { vec4 u0; vec4 u1; vec4 u2; vec4 u3; vec4 self; int nodeIdx; };

ElemSamples sampleElement(sampler2D tex, vec2 v_texCoord) {
  ivec2 texSize = textureSize(tex, 0);
  float texelW = 1.0 / float(texSize.x);
  int globalIdx = int(floor(v_texCoord.x * float(texSize.x)));
  int elemIdx = globalIdx / N_NODES;
  int nodeIdx = globalIdx - elemIdx * N_NODES;
  int base = elemIdx * N_NODES;
  ElemSamples s;
  s.u0 = texture(tex, vec2((float(base + 0) + 0.5) * texelW, v_texCoord.y));
  s.u1 = texture(tex, vec2((float(base + 1) + 0.5) * texelW, v_texCoord.y));
  s.u2 = texture(tex, vec2((float(base + 2) + 0.5) * texelW, v_texCoord.y));
  s.u3 = texture(tex, vec2((float(base + 3) + 0.5) * texelW, v_texCoord.y));
  s.self = texture(tex, v_texCoord);
  s.nodeIdx = nodeIdx;
  return s;
}
`;
}

// Init pass: q = u_state + c·Δt·F(u_state). Used twice (c = c1, c = c2).
function buildInitShader() {
  return `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D u_state;
uniform float u_dt;
uniform float u_c;
uniform float u_dxElement;
uniform float u_advectionSpeed;

in vec2 v_texCoord;
out vec4 outColor;

${dgPreamble()}

void main() {
  ElemSamples s = sampleElement(u_state, v_texCoord);
  vec4 du_dx = (2.0 / u_dxElement) * ddxi(s.nodeIdx, s.u0, s.u1, s.u2, s.u3);
  vec4 F = -u_advectionSpeed * du_dx;
  outColor = s.self + u_c * u_dt * F;
}`;
}

// Picard pass: q_out = u^n + Δt·(a_alpha·F(q_alpha) + a_beta·F(q_beta)).
// Used twice (alpha=q1_init, beta=q2_init; weights from rows of the
// Picard weight matrix).
function buildPicardShader() {
  return `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D u_state;     // u^n
uniform sampler2D u_qAlpha;    // first predictor state
uniform sampler2D u_qBeta;     // second predictor state
uniform float u_dt;
uniform float u_aAlpha;
uniform float u_aBeta;
uniform float u_dxElement;
uniform float u_advectionSpeed;

in vec2 v_texCoord;
out vec4 outColor;

${dgPreamble()}

void main() {
  ElemSamples a = sampleElement(u_qAlpha, v_texCoord);
  ElemSamples b = sampleElement(u_qBeta,  v_texCoord);
  ElemSamples u = sampleElement(u_state,  v_texCoord);
  vec4 du_dx_a = (2.0 / u_dxElement) * ddxi(a.nodeIdx, a.u0, a.u1, a.u2, a.u3);
  vec4 du_dx_b = (2.0 / u_dxElement) * ddxi(b.nodeIdx, b.u0, b.u1, b.u2, b.u3);
  vec4 F_a = -u_advectionSpeed * du_dx_a;
  vec4 F_b = -u_advectionSpeed * du_dx_b;
  outColor = u.self + u_dt * (u_aAlpha * F_a + u_aBeta * F_b);
}`;
}

// =====================================================================
// GPU pipeline
// =====================================================================

function runGpuPredictor(uInit) {
  const gl = setupGL();
  const initProg   = linkProgram(gl, VERT_SRC, buildInitShader());
  const picardProg = linkProgram(gl, VERT_SRC, buildPicardShader());

  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );

  // Textures: u^n, q1_init, q2_init, q1, q2.
  const texU       = makeRGBA32FTexture(gl, uInit);
  const texQ1Init  = makeRGBA32FTexture(gl);
  const texQ2Init  = makeRGBA32FTexture(gl);
  const texQ1      = makeRGBA32FTexture(gl);
  const texQ2      = makeRGBA32FTexture(gl);

  const fboQ1Init = makeFBO(gl, texQ1Init);
  const fboQ2Init = makeFBO(gl, texQ2Init);
  const fboQ1     = makeFBO(gl, texQ1);
  const fboQ2     = makeFBO(gl, texQ2);

  function bindQuad(prog) {
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const aPos = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  }

  // ---- Pass 1: init at c1 → texQ1Init ----
  bindQuad(initProg);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboQ1Init);
  gl.viewport(0, 0, N_X, N_Y);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texU);
  gl.uniform1i(gl.getUniformLocation(initProg, "u_state"), 0);
  gl.uniform1f(gl.getUniformLocation(initProg, "u_dt"), DT);
  gl.uniform1f(gl.getUniformLocation(initProg, "u_c"), C1);
  gl.uniform1f(gl.getUniformLocation(initProg, "u_dxElement"), DX_ELEMENT);
  gl.uniform1f(gl.getUniformLocation(initProg, "u_advectionSpeed"), ADVECTION_SPEED);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // ---- Pass 2: init at c2 → texQ2Init ----
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboQ2Init);
  gl.uniform1f(gl.getUniformLocation(initProg, "u_c"), C2);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // ---- Pass 3: Picard row 1 → texQ1 ----
  bindQuad(picardProg);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboQ1);
  gl.viewport(0, 0, N_X, N_Y);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texU);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texQ1Init);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texQ2Init);
  gl.uniform1i(gl.getUniformLocation(picardProg, "u_state"),  0);
  gl.uniform1i(gl.getUniformLocation(picardProg, "u_qAlpha"), 1);
  gl.uniform1i(gl.getUniformLocation(picardProg, "u_qBeta"),  2);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_dt"), DT);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_aAlpha"), A11);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_aBeta"),  A12);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_dxElement"), DX_ELEMENT);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_advectionSpeed"), ADVECTION_SPEED);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // ---- Pass 4: Picard row 2 → texQ2 ----
  gl.bindFramebuffer(gl.FRAMEBUFFER, fboQ2);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_aAlpha"), A21);
  gl.uniform1f(gl.getUniformLocation(picardProg, "u_aBeta"),  A22);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // ---- Read back ----
  const result = {
    q1Init: readbackR(gl, fboQ1Init),
    q2Init: readbackR(gl, fboQ2Init),
    q1:     readbackR(gl, fboQ1),
    q2:     readbackR(gl, fboQ2),
  };

  // Tear down
  for (const t of [texU, texQ1Init, texQ2Init, texQ1, texQ2]) gl.deleteTexture(t);
  for (const f of [fboQ1Init, fboQ2Init, fboQ1, fboQ2]) gl.deleteFramebuffer(f);
  gl.deleteBuffer(vbo);
  gl.deleteProgram(initProg);
  gl.deleteProgram(picardProg);

  return result;
}

// =====================================================================
// Visualisation
// =====================================================================

function plotLines(canvas, traces) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const padL = 30, padR = 10, padT = 10, padB = 25;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const yMin = -1.1, yMax = 1.1;

  ctx.strokeStyle = "#bbb"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  const yZero = padT + plotH * (yMax / (yMax - yMin));
  ctx.strokeStyle = "#eee";
  ctx.beginPath(); ctx.moveTo(padL, yZero); ctx.lineTo(padL + plotW, yZero); ctx.stroke();
  ctx.strokeStyle = "#eef3ff";
  for (let e = 0; e <= N_ELEM; e++) {
    const xPx = padL + plotW * (e / N_ELEM);
    ctx.beginPath(); ctx.moveTo(xPx, padT); ctx.lineTo(xPx, padT + plotH); ctx.stroke();
  }
  for (const { arr, color, dashed } of traces) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    for (let j = 0; j < N_X; j++) {
      const x = physicalXAtTexel(j);
      const xPx = padL + plotW * x / L;
      const yPx = padT + plotH * (yMax - arr[j]) / (yMax - yMin);
      if (j === 0) ctx.moveTo(xPx, yPx); else ctx.lineTo(xPx, yPx);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    for (let j = 0; j < N_X; j++) {
      const x = physicalXAtTexel(j);
      const xPx = padL + plotW * x / L;
      const yPx = padT + plotH * (yMax - arr[j]) / (yMax - yMin);
      ctx.beginPath(); ctx.arc(xPx, yPx, 1.6, 0, 2 * Math.PI); ctx.fill();
    }
  }
  ctx.fillStyle = "#444"; ctx.font = "11px monospace";
  ctx.fillText("x", padL + plotW / 2 - 4, padT + plotH + 18);
  ctx.fillText("u", 6, padT + plotH / 2);
  ctx.fillText("0", padL - 10, yZero + 4);
}

// =====================================================================
// Drive
// =====================================================================

function compare(a, b) {
  let maxAbs = 0, sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    sumSq += d * d;
  }
  return { maxAbs, l2: Math.sqrt(sumSq / a.length) };
}

function fmt(x) { return x.toExponential(3); }

function main() {
  const out = document.getElementById("output");
  const canvas = document.getElementById("plot");
  try {
    const uInit = initialState();
    const cpu = cpuPredictor(uInit);
    const gpu = runGpuPredictor(uInit);

    const dQ1Init = compare(cpu.q1Init, gpu.q1Init);
    const dQ2Init = compare(cpu.q2Init, gpu.q2Init);
    const dQ1     = compare(cpu.q1,     gpu.q1);
    const dQ2     = compare(cpu.q2,     gpu.q2);

    const tol = 1e-5;
    const allPass =
      dQ1Init.maxAbs < tol && dQ2Init.maxAbs < tol &&
      dQ1.maxAbs < tol && dQ2.maxAbs < tol;
    const status = allPass
      ? `<span class="ok">PASS</span> (max|cpu-gpu| &lt; ${tol} for all four predictor states)`
      : `<span class="bad">FAIL</span> (some predictor state exceeds tol = ${tol})`;

    out.innerHTML = [
      `${status}`,
      ``,
      `domain L          = ${L}`,
      `elements          = ${N_ELEM}`,
      `nodes per element = ${N_NODES}`,
      `texels in x       = ${N_X}`,
      `Δt                = ${DT}`,
      `c1, c2            = ${C1.toFixed(15)}, ${C2.toFixed(15)}`,
      `Picard weights    = a11=${A11.toFixed(15)}, a12=${A12.toFixed(15)}`,
      `                    a21=${A21.toFixed(15)}, a22=${A22.toFixed(15)}`,
      ``,
      `||q1_init_cpu - q1_init_gpu||_inf = ${fmt(dQ1Init.maxAbs)}    L2 = ${fmt(dQ1Init.l2)}`,
      `||q2_init_cpu - q2_init_gpu||_inf = ${fmt(dQ2Init.maxAbs)}    L2 = ${fmt(dQ2Init.l2)}`,
      `||q1_cpu      - q1_gpu||_inf      = ${fmt(dQ1.maxAbs)}    L2 = ${fmt(dQ1.l2)}`,
      `||q2_cpu      - q2_gpu||_inf      = ${fmt(dQ2.maxAbs)}    L2 = ${fmt(dQ2.l2)}`,
      ``,
      `||q1 - u^n||_inf = ${fmt(Math.max(...gpu.q1.map((v, i) => Math.abs(v - uInit[i]))))}`,
      `||q2 - u^n||_inf = ${fmt(Math.max(...gpu.q2.map((v, i) => Math.abs(v - uInit[i]))))}`,
    ].join("\n");

    plotLines(canvas, [
      { arr: uInit,   color: "#1f77b4", dashed: false },
      { arr: cpu.q1,  color: "#2ca02c", dashed: false },
      { arr: gpu.q1,  color: "#d62728", dashed: true  },
      { arr: cpu.q2,  color: "#9467bd", dashed: false },
      { arr: gpu.q2,  color: "#ff7f0e", dashed: true  },
    ]);
  } catch (err) {
    out.innerHTML = `<span class="bad">ERROR</span>\n${err.message}`;
    console.error(err);
  }
}

main();
