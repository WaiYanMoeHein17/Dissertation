// Verify RDShaderDGPreamble produces sane GLSL for orders 3 and 4 and that
// the embedded D matrix matches the reference computed by basis.js.
//
// Usage: node sim/scripts/dg/sandbox/_preamble_check.mjs

import { RDShaderDGPreamble } from "../../RD/simulation_shaders.js";
import { makeBasis } from "../basis.js";

function extractD(glsl, p) {
  const n = p + 1;
  const got = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      // Boundary-anchor with `\b` so D33 doesn't match DD33 etc.
      const m = glsl.match(new RegExp(`\\bconst float D${i}${j} = ([^;]+);`));
      if (!m) throw new Error(`missing D${i}${j} in preamble`);
      got[i * n + j] = parseFloat(m[1]);
    }
  }
  return got;
}

function extractDD(glsl, p) {
  const n = p + 1;
  const got = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const m = glsl.match(new RegExp(`\\bconst float DD${i}${j} = ([^;]+);`));
      if (!m) throw new Error(`missing DD${i}${j} in preamble`);
      got[i * n + j] = parseFloat(m[1]);
    }
  }
  return got;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Reference D² = D · D for verification.
function refD2(D, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += D[i * n + k] * D[k * n + j];
      out[i * n + j] = s;
    }
  }
  return out;
}

for (const p of [2, 3]) {
  const glsl = RDShaderDGPreamble(p);
  const got = extractD(glsl, p);
  const ref = makeBasis(p).D;
  const diff = maxAbsDiff(got, ref);
  const n = p + 1;
  console.log(`order ${p + 1} (DG poly degree ${p}, ${n} nodes/elem):`);
  console.log(`  GLSL preamble length: ${glsl.length} chars`);
  console.log(`  D matrix max-abs diff vs basis.js reference: ${diff.toExponential(3)}`);
  // D² check
  const gotDD = extractDD(glsl, p);
  const refDD = refD2(ref, n);
  const diffDD = maxAbsDiff(gotDD, refDD);
  console.log(`  D² matrix max-abs diff vs (D·D) reference: ${diffDD.toExponential(3)}`);
  // Spot check: each row of D should sum to 0 (constants are in the kernel).
  let maxRowSum = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += got[i * n + j];
    maxRowSum = Math.max(maxRowSum, Math.abs(sum));
  }
  console.log(`  max |row-sum| (should be ~0): ${maxRowSum.toExponential(3)}`);
  // Sanity: the LGL_NODES_PER_ELEM constant
  const nMatch = glsl.match(/const int LGL_NODES_PER_ELEM = (\d+);/);
  console.log(`  LGL_NODES_PER_ELEM = ${nMatch ? nMatch[1] : "MISSING"} (expected ${n})`);
  console.log();
}
