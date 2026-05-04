// Dump the Session-D shaders so we can eyeball the GLSL.
// Usage: node sim/scripts/dg/sandbox/_dump_shaders.mjs
import {
  RDShaderDGPredictorInit,
  RDShaderDGPicard,
  RDShaderDGCorrector,
  RDShaderDGPredictorInitMRT,
  RDShaderDGPicardMRT,
} from "../../RD/simulation_shaders.js";

// Synthetic "user reaction body" mimicking what the parser would emit for
// the linearised SWE preset (h_t = -H_e u_x ; u_t = -g h_x with H_e = g = 1):
const fakeBody = {
  kineticUniformsGLSL: "// (no kineticParams)",
  body: `
  vec4 uvwq   = s.self;
  vec4 uvwqX  = (2.0 / u_dxElement) * dgddxi(s.nodeIdx, ${[0,1,2,3].map(i => "s.u" + i).join(", ")});
  vec4 uvwqY  = vec4(0.0);
  vec4 uvwqXX = vec4(0.0);
  vec4 uvwqYY = vec4(0.0);
  // Linearised SWE: h_t = -H_e * u_x, u_t = -g * h_x, with H_e = g = 1.
  float UFUN = -1.0 * uvwqX.g;   // h channel = uvwq.r
  float VFUN = -1.0 * uvwqX.r;   // u channel = uvwq.g
  float WFUN = 0.0;
  float QFUN = 0.0;
  return vec4(UFUN, VFUN, WFUN, QFUN);`,
};

for (const [name, src] of [
  ["RDShaderDGPredictorInit(3, fakeBody)",    RDShaderDGPredictorInit(3, fakeBody)],
  ["RDShaderDGPicard(3, fakeBody)",           RDShaderDGPicard(3, fakeBody)],
  ["RDShaderDGCorrector(3, fakeBody)",        RDShaderDGCorrector(3, fakeBody)],
  ["RDShaderDGPredictorInitMRT(3, fakeBody)", RDShaderDGPredictorInitMRT(3, fakeBody)],
  ["RDShaderDGPicardMRT(3, fakeBody)",        RDShaderDGPicardMRT(3, fakeBody)],
]) {
  console.log(`========== ${name} ==========`);
  console.log(src);
  console.log();
}
