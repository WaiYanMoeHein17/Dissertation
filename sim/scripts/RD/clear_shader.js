// clear_shader.js

export function clearShaderTop() {
  return `varying vec2 textureCoords;
    uniform sampler2D textureSource;
    uniform sampler2D imageSourceOne;
    uniform sampler2D imageSourceTwo;
    uniform float dx;
    uniform float dy;
    uniform float L;
    uniform float L_x;
    uniform float L_y;
    uniform float L_min;
    uniform float t;
    uniform int nXDisc;
    uniform int nYDisc;
    uniform float seed;
    uniform int dgOrder;
    uniform float u_dxElement;

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

    // LGL node positions on [-1, 1] for the polynomial degrees this app
    // actually uses (p=2 for ADER3, p=3 for ADER4). Used in DG mode to
    // project the user's IC expression at the LGL physical positions
    // instead of uniform texel centres.
    float dgLglRefNodeP2(int i) {
      if (i == 0) return -1.0;
      if (i == 1) return  0.0;
      return 1.0;
    }
    float dgLglRefNodeP3(int i) {
      if (i == 0) return -1.0;
      if (i == 1) return -0.4472135954999579;
      if (i == 2) return  0.4472135954999579;
      return 1.0;
    }

    void main()
    {
        float x = MINX + textureCoords.x * float(nXDisc) * dx;
        float y = MINY + textureCoords.y * float(nYDisc) * dy;
        // ADER-DG mode: replace x with the physical LGL node position. Each
        // x-element of width u_dxElement holds (dgOrder + 1) LGL nodes;
        // texel j is the (j mod nodesPerElem)-th LGL node of the
        // (j / nodesPerElem)-th element. y stays on the uniform FD grid.
        if (dgOrder >= 2) {
            int globalIdx = int(floor(textureCoords.x * float(nXDisc)));
            int nodesPerElem = dgOrder + 1;
            int elemIdx = globalIdx / nodesPerElem;
            int nodeIdx = globalIdx - elemIdx * nodesPerElem;
            float refNode = dgOrder == 2
                ? dgLglRefNodeP2(nodeIdx)
                : dgLglRefNodeP3(nodeIdx);
            x = MINX + float(elemIdx) * u_dxElement
                     + 0.5 * u_dxElement * (refNode + 1.0);
        }
    `;
}

export function clearShaderBot() {
  return `
        gl_FragColor = vec4(u, v, w, q);
    }`;
}
