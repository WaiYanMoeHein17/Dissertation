// Run a quick MMS convergence test for 1D and 2D RK-DG and ADER-DG.
// Usage:  node sim/scripts/dg/_smoketest.mjs
//
// Expected: rates near (p+1) for both schemes, both 1D and 2D.

import { runConvergence1D, runConvergence2D } from "./bench.js";

function fmt(x) { return x.toExponential(3); }
function pad(s, n) { s = String(s); while (s.length < n) s = " " + s; return s; }

function report(label, res) {
  console.log(`\n=== ${label} ===`);
  console.log("  N     dx          err         rate     steps   wall(ms)");
  for (let i = 0; i < res.Ns.length; i++) {
    const r = i === 0 ? "  -" : pad(res.rates[i - 1].toFixed(3), 7);
    console.log(`  ${pad(res.Ns[i], 4)} ${fmt(res.dxs[i])}  ${fmt(res.errs[i])}  ${r}  ${pad(res.steps[i], 5)}  ${pad(res.runtimes[i].toFixed(1), 8)}`);
  }
}

(function () {
  // 1D — all polynomial orders, p=4 included to verify order 5
  for (const p of [1, 2, 3, 4]) {
    const NList = p === 4 ? [8, 16, 32, 64] : [16, 32, 64, 128];
    const T = 0.1;
    const res_rk = runConvergence1D({ p, NList, scheme: "rk", T });
    const res_ad = runConvergence1D({ p, NList, scheme: "ader", T });
    report(`1D p=${p}  RK-DG (expected order ${p + 1})`, res_rk);
    report(`1D p=${p}  ADER-DG (expected order ${p + 1})`, res_ad);
  }

  // 2D — fewer to keep runtime reasonable
  for (const p of [1, 2]) {
    const NList = [8, 16, 32];
    const T = 0.05;
    const res_rk = runConvergence2D({ p, NList, scheme: "rk", T });
    const res_ad = runConvergence2D({ p, NList, scheme: "ader", T });
    report(`2D p=${p}  RK-DG`, res_rk);
    report(`2D p=${p}  ADER-DG`, res_ad);
  }
})();
