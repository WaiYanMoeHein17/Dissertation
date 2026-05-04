// app.js — Browser glue for the ADER-DG vs RK-DG benchmark page.
//
// Wires the dg/ modules into a small interactive UI:
//   * "Run convergence" — runs both schemes on a list of meshes (1D or 2D)
//                         and renders log-log error vs dx with empirical
//                         convergence rates and runtimes.
//   * "Run runtime"     — single-config wall-time benchmark with bar chart.
//   * "Live demo"       — animates a smooth dam-break-like initial pulse on
//                         a chosen mesh using either scheme.
//
// Uses Chart.js (already shipped under sim/scripts/charts.umd.min.js).

import { runConvergence1D, runConvergence2D } from "./bench.js";
import { makeBasis } from "./basis.js";
import { Mesh1D, projectField1D, l2Error1D } from "./dg1d.js";
import { Mesh2D, projectField2D, l2Error2D } from "./dg2d.js";
import { makeRKDG1D, makeRKDG2D } from "./rkdg.js";
import { makeADERDG1D, makeADERDG2D } from "./aderdg.js";
import { makeMMS1D, makeMMS2D } from "./mms.js";
import { maxWavespeed1Dflat, maxWavespeed2D } from "./swe.js";

const PALETTE = {
  rk:   { stroke: "#1f77b4", fill: "rgba(31,119,180,0.15)" },
  ader: { stroke: "#d62728", fill: "rgba(214,39,40,0.15)" },
};

// =====================================================================
// Helpers
// =====================================================================
function $id(id) { return document.getElementById(id); }

function parseIntList(s) {
  return s.split(/[,\s]+/).map(t => t.trim()).filter(Boolean).map(t => parseInt(t, 10)).filter(Number.isFinite);
}

function fmt(x, digits = 3) {
  if (x === 0) return "0";
  return x.toExponential(digits);
}

function tableRow(...cells) {
  const tr = document.createElement("tr");
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = c;
    tr.appendChild(td);
  }
  return tr;
}

function tableHeader(...cells) {
  const tr = document.createElement("tr");
  for (const c of cells) {
    const th = document.createElement("th");
    th.textContent = c;
    tr.appendChild(th);
  }
  return tr;
}

// =====================================================================
// Convergence runner + chart
// =====================================================================

let convChart = null;
let runtimeChart = null;

function renderConvergenceTables(parent, p, dim, resRk, resAd) {
  parent.innerHTML = "";
  const heading = document.createElement("h3");
  heading.textContent = `Convergence at p = ${p}, ${dim}D, T = ${dim === 1 ? 0.1 : 0.05}`;
  parent.appendChild(heading);

  for (const [label, res] of [["RK-DG", resRk], ["ADER-DG", resAd]]) {
    const wrap = document.createElement("div");
    wrap.className = "dg-table-wrap";
    const sub = document.createElement("h4");
    sub.textContent = label;
    wrap.appendChild(sub);
    const tbl = document.createElement("table");
    tbl.className = "dg-table";
    tbl.appendChild(tableHeader("N", "dx", "L2 error", "rate", "steps", "wall (ms)"));
    for (let i = 0; i < res.Ns.length; i++) {
      const rate = i === 0 ? "—" : res.rates[i - 1].toFixed(3);
      tbl.appendChild(tableRow(
        res.Ns[i],
        fmt(res.dxs[i]),
        fmt(res.errs[i]),
        rate,
        res.steps[i],
        res.runtimes[i].toFixed(1)
      ));
    }
    wrap.appendChild(tbl);
    parent.appendChild(wrap);
  }
}

function renderConvergenceChart(canvas, p, resRk, resAd) {
  const Chart = window.Chart;
  if (!Chart) { canvas.parentElement.innerHTML = "<em>Chart.js failed to load.</em>"; return; }
  if (convChart) { convChart.destroy(); }
  // Reference slope lines: dx^{p+1}
  const dxs = resRk.dxs;
  const refOrder = p + 1;
  const refY = dxs.map(dx => Math.pow(dx, refOrder) * (resRk.errs[0] / Math.pow(dxs[0], refOrder)));
  const data = {
    datasets: [
      {
        label: "RK-DG",
        data: resRk.dxs.map((dx, i) => ({ x: dx, y: resRk.errs[i] })),
        borderColor: PALETTE.rk.stroke,
        backgroundColor: PALETTE.rk.fill,
        showLine: true,
        pointRadius: 4,
      },
      {
        label: "ADER-DG",
        data: resAd.dxs.map((dx, i) => ({ x: dx, y: resAd.errs[i] })),
        borderColor: PALETTE.ader.stroke,
        backgroundColor: PALETTE.ader.fill,
        showLine: true,
        pointRadius: 4,
      },
      {
        label: `O(dx^${refOrder}) reference`,
        data: dxs.map((dx, i) => ({ x: dx, y: refY[i] })),
        borderColor: "#888",
        borderDash: [4, 4],
        showLine: true,
        pointRadius: 0,
      },
    ],
  };
  const config = {
    type: "scatter",
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "logarithmic", title: { display: true, text: "dx" }, reverse: true },
        y: { type: "logarithmic", title: { display: true, text: "L2 error" } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  };
  convChart = new Chart(canvas, config);
}

async function runConvergenceUI() {
  const dim = parseInt($id("dg-dim").value, 10);
  const p   = parseInt($id("dg-p").value, 10);
  const NList = parseIntList($id("dg-N").value);
  const T   = parseFloat($id("dg-T").value);
  const CFL = parseFloat($id("dg-cfl").value);
  const status = $id("dg-status");
  status.textContent = `Running ${dim}D p=${p} on N = [${NList.join(", ")}], T=${T}, CFL=${CFL} ...`;
  // Yield to the event loop so the message paints before the heavy work.
  await new Promise(r => setTimeout(r, 0));

  const args = { p, NList, T, CFL };
  let resRk, resAd;
  try {
    if (dim === 1) {
      resRk = runConvergence1D({ ...args, scheme: "rk" });
      resAd = runConvergence1D({ ...args, scheme: "ader" });
    } else {
      resRk = runConvergence2D({ ...args, scheme: "rk" });
      resAd = runConvergence2D({ ...args, scheme: "ader" });
    }
  } catch (err) {
    status.textContent = "Error: " + err.message;
    console.error(err);
    return;
  }

  renderConvergenceTables($id("dg-tables"), p, dim, resRk, resAd);
  renderConvergenceChart($id("dg-conv-chart"), p, resRk, resAd);
  // Speedup summary
  const speedups = resRk.runtimes.map((r, i) => r / resAd.runtimes[i]);
  const avg = speedups.reduce((a, b) => a + b, 0) / speedups.length;
  status.innerHTML = `Done. Mean ADER-DG speedup over RK-DG: <b>${avg.toFixed(2)}×</b>`
    + ` (per N: ${speedups.map(s => s.toFixed(2)).join(", ")})`;
}

// =====================================================================
// Live demo: animate a smooth periodic pulse with the chosen scheme
// =====================================================================

let liveCancel = null;

function makeLiveIC1D(L) {
  const x0 = 0.5 * L, sigma = 0.05 * L;
  return function (x, t) {
    const h = 1.0 + 0.3 * Math.exp(-((x - x0) / sigma) ** 2);
    const u = 0.0;
    return [h, h * u];
  };
}
function makeLiveIC2D(L) {
  const x0 = 0.5 * L, y0 = 0.5 * L, sigma = 0.07 * L;
  return function (x, y, t) {
    const h = 1.0 + 0.3 * Math.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2));
    return [h, 0, 0];
  };
}

function drawField1D(canvas, mesh, q, hMin, hMax) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  const { Nx, n, M, xn } = mesh;
  for (let i = 0; i < Nx; i++) {
    for (let ix = 0; ix < n; ix++) {
      const x = xn[i * n + ix];
      const h = q[(i * n + ix) * M];
      const px = (x / mesh.L) * W;
      const py = H - ((h - hMin) / (hMax - hMin)) * H * 0.95 - H * 0.025;
      if (i === 0 && ix === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
  }
  ctx.stroke();
}

function drawField2D(canvas, mesh, q, hMin, hMax) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const { Nx, Ny, n, M } = mesh;
  const cellPxW = W / (Nx * n);
  const cellPxH = H / (Ny * n);
  const img = ctx.createImageData(W, H);
  const data = img.data;
  for (let iy = 0; iy < Ny; iy++) {
    for (let ix = 0; ix < Nx; ix++) {
      for (let jy = 0; jy < n; jy++) {
        for (let jx = 0; jx < n; jx++) {
          const idx = ((iy * Nx + ix) * n + jy) * n + jx;
          const h = q[idx * M];
          const t = (h - hMin) / (hMax - hMin);
          // Viridis-ish interpolation: (dark blue) -> (cyan) -> (yellow)
          const r = Math.round(255 * Math.max(0, Math.min(1, 2 * t - 0.7)));
          const g = Math.round(255 * Math.max(0, Math.min(1, 2 * t * (1 - t) * 4)));
          const b = Math.round(255 * Math.max(0, Math.min(1, 1.2 - 1.5 * t)));
          // Fill the (jx, jy) sub-cell of element (ix, iy)
          const px0 = Math.floor((ix * n + jx) * cellPxW);
          const px1 = Math.floor((ix * n + jx + 1) * cellPxW);
          const py0 = Math.floor((iy * n + jy) * cellPxH);
          const py1 = Math.floor((iy * n + jy + 1) * cellPxH);
          for (let py = py0; py < py1; py++) {
            for (let px = px0; px < px1; px++) {
              const off = (py * W + px) * 4;
              data[off]     = r;
              data[off + 1] = g;
              data[off + 2] = b;
              data[off + 3] = 255;
            }
          }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function startLive() {
  stopLive();
  const dim = parseInt($id("live-dim").value, 10);
  const p   = parseInt($id("live-p").value, 10);
  const N   = parseInt($id("live-N").value, 10);
  const scheme = $id("live-scheme").value;
  const canvas = $id("live-canvas");
  const status = $id("live-status");
  const basis = makeBasis(p);
  const L = 1.0;
  const g = 1.0;
  let stepper, mesh, q, drawer, hMin, hMax;
  if (dim === 1) {
    mesh = new Mesh1D(L, N, basis);
    q = projectField1D(mesh, makeLiveIC1D(L));
    stepper = scheme === "rk" ? makeRKDG1D(mesh, g, null) : makeADERDG1D(mesh, g, null);
    drawer = (cv) => drawField1D(cv, mesh, q, 0.7, 1.4);
    hMin = 0.7; hMax = 1.4;
  } else {
    mesh = new Mesh2D(L, L, N, N, basis);
    q = projectField2D(mesh, makeLiveIC2D(L));
    stepper = scheme === "rk" ? makeRKDG2D(mesh, g, null) : makeADERDG2D(mesh, g, null);
    drawer = (cv) => drawField2D(cv, mesh, q, 0.95, 1.3);
  }
  let t = 0;
  let frame = 0;
  let running = true;
  liveCancel = () => { running = false; };

  const t0 = performance.now();
  const stepsPerFrame = dim === 1 ? 4 : 1;
  function tick() {
    if (!running) return;
    for (let s = 0; s < stepsPerFrame; s++) {
      const amax = Math.max(1e-10, dim === 1 ? maxWavespeed1Dflat(q, mesh.M, g) : maxWavespeed2D(q, mesh.M, g));
      const dt = 0.25 * mesh.dx / ((2 * p + 1) * amax);
      t = stepper.step(q, dt, t);
    }
    drawer(canvas);
    frame++;
    if (frame % 30 === 0) {
      const fps = (frame * 1000) / (performance.now() - t0);
      status.textContent = `${stepper.name} — t = ${t.toFixed(3)} — ${fps.toFixed(0)} frames/s`;
    }
    requestAnimationFrame(tick);
  }
  tick();
}

function stopLive() {
  if (liveCancel) { liveCancel(); liveCancel = null; }
}

// =====================================================================
// Wire-up
// =====================================================================

export function initApp() {
  $id("dg-run").addEventListener("click", runConvergenceUI);
  $id("live-start").addEventListener("click", startLive);
  $id("live-stop").addEventListener("click", stopLive);
  // Sensible default mesh list depending on dimension.
  $id("dg-dim").addEventListener("change", () => {
    if ($id("dg-dim").value === "1") $id("dg-N").value = "16, 32, 64, 128";
    else $id("dg-N").value = "8, 16, 32";
  });
}
