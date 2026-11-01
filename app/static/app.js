/* NeuroQ-ASD — dashboard logic (vanilla JS, no dependencies). */

const $ = (id) => document.getElementById(id);
const status = (msg, isErr = false) => {
  const el = $("status");
  el.className = "status" + (isErr ? " err" : "");
  el.innerHTML = msg;
};

let SUBJECTS = [];
let CsvUrls = {};
let pendingFile = null;

/* ---------- boot ---------- */
async function boot() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("model-chips").innerHTML = [
      `<b>${info.n_qubits}q</b> zz featuremap`,
      `depth <b>${info.circuit_metrics.depth}</b>`,
      `<b>${info.circuit_metrics.cx_count}</b> cnots`,
      `pca var <b>${info.pca_total_variance_pct}%</b>`,
      `<b>${info.n_train}</b> train · <b>${info.n_test}</b> test`,
    ].map(c => `<span class="chip">${c}</span>`).join("");

    SUBJECTS = await (await fetch("/api/subjects")).json();
    renderSubjectList("");
  } catch (e) {
    status("Backend not reachable: " + e, true);
  }
}

function renderSubjectList(filter) {
  const sel = $("subject-select");
  const q = filter.toLowerCase();
  const rows = SUBJECTS.filter(s =>
    String(s.subject_id).includes(q) || s.label.toLowerCase().includes(q) ||
    s.split.includes(q));
  sel.innerHTML = rows.slice(0, 400).map(s =>
    `<option value="${s.subject_id}">#${s.subject_id} — ${s.label} · ${s.split}</option>`
  ).join("");
  if (rows.length) sel.selectedIndex = 0;
}

/* ---------- classify ---------- */
async function runClassify(fetchPromise, busyMsg) {
  status(`<span class="spin"></span>${busyMsg}`);
  const btn = $("btn-classify");
  btn.disabled = true;
  try {
    const res = await fetchPromise;
    const data = await res.json();
    if (!res.ok) { status("Error — " + (data.error || res.statusText), true); return; }
    renderResult(data);
    status(`pipeline latency ${data.timings.total_ms ?? "?"} ms`);
  } catch (e) {
    status("Request failed: " + e, true);
  } finally {
    btn.disabled = false;
  }
}

function retriggerInkIn(el) {
  el.classList.remove("ink-in");
  void el.offsetWidth;          /* restart animation */
  el.classList.add("ink-in");
}

function renderResult(d) {
  const q = d.quantum, c = d.classical, v = d.vqc;
  const cls = q.prediction === 1 ? "autism" : "control";

  /* verdict */
  const lbl = $("pred-label");
  lbl.textContent = q.label;
  lbl.className = cls;
  retriggerInkIn(lbl);

  $("verdict-kicker").textContent = d.scan_file
    ? `scan · ${d.scan_file}`
    : `subject · #${d.subject?.subject_id ?? "—"} · ${d.subject?.split ?? ""} split`;

  /* consensus line */
  const names = { qsvm: "QSVM", rbf: "RBF-SVM", vqc: "VQC" };
  const dissent = d.consensus
    ? Object.entries(d.consensus.votes)
        .filter(([k, val]) => val !== q.prediction)
        .map(([k]) => names[k])
    : (c.prediction !== q.prediction ? ["RBF-SVM"] : []);
  const concur = v ? `${(v.probabilities.autism * 100).toFixed(1)}%` : "—";
  $("pred-sub").innerHTML =
    `Quantum kernel decision score <b>${q.decision_score.toFixed(3)}</b>. ` +
    (dissent.length === 0
      ? `All three models concur — QSVM, VQC (<b>${concur} autism</b>), and RBF-SVM (<b>${(c.probabilities.autism * 100).toFixed(1)}% autism</b>).`
      : `<b>${d.consensus?.n_agree ?? "?"}/${d.consensus?.n_models ?? "?"}</b> consensus — ` +
        `${dissent.join(" and ")} dissent${dissent.length > 1 ? "" : "s"}.`);

  /* spec row */
  const t = d.timings;
  const specs = [
    [q.confidence.toFixed(3), "", "qsvm confidence"],
    [(v?.probabilities.autism * 100).toFixed(1), "%", "vqc autism prob"],
    [(c.probabilities.autism * 100).toFixed(1), "%", "rbf autism prob"],
    [t.quantum_infer_ms ?? "—", "ms", "quantum infer"],
    [t.roi_extraction_ms ?? t.encode_ms ?? "—", "ms",
     t.roi_extraction_ms ? "roi extraction" : "encoding"],
    [t.total_ms ?? "—", "ms", "total"],
  ];
  $("spec-row").innerHTML = specs.map(([v, u, k]) =>
    `<div class="spec"><div class="v">${v}<span class="u">${u}</span></div><div class="k">${k}</div></div>`
  ).join("");
  $("spec-row").classList.remove("hidden");

  /* ground truth (subjects only) */
  const truth = $("subject-truth");
  if (d.subject && $("reveal-label").checked) {
    const s = d.subject;
    const hit = (s.true_dx_group === 1) === (q.prediction === 1);
    truth.classList.remove("hidden");
    truth.innerHTML =
      `ground truth — <b>${s.true_label}</b> <span class="${hit ? "hit" : "miss"}">${hit ? "· correct" : "· misclassified"}</span>`;
  } else {
    truth.classList.add("hidden");
  }

  /* exports */
  CsvUrls = d.csv_urls || {};
  $("csv-bar").classList.remove("hidden");
  document.querySelectorAll("[data-csv]").forEach(a => {
    a.setAttribute("href", CsvUrls[a.dataset.csv] || "#");
    a.setAttribute("download", "");
  });

  /* figures */
  $("fig-grid").classList.remove("hidden");
  for (const [key, b64] of Object.entries(d.figures || {})) {
    const img = $("fig-" + key);
    if (img && b64) img.src = "data:image/png;base64," + b64;
  }
  $("card-mri").classList.toggle("hidden", !d.figures?.mri_slices);

  /* encoding table */
  document.querySelector("#angle-table tbody").innerHTML =
    d.encoding.qubit_angles.map((a, i) =>
      `<tr><td>q${i}</td><td>${d.encoding.pca_components[i].toFixed(4)}</td>` +
      `<td>${a.toFixed(4)}</td><td>${d.encoding.angles_deg[i].toFixed(1)}°</td></tr>`
    ).join("");

  mirrorToPlayground(d);
  $("fig-grid").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- events ---------- */
$("subject-filter").addEventListener("input", e => renderSubjectList(e.target.value));
$("subject-select").addEventListener("change", () => {
  pendingFile = null;
  $("file-name").textContent = "";
  $("file-input").value = "";
});

$("btn-classify").addEventListener("click", () => {
  if (pendingFile) {
    const fd = new FormData();
    fd.append("file", pendingFile);
    runClassify(
      fetch("/api/classify/scan", { method: "POST", body: fd }),
      `extracting 69 rois from ${pendingFile.name}, encoding into hilbert space…`);
  } else {
    const sid = $("subject-select").value;
    if (!sid) { status("Pick a subject or drop a scan first.", true); return; }
    runClassify(
      fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject_id: Number(sid) }),
      }),
      `encoding subject #${sid} into hilbert space…`);
  }
});

const dz = $("dropzone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", e => {
  e.preventDefault(); dz.classList.remove("drag");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
dz.addEventListener("click", () => $("file-input").click());
dz.addEventListener("keydown", e => { if (e.key === "Enter") $("file-input").click(); });
$("file-input").addEventListener("change", e => {
  if (e.target.files.length) setFile(e.target.files[0]);
});

function setFile(f) {
  pendingFile = f;
  $("file-name").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} mb`;
}

/* ============================================================
   Quantum playground — live canvas viz of the encoded state
   ============================================================ */
const COL = { ink: "#1c1a15", muted: "#8a8375", hair: "#e3ded1",
              indigo: "#3b34b0", teal: "#177e6c", rose: "#bd3e63", amber: "#b8862e" };
let Y_TRAIN = null;
let exploreTimer = null;

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth;
  const h = +cv.getAttribute("height");
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

function drawBloch(vectors) {
  const [ctx, w, h] = fitCanvas($("cv-bloch"));
  ctx.clearRect(0, 0, w, h);
  const r = Math.min(h * 0.42, w / vectors.length * 0.36);
  const cell = w / vectors.length;
  vectors.forEach(([bx, by, bz], i) => {
    const cx = cell * i + cell / 2, cy = h / 2;
    ctx.strokeStyle = COL.hair; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(cx, cy, r, r * 0.28, 0, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.stroke();
    /* vector: x -> horizontal, z -> vertical (y shown as opacity hint) */
    const vx = cx + bx * r, vy = cy - bz * r;
    ctx.strokeStyle = COL.indigo; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(vx, vy); ctx.stroke();
    ctx.fillStyle = COL.indigo;
    ctx.beginPath(); ctx.arc(vx, vy, 3.4, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = COL.muted; ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`q${i}  (${bx.toFixed(2)}, ${by.toFixed(2)}, ${bz.toFixed(2)})`, cx, h - 4);
  });
}

function drawPhases(re, im) {
  const [ctx, w, h] = fitCanvas($("cv-phases"));
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2, n = re.length, bw = w / n * 0.62, gap = w / n;
  ctx.strokeStyle = COL.hair; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  for (let i = 0; i < n; i++) {
    const ph = Math.atan2(im[i], re[i]);         /* [-π, π] */
    const bh = (ph / Math.PI) * mid;
    ctx.fillStyle = ph >= 0 ? COL.amber : COL.indigo;
    ctx.fillRect(gap * i + (gap - bw) / 2, mid - Math.max(bh, 0), bw, Math.abs(bh));
    ctx.fillStyle = COL.muted; ctx.font = "8px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(i.toString(2).padStart(4, "0"), gap * i + gap / 2,
                 ph >= 0 ? mid - bh - 3 : mid - bh + 10);
  }
}

function drawKernel(row) {
  const [ctx, w, h] = fitCanvas($("cv-kernel"));
  ctx.clearRect(0, 0, w, h);
  if (!Y_TRAIN || !row) return;
  const B = 42, bins = new Float64Array(B), binsA = new Float64Array(B);
  let kMax = 0;
  row.forEach((k, i) => {
    const b = Math.min(B - 1, Math.floor(k * B));
    (Y_TRAIN[i] === 1 ? binsA : bins)[b]++;   /* 1=autism rose, 0=control teal */
    if (k > kMax) kMax = k;
  });
  const peak = Math.max(...bins, ...binsA, 1);
  const bw = w / B;
  for (let i = 0; i < B; i++) {
    const x = i * bw, hC = (bins[i] / peak) * (h - 14), hA = (binsA[i] / peak) * (h - 14);
    ctx.fillStyle = "rgba(23,126,108,.75)";
    ctx.fillRect(x, h - 10 - hC, bw * 0.9, hC);
    ctx.fillStyle = "rgba(189,62,99,.6)";
    ctx.fillRect(x, h - 10 - hA, bw * 0.9, hA);
  }
  ctx.strokeStyle = COL.hair;
  ctx.beginPath(); ctx.moveTo(0, h - 10); ctx.lineTo(w, h - 10); ctx.stroke();
  const mx = kMax * w;
  ctx.strokeStyle = COL.indigo; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, h - 10); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = COL.indigo; ctx.font = "9px 'IBM Plex Mono', monospace";
  ctx.textAlign = mx > w - 70 ? "right" : "left";
  ctx.fillText(`max ${kMax.toFixed(3)}`, mx + (mx > w - 70 ? -4 : 4), 10);
  ctx.fillStyle = COL.muted;
  ctx.fillText("control", 4, h - 1); ctx.textAlign = "right";
  ctx.fillText("autism", w - 4, h - 1);
}

function renderPlayground(d) {
  /* verdict + consensus */
  const v = $("pg-verdict");
  const cls = d.prediction === 1 ? "autism" : "control";
  v.className = "pg-verdict " + cls;
  v.innerHTML = `${d.label}<small>score ${d.decision_score.toFixed(3)} · ${d.consensus_n}/3 concur</small>`;

  /* model bars (autism probability) */
  const probs = { qsvm: d.confidence, vqc: d.vqc_autism, rbf: d.rbf_autism };
  for (const [k, p] of Object.entries(probs)) {
    $("pb-" + k).style.width = (p * 100).toFixed(1) + "%";
    $("pv-" + k).textContent = (p * 100).toFixed(0) + "%";
  }

  drawBloch(d.bloch_vectors);
  drawPhases(d.amplitudes_re, d.amplitudes_im);
  drawKernel(d.kernel_row);
}

function exploreNow(body) {
  return fetch("/api/explore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json()).then(renderPlayground)
    .catch(e => console.warn("explore failed", e));
}

function scheduleExplore(body) {
  clearTimeout(exploreTimer);
  exploreTimer = setTimeout(() => exploreNow(body), 110);
}

function currentAngles() {
  return [...document.querySelectorAll("input.angle")]
    .sort((a, b) => +a.dataset.q - +b.dataset.q)
    .map(s => +s.value);
}

function syncAngleUI(angles) {
  document.querySelectorAll("input.angle").forEach(s => {
    const a = angles[+s.dataset.q];
    s.value = a;
    s.style.setProperty("--fill", (a / Math.PI * 100) + "%");
    $("sv-q" + s.dataset.q).textContent = a.toFixed(3) + " rad · " +
      (a * 180 / Math.PI).toFixed(0) + "°";
  });
}

document.querySelectorAll("input.angle").forEach(s => {
  s.addEventListener("input", () => {
    s.style.setProperty("--fill", (+s.value / Math.PI * 100) + "%");
    $("sv-q" + s.dataset.q).textContent =
      (+s.value).toFixed(3) + " rad · " + (+s.value * 180 / Math.PI).toFixed(0) + "°";
    $("sv-morph").textContent = "manual";
    scheduleExplore({ angles: currentAngles() });
  });
});

$("sl-morph").addEventListener("input", e => {
  const t = +e.target.value / 100;
  e.target.style.setProperty("--fill", (t * 100) + "%");
  $("sv-morph").textContent =
    t === 0 ? "control prototype" : t === 1 ? "autism prototype" : `t = ${t.toFixed(2)}`;
  scheduleExplore({ morph: t });
});

/* after a real classification, mirror its angles into the playground */
function mirrorToPlayground(d) {
  if (!d.encoding) return;
  syncAngleUI(d.encoding.qubit_angles);
  exploreNow({ angles: d.encoding.qubit_angles });
}

async function bootPlayground() {
  const meta = await (await fetch("/api/kernel/meta")).json();
  Y_TRAIN = meta.y_train;
  syncAngleUI(meta.medoid_control);
  exploreNow({ morph: 0 });
}

window.addEventListener("resize", () => exploreNow({ angles: currentAngles() }));

boot();
bootPlayground();
