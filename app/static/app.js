/* NeuroQ-ASD real-time classifier dashboard — vanilla JS, no deps. */

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
      `<b>${info.n_qubits} qubits</b>`,
      `entanglement <b>${info.entanglement}</b>`,
      `depth <b>${info.circuit_metrics.depth}</b>`,
      `CNOTs <b>${info.circuit_metrics.cx_count}</b>`,
      `PCA var <b>${info.pca_total_variance_pct}%</b>`,
      `train <b>${info.n_train}</b> / test <b>${info.n_test}</b>`,
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
    `<option value="${s.subject_id}">#${s.subject_id} · ${s.label} · ${s.split}</option>`
  ).join("");
  if (rows.length) sel.selectedIndex = 0;
}

/* ---------- classification ---------- */
async function runClassify(fetchPromise, busyMsg) {
  status(`<span class="spin"></span>${busyMsg}`);
  const t0 = performance.now();
  try {
    const res = await fetchPromise;
    const data = await res.json();
    if (!res.ok) { status("Error: " + (data.error || res.statusText), true); return; }
    renderResult(data, (performance.now() - t0).toFixed(0));
    status(`Done in ${data.timings.total_ms ?? "?"} ms server-side.`);
  } catch (e) {
    status("Request failed: " + e, true);
  }
}

function renderResult(d, clientMs) {
  const q = d.quantum, c = d.classical;
  const cls = q.prediction === 1 ? "autism" : "control";
  const banner = $("banner");
  banner.className = "banner " + cls;
  $("pred-label").textContent = q.label;
  $("pred-sub").textContent =
    `Quantum Kernel SVM · decision score ${q.decision_score.toFixed(3)} · ` +
    (d.agree ? "classical RBF-SVM agrees" : "classical RBF-SVM disagrees → " + c.label);

  const t = d.timings;
  $("banner-stats").innerHTML = [
    [q.confidence.toFixed(3), "QSVM confidence"],
    [(c.probabilities.autism * 100).toFixed(1) + "%", "RBF autism prob"],
    [(t.quantum_infer_ms ?? "?") + " ms", "quantum infer"],
    [(t.roi_extraction_ms ?? t.encode_ms ?? "?") + " ms", t.roi_extraction_ms ? "ROI extract" : "encode"],
    [(t.total_ms ?? clientMs) + " ms", "total"],
  ].map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

  // true-label reveal for dataset subjects
  const truth = $("subject-truth");
  if (d.subject && $("reveal-label").checked) {
    const s = d.subject;
    const hit = (s.true_dx_group === 1) === (q.prediction === 1);
    truth.classList.remove("hidden");
    truth.innerHTML =
      `Subject <b>#${s.subject_id}</b> (${s.split} split) — true diagnosis: <b>${s.true_label}</b> · ` +
      `<span class="${hit ? "hit" : "miss"}">${hit ? "✓ correct" : "✗ misclassified"}</span>`;
  } else {
    truth.classList.add("hidden");
  }

  // csv links
  CsvUrls = d.csv_urls || {};
  $("csv-bar").classList.remove("hidden");
  document.querySelectorAll("[data-csv]").forEach(a => {
    a.setAttribute("href", CsvUrls[a.dataset.csv] || "#");
    a.setAttribute("download", "");
  });

  // figures
  $("fig-grid").classList.remove("hidden");
  for (const [key, b64] of Object.entries(d.figures || {})) {
    const img = $("fig-" + key);
    if (img && b64) img.src = "data:image/png;base64," + b64;
    if (key === "mri_slices" && b64) $("card-mri").classList.remove("hidden");
  }
  if (!d.figures?.mri_slices) $("card-mri").classList.add("hidden");

  // angle table
  const tb = document.querySelector("#angle-table tbody");
  tb.innerHTML = d.encoding.qubit_angles.map((a, i) =>
    `<tr><td>q${i}</td><td>${d.encoding.pca_components[i].toFixed(4)}</td>` +
    `<td>${a.toFixed(4)}</td><td>${d.encoding.angles_deg[i].toFixed(1)}°</td></tr>`
  ).join("");
}

/* ---------- events ---------- */
$("subject-filter").addEventListener("input", e => renderSubjectList(e.target.value));

$("btn-classify-subject").addEventListener("click", () => {
  const sid = $("subject-select").value;
  if (!sid) { status("Pick a subject first.", true); return; }
  runClassify(
    fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: Number(sid) }),
    }),
    `Encoding subject #${sid} into Hilbert space…`
  );
});

const dz = $("dropzone");
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", e => {
  e.preventDefault(); dz.classList.remove("drag");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
$("btn-browse").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", e => {
  if (e.target.files.length) setFile(e.target.files[0]);
});

function setFile(f) {
  pendingFile = f;
  $("file-name").textContent = `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`;
  $("btn-classify-scan").disabled = false;
}

$("btn-classify-scan").addEventListener("click", () => {
  if (!pendingFile) return;
  const fd = new FormData();
  fd.append("file", pendingFile);
  runClassify(
    fetch("/api/classify/scan", { method: "POST", body: fd }),
    `Extracting 69 ROIs from ${pendingFile.name} and encoding…`
  );
});

boot();
