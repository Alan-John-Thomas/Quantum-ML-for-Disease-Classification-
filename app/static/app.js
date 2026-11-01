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
  const q = d.quantum, c = d.classical;
  const cls = q.prediction === 1 ? "autism" : "control";

  /* verdict */
  const lbl = $("pred-label");
  lbl.textContent = q.label;
  lbl.className = cls;
  retriggerInkIn(lbl);

  $("verdict-kicker").textContent = d.scan_file
    ? `scan · ${d.scan_file}`
    : `subject · #${d.subject?.subject_id ?? "—"} · ${d.subject?.split ?? ""} split`;

  $("pred-sub").innerHTML =
    `Quantum kernel decision score <b>${q.decision_score.toFixed(3)}</b>. ` +
    (d.agree
      ? `The classical RBF-SVM concurs (<b>${(c.probabilities[q.prediction === 1 ? "autism" : "control"] * 100).toFixed(1)}%</b>).`
      : `The classical RBF-SVM dissents — it predicts <b>${c.label}</b> (<b>${(c.probabilities.autism * 100).toFixed(1)}% autism</b>).`);

  /* spec row */
  const t = d.timings;
  const specs = [
    [q.confidence.toFixed(3), "", "qsvm confidence"],
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

boot();
