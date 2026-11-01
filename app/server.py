"""
Flask backend for the real-time Quantum MRI classifier dashboard.

Run from the repository root:
    python app/server.py            # serves http://127.0.0.1:5000
    python app/server.py --retrain  # rebuild models/pipeline.joblib first
    python app/server.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, jsonify, request, send_from_directory

from src import quantum_viz as qv
from src.inference import InferencePipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_SUFFIXES = {".nii", ".gz"}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

PIPELINE: InferencePipeline | None = None
JOBS: "OrderedDict[str, dict]" = OrderedDict()  # job_id -> classify payload
MAX_JOBS = 20
_CIRCUIT_B64: str | None = None


def get_pipeline() -> InferencePipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = InferencePipeline()
    return PIPELINE


def _store_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = payload
    while len(JOBS) > MAX_JOBS:
        JOBS.popitem(last=False)
    return job_id


def _attach_figures(payload: dict, nii_path: str | None = None) -> dict:
    """Render server-side figures and attach base64 PNGs + csv links."""
    global _CIRCUIT_B64
    p = get_pipeline()
    figs = {
        "roi_profile": qv.fig_roi_profile(
            payload["roi_values"], payload["roi_names"]),
        "pca_angles": qv.fig_pca_and_angles(
            payload["encoding"]["pca_components"],
            payload["encoding"]["qubit_angles"],
            [v * 100 for v in p.pca_variance]),
        "statevector": qv.fig_statevector(
            payload["quantum_state"]["basis_states"],
            payload["quantum_state"]["amplitudes_re"],
            payload["quantum_state"]["amplitudes_im"]),
        "bloch": qv.fig_bloch(payload["quantum_state"]["bloch_vectors"]),
        "kernel_hist": qv.fig_kernel_hist(payload["kernel_row"], p.y_train),
    }
    if nii_path is not None:
        figs["mri_slices"] = qv.fig_mri_slices(nii_path)
    if _CIRCUIT_B64 is None:
        _CIRCUIT_B64 = qv.fig_circuit(p.qsvm.feature_map)
    figs["circuit"] = _CIRCUIT_B64
    payload["figures"] = figs
    return payload


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "pipeline_ready": PIPELINE is not None})


@app.get("/api/info")
def info():
    return jsonify(get_pipeline().model_info())


@app.get("/api/subjects")
def subjects():
    return jsonify(get_pipeline().subjects_table())


@app.post("/api/classify")
def classify_subject():
    body = request.get_json(force=True) or {}
    sid = body.get("subject_id")
    if sid is None:
        return jsonify({"error": "subject_id required"}), 400
    t0 = time.perf_counter()
    try:
        payload = get_pipeline().classify_subject(int(sid))
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    payload["timings"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    job_id = _store_job(payload)
    payload["job_id"] = job_id
    payload["csv_urls"] = {
        k: f"/api/export/{job_id}/{k}.csv"
        for k in ("roi_features", "pca_components", "qubit_angles",
                  "statevector", "kernel_similarities")
    }
    return jsonify(_attach_figures(payload))


@app.post("/api/classify/scan")
def classify_scan():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "multipart field 'file' required"}), 400
    name = f.filename
    if not any(name.endswith(s) for s in UPLOAD_SUFFIXES):
        return jsonify({"error": "expected a .nii or .nii.gz file"}), 400

    tmp = Path(tempfile.mkdtemp()) / name
    f.save(tmp)
    t0 = time.perf_counter()
    try:
        payload = get_pipeline().classify_nii(tmp)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"scan processing failed: {e}"}), 422
    payload["timings"]["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    job_id = _store_job(payload)
    payload["job_id"] = job_id
    payload["csv_urls"] = {
        k: f"/api/export/{job_id}/{k}.csv"
        for k in ("roi_features", "pca_components", "qubit_angles",
                  "statevector", "kernel_similarities")
    }
    return jsonify(_attach_figures(payload, nii_path=tmp))


@app.get("/api/export/<job_id>/<kind>.csv")
def export_csv(job_id: str, kind: str):
    payload = JOBS.get(job_id)
    if payload is None:
        return jsonify({"error": "job not found (classify first)"}), 404
    p = get_pipeline()
    builders = {
        "roi_features": lambda: qv.csv_roi_features(
            payload["roi_names"], payload["roi_values"]),
        "pca_components": lambda: qv.csv_pca_components(
            payload["encoding"]["pca_components"]),
        "qubit_angles": lambda: qv.csv_qubit_angles(
            payload["encoding"]["qubit_angles"]),
        "statevector": lambda: qv.csv_statevector(payload),
        "kernel_similarities": lambda: qv.csv_kernel_similarities(
            payload["kernel_row"], p.train_subject_ids, p.y_train),
    }
    if kind not in builders:
        return jsonify({"error": f"unknown export '{kind}'"}), 404
    csv_text = builders[kind]()
    stem = payload.get("scan_file") or f"subject_{payload.get('subject', {}).get('subject_id', job_id)}"
    return Response(
        csv_text, mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename={stem}_{kind}.csv"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--retrain", "-r", action="store_true",
                    help="rebuild models/pipeline.joblib before serving")
    args, _ = ap.parse_known_args()
    global PIPELINE
    PIPELINE = InferencePipeline(retrain=args.retrain)  # warm up before serving
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
