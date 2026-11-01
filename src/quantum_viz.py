"""
Server-side visualization builders for the real-time classification dashboard.

Every figure is rendered with matplotlib (Agg backend) and returned as a
base64 PNG string so the frontend needs no charting libraries or CDN access.
CSV helpers serialize the quantum encoding artefacts per classification.
"""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _dark_fig(*args, **kwargs):
    kwargs.setdefault("facecolor", "#0f1420")
    fig = plt.figure(*args, **kwargs)
    return fig


def _style_ax(ax):
    ax.set_facecolor("#0f1420")
    for s in ax.spines.values():
        s.set_color("#3a4668")
    ax.tick_params(colors="#9fb0d0", labelsize=8)
    ax.xaxis.label.set_color("#c6d2ec")
    ax.yaxis.label.set_color("#c6d2ec")
    ax.title.set_color("#e8eefc")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_mri_slices(nii_path: str | Path) -> str | None:
    """Axial / coronal / sagittal mid-slices of the uploaded scan."""
    try:
        import nibabel as nib
    except ImportError:
        return None
    img = nib.load(str(nii_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    # move to a canonical-ish display: largest dim last
    cx, cy, cz = (s // 2 for s in data.shape[:3])
    slices = [
        ("Axial", np.rot90(data[:, :, cz])),
        ("Coronal", np.rot90(data[:, cy, :])),
        ("Sagittal", np.rot90(data[cx, :, :])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="#0f1420")
    for ax, (name, sl) in zip(axes, slices):
        ax.imshow(sl, cmap="gray")
        ax.set_title(name, color="#e8eefc", fontsize=10)
        ax.axis("off")
    fig.suptitle(Path(nii_path).name, color="#9fb0d0", fontsize=9)
    return _fig_to_b64(fig)


def fig_roi_profile(roi_values, roi_names, top_k: int = 10) -> str:
    """Bar chart of all ROI means; top |value| regions highlighted."""
    vals = np.asarray(roi_values, dtype=float)
    order = np.argsort(np.abs(vals))[::-1]
    highlight = set(order[:top_k].tolist())
    colors = ["#e05cd8" if i in highlight else "#3d6fd8" for i in range(len(vals))]
    short = [n.replace("roi_", "").split("_", 1)[-1] for n in roi_names]

    fig, ax = plt.subplots(figsize=(11, 3.4), facecolor="#0f1420")
    ax.bar(range(len(vals)), vals, color=colors, width=0.8)
    ax.axhline(0, color="#3a4668", lw=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(short, rotation=90, fontsize=5.5)
    ax.set_ylabel("z-normalized mean")
    ax.set_title("Harvard-Oxford ROI Intensity Profile "
                 f"(top {top_k} highlighted)")
    _style_ax(ax)
    return _fig_to_b64(fig)


def fig_pca_and_angles(pca_components, qubit_angles, pca_variance_pct) -> str:
    """PCA projections (left) and resulting [0, pi] qubit angles (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor="#0f1420")
    pcs = np.asarray(pca_components)
    axes[0].bar([f"PC{i+1}" for i in range(len(pcs))], pcs,
                color="#41b6c4", width=0.55)
    axes[0].axhline(0, color="#3a4668", lw=0.8)
    axes[0].set_title("PCA Projection (variance: "
                      + ", ".join(f"{v:.0f}%" for v in pca_variance_pct) + ")")
    axes[0].set_ylabel("component value")
    _style_ax(axes[0])

    ang = np.asarray(qubit_angles)
    axes[1].bar([f"q{i}" for i in range(len(ang))], ang,
                color="#a06bff", width=0.55)
    axes[1].axhline(np.pi, color="#e05cd8", ls="--", lw=1, label="π bound")
    axes[1].set_ylim(0, np.pi * 1.12)
    for i, a in enumerate(ang):
        axes[1].text(i, a + 0.08, f"{a:.3f}", ha="center",
                     color="#e8eefc", fontsize=9)
    axes[1].set_title("Qubit Rotation Angles (rad, encoded in [0, π])")
    axes[1].legend(facecolor="#0f1420", labelcolor="#c6d2ec", fontsize=8)
    _style_ax(axes[1])
    return _fig_to_b64(fig)


def fig_statevector(basis_states, amplitudes_re, amplitudes_im) -> str:
    """Two-panel: basis-state probabilities + phases of |phi(x)>.

    ZZFeatureMap is a pure phase encoding (all |amplitudes| = 1/sqrt(16)),
    so the discriminative signature lives in the per-basis-state phases.
    """
    re = np.asarray(amplitudes_re)
    im = np.asarray(amplitudes_im)
    probs = re * re + im * im
    phases = np.arctan2(im, re)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), facecolor="#0f1420",
                             sharex=True)
    axes[0].bar(basis_states, probs, color="#4dd0a6", width=0.6)
    axes[0].set_ylabel("probability")
    axes[0].set_title("Encoded State |φ(x)⟩ — uniform magnitudes "
                      "(phase encoding); information lives in the phases ↓")
    axes[1].bar(basis_states, phases, color="#e0a55c", width=0.6)
    axes[1].axhline(0, color="#3a4668", lw=0.8)
    axes[1].set_ylabel("phase arg(a_k) [rad]")
    axes[1].set_xlabel("basis state |q3 q2 q1 q0⟩")
    axes[1].set_ylim(-np.pi * 1.15, np.pi * 1.15)
    for ax in axes:
        _style_ax(ax)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return _fig_to_b64(fig)


def fig_bloch(bloch_vectors) -> str:
    """Four per-qubit Bloch spheres from reduced density matrices."""
    from qiskit.visualization import plot_bloch_vector

    n = len(bloch_vectors)
    fig = plt.figure(figsize=(2.4 * n, 2.9), facecolor="#0f1420")
    for i, bv in enumerate(bloch_vectors):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_bloch_vector(bv, ax=ax)
        ax.set_title(f"q{i}", color="#e8eefc", fontsize=10, pad=-6)
    return _fig_to_b64(fig)


def fig_kernel_hist(kernel_row, y_train) -> str:
    """Fidelity distribution of the uploaded scan vs all training states."""
    k = np.asarray(kernel_row)
    y = np.asarray(y_train)
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="#0f1420")
    bins = np.linspace(0, 1, 50)
    ax.hist(k[y == 0], bins=bins, alpha=0.75, color="#3d6fd8",
            label="vs Control train states")
    ax.hist(k[y == 1], bins=bins, alpha=0.75, color="#e05cd8",
            label="vs Autism train states")
    ax.set_xlabel("kernel fidelity  K(x, x_train) = |⟨φ(x)|φ(x_train)⟩|²")
    ax.set_ylabel("count")
    ax.set_title(f"Quantum Kernel Similarity vs {len(k)} Training Subjects")
    ax.legend(facecolor="#0f1420", labelcolor="#c6d2ec", fontsize=8)
    _style_ax(ax)
    return _fig_to_b64(fig)


def fig_circuit(feature_map) -> str:
    """ZZFeatureMap circuit diagram."""
    fig = feature_map.draw(output="mpl", style="iqp")
    return _fig_to_b64(fig)


# --------------------------------------------------------------------------
# CSV builders
# --------------------------------------------------------------------------
def _csv_text(headers, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue()


def csv_roi_features(roi_names, roi_values) -> str:
    return _csv_text(
        ["roi_column", "value"],
        zip(roi_names, np.asarray(roi_values).tolist()),
    )


def csv_pca_components(pca_components) -> str:
    return _csv_text(
        ["component", "value"],
        ((f"PC{i+1}", v) for i, v in enumerate(pca_components)),
    )


def csv_qubit_angles(qubit_angles) -> str:
    rows = (
        (f"q{i}", float(a), float(a) * 180.0 / np.pi)
        for i, a in enumerate(qubit_angles)
    )
    return _csv_text(["qubit", "angle_rad", "angle_deg"], rows)


def csv_statevector(payload) -> str:
    qs = payload["quantum_state"]
    return _csv_text(
        ["basis_state", "amplitude_re", "amplitude_im", "probability"],
        zip(qs["basis_states"], qs["amplitudes_re"],
            qs["amplitudes_im"], qs["probabilities"]),
    )


def csv_kernel_similarities(kernel_row, train_subject_ids, y_train) -> str:
    return _csv_text(
        ["train_subject_id", "train_label", "kernel_fidelity"],
        ((int(sid), int(yl), float(k))
         for sid, yl, k in zip(train_subject_ids, y_train, kernel_row)),
    )
