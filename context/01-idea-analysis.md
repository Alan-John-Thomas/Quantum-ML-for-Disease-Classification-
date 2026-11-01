# Idea Analysis — QML for Disease Classification

> Brainstormed with Devin on 2026-11-01. Shortlist ranked against the judging rubric.

## Comparison matrix

| Idea | Data readiness | Innovation | Feasibility | Quantum story | Demo appeal |
|---|---|---|---|---|---|
| **TB chest X-ray + transfer learning** | ★★★★ (~662 imgs) | ★★★★ | ★★★★ | ★★★★ | ★★★★★ |
| **Autism ABIDE MRI** | ★★★ (need autism cohort) | ★★★★ | ★★★ | ★★★★ | ★★★ |
| **Gene expression cancer subtype** | ★★★★★ (tiny, instant) | ★★★★ | ★★★★★ | ★★★★★ | ★★★ |
| **EEG seizure classification** | ★★★★ | ★★★ | ★★★★ | ★★★ | ★★★ |
| **Molecular screening (MoleculeNet)** | ★★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★ |
| **Federated quantum learning** | ★★★★ | ★★★★★ | ★★ | ★★★★ | ★★★★ |

---

## Option A — Autism classification from ABIDE MRI

**Data already in hand**: ~90 skull-stripped structural MRI scans (`.nii.gz`) in
`../control/`, subject IDs 50030–50346 → NYU site of ABIDE I. Missing: the
autism cohort scans + `Phenotypic_V1_0b_preprocessed1.csv` (has `DX_GROUP`:
1=autism, 2=control). Both are public downloads.

**Pipeline**:

```
.nii.gz scans → ROI features (nilearn + Harvard-Oxford atlas, mean intensity
                per region ~50–100 features)
             → StandardScaler → PCA (4–8 components = 4–8 qubits)
             → ZZFeatureMap → FidelityQuantumKernel + SVC   (or VQC)
             → metrics vs classical SVM/RF on identical features
```

- **Classical→quantum mapping**: each PCA component = one rotation angle in the
  feature map; 1 feature = 1 qubit. Easy to justify and draw.
- **Qiskit pieces**: `qiskit-machine-learning` (`ZZFeatureMap`,
  `FidelityQuantumKernel`, `QSVC`, `VQC`), `qiskit-aer` (`AerSimulator`),
  optionally one real IBM backend run for the demo.
- **Classical baseline** (rubric requires): SVM-RBF, RandomForest, LogReg on the
  same PCA features.
- **Why it scores**: ABIDE is THE public benchmark for autism MRI; published
  classical accuracy is ~60–70%, so results will be credible. SDG 3 (+ SDG 10:
  diagnostic access in low-resource settings).
- **Risks**: MRI feature extraction is the heavy step (~20 lines with
  `nilearn.NiftiLabelsMasker`); need to download the autism cohort early —
  that's the long pole. Modest accuracy is expected and honest.

## Option B — TB chest X-ray + quantum transfer learning (best all-around demo)

- **Hook**: TB kills ~1.3M people/year; SDG target 3.3 literally names
  tuberculosis — one citation carries the whole SDG-relevance score.
- **Data**: Shenzhen + Montgomery CXR datasets (~800 images total, public).
- **Pipeline**: frozen pretrained ResNet-18 → 512-d bottleneck features →
  PCA to 4–6 → quantum kernel/VQC head ("dressed quantum circuit").
- **Demo**: show a chest X-ray → "TB: 94%". Most visceral demo of all options.
- **Risk**: small dataset → high variance; CNN + quantum = two moving parts.
  Mitigate by freezing the backbone (features extracted once, ~10 min).

## Option C — Gene expression cancer subtype (best quantum rationale)

- **Hook**: p ≫ n (thousands of genes, tens of samples) — the one regime where
  quantum kernels have a plausible edge (Havlíček et al. feature-space argument).
- **Data**: Golub leukemia (72 samples, ~7k genes) or SRBCT (83 samples,
  4-class multiclass). Downloads in seconds.
- **Bonus**: multiclass VQC is rarer than binary → innovation points.
- **Risk**: no visual demo — table + confusion matrix only. "Quantum advantage
  plausible" framing must be worded carefully to stay honest.

## Option D — Molecular/drug screening (MoleculeNet BBBP or HIV)

- **Hook**: molecules ARE quantum systems → "quantum computing classifying
  quantum objects" is the most thematically justified mapping possible.
- **Data**: BBBP (~2k molecules, blood-brain barrier penetration → ties to
  neurological disease). RDKit → Morgan fingerprints → PCA → quantum kernel.
- **Risk**: drug screening is slightly off-brief vs "disease classification" —
  needs framing ("prioritizing compounds for brain-disease treatment").

## Option E — Federated quantum learning (privacy angle)

- **Hook**: hospitals can't share patient data → simulate 3 "hospital nodes"
  each training a VQC on a shard, federated-average parameters.
- **Risk**: highest engineering risk — building federated averaging AND a
  quantum model in hackathon time. Only with a strong 3–4 person team.

## Option F — EEG seizure detection (Bonn dataset)

- 500 segments, spectral-band features → quantum classifier.
- **Risk**: classical methods already hit ~98% on Bonn — no room for a quantum
  contribution story. "Too easy" trap. Skip unless paired with harder data.

---

## Verdict (Devin's take)

- **If demo day matters most** → TB X-ray transfer learning (best SDG hook +
  visceral demo + lower data risk).
- **If leveraging data already downloaded** → Autism ABIDE (top-3 idea; 90
  control scans already local; needs nilearn confidence).
- **Safest MVP / strongest quantum story** → Gene expression subtype.

## Rubric-maximizing tactics (apply to ANY option)

1. **Feature-map ablation**: PauliFeatureMap vs ZZFeatureMap (linear vs full
   entanglement) vs amplitude encoding → one table, big Innovation points.
2. **Quantum kernel vs RBF kernel** on the SAME SVC → cleanest honest
   comparison. If quantum doesn't win, say so (rubric explicitly doesn't expect
   quantum advantage).
3. **Noise study**: rerun kernel on AerSimulator with an IBM noise model →
   accuracy degradation curve. Cheap, looks mature.
4. **Reproducibility**: pinned requirements.txt, fixed seeds, one-command run,
   README with dataset-access steps. Don't ship ABIDE `.nii.gz` files (data-use
   agreement) — link + subject ID list instead; also satisfies the no-PII rule.
5. **Demo artifacts**: `circuit.draw()` diagrams, side-by-side quantum vs
   classical confusion matrices, 2-PCA-component decision-boundary plot,
   60-second screen recording.
