\
# FOCR1 longitudinal hyperspectral analysis

Reproducible Python implementation of the analysis underlying:

**From Spectra to Signal: Temporal Detection of Fusarium Wilt in Banana with Block-Aware Machine Learning**

## Scope

This repository reproduces the paper analysis from the preserved curated dataset forward.
It covers final QC, the E5 Block 7 sensitivity decision, block-aware temporal inference,
leaf/position analysis, exploratory spectral-region identification, leave-one-block-out
machine learning, and QC sensitivity.

The original one-off analysis scripts were not preserved in the exported analysis packages.
This repository is therefore a **reconstructed reproducible implementation**, validated
against the frozen numerical results. It should not be described as a byte-for-byte copy
of the original transient code.

## Input

Copy the file below from `00. FOCR1_curated_data_package`:

```text
banana_spectral_master_clean.csv
```

to:

```text
data/banana_spectral_master_clean.csv
```

The curated input contains 2,880 expected measurement slots, including 2,873 observed
spectra and seven missing measurements. It also preserves traceable metadata/ID corrections,
three reconstructed split spectra, and the low-signal QC flag.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce all analyses

```bash
python run_pipeline.py
```

or run stages separately:

```bash
python src/00_validate_curated.py
python src/01_qc_final.py
python src/02_exploration.py
python src/03_e5_sensitivity.py
python src/04_temporal_separability.py
python src/05_leaf_position.py
python src/06_spectral_regions.py
python src/07_machine_learning.py
python src/08_qc_sensitivity.py
python src/09_validate_paper_results.py
```

## Fixed machine-learning configuration

No hyperparameter tuning or nested cross-validation was used. Hyperparameters are fixed
before leave-one-block-out evaluation; no held-out block is used for tuning.

| Model | Fixed configuration |
|---|---|
| Logistic regression | L2, `C=1.0`, `solver="liblinear"`, `class_weight="balanced"` |
| RBF-SVM | `C=1.0`, `gamma="scale"`, `class_weight="balanced"` |
| PLS-DA | 5 latent components; class threshold 0.5 |

All standardization parameters are estimated only on the training blocks in each fold.

These settings were recovered by reconstructing the preserved LOBO pipeline and matching
the archived out-of-fold prediction scores. In particular, the archived logistic scores
match `liblinear, C=1`, the SVM scores match `C=1, gamma="scale"`, and the PLS-DA scores
match five latent components.

## Frozen checks

`src/09_validate_paper_results.py` verifies the principal paper claims, including:

- 2,761 spectra after final QC;
- 2,703 spectra after excluding E5 Block 7;
- Foc R1-only E4 VIS mean difference ≈ +0.00441;
- exact block sign-flip `p = 0.0078125`;
- Holm-adjusted `p = 0.0390625`;
- E4 whole-plant 520–749 nm RBF-SVM balanced accuracy ≈ 0.703;
- E5 whole-plant full-spectrum logistic balanced accuracy ≈ 0.799.

Run unit checks with:

```bash
pytest -q
```

## Statistical design

The independently randomized unit is the plant, not an individual spectrum. Repeated
leaf/position spectra are therefore never treated as independent plants.

Temporal confirmatory inference:
- plant-level averaging within evaluation;
- treatment-control contrasts computed within randomized blocks;
- exact two-sided sign-flip tests;
- Holm correction across five predefined regions;
- exact maxT family-wise correction for bandwise inference.

Predictive evaluation:
- leave one complete experimental block out at each fold;
- preprocessing/scaling fitted on training blocks only;
- predictions pooled across held-out blocks.

## Important terminology

The labels represent **experimental inoculation status**, not independently confirmed
infection status. Code and outputs use `FOC_R1_only`, `Radopholus_only`,
`Coinoculation`, and `Control` as treatment-group labels.

## Data availability

The repository does not make a claim about public availability of the biological data.
Update this section only after the authors/data owners decide whether the dataset can be
public, available on request, or restricted.
