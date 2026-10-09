\
from __future__ import annotations
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, roc_auc_score, accuracy_score,
    precision_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from statsmodels.stats.multitest import multipletests

WAVELENGTHS = np.arange(350, 2501)
SPECTRAL_COLS = [f"WL_{w}" for w in WAVELENGTHS]
REGIONS = {
    "VIS_400_700": (400, 700),
    "RedEdge_680_750": (680, 750),
    "NIR_751_1300": (751, 1300),
    "SWIR1_1301_1900": (1301, 1900),
    "SWIR2_1901_2500": (1901, 2500),
}
TASKS = ["FOC_R1_only", "Radopholus_only", "Coinoculation"]

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_csv(path):
    return pd.read_csv(path, low_memory=False)

def add_analysis_labels(df):
    d = df.copy()
    d["group"] = np.select(
        [
            (d["foc_r1_inoculated"] == 0) & (d["radopholus_inoculated"] == 0),
            (d["foc_r1_inoculated"] == 1) & (d["radopholus_inoculated"] == 0),
            (d["foc_r1_inoculated"] == 0) & (d["radopholus_inoculated"] == 1),
            (d["foc_r1_inoculated"] == 1) & (d["radopholus_inoculated"] == 1),
        ],
        ["Control", "FOC_R1_only", "Radopholus_only", "Coinoculation"],
        default="Other",
    )
    d["configuration"] = (
        "H" + d["leaf_number"].astype("Int64").astype(str) + "_" + d["position"].astype(str)
    )
    return d

def spectral_columns(lo=350, hi=2500):
    return [f"WL_{w}" for w in range(lo, hi + 1)]

def exact_signflip_p(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    obs = abs(x.mean())
    null = []
    for signs in product((-1.0, 1.0), repeat=len(x)):
        null.append(abs(np.mean(x * np.asarray(signs))))
    return float(np.mean(np.asarray(null) >= obs - 1e-15))

def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out

def plant_level_mean(df):
    d = add_analysis_labels(df)
    keys = ["plant_id", "evaluation", "block", "treatment", "group"]
    meta = d.groupby(keys, as_index=False).size().rename(columns={"size":"n_leaf_position_spectra"})
    spec = d.groupby(keys, as_index=False)[SPECTRAL_COLS].mean()
    return spec, meta

def block_region_differences(plant_df, evaluation, group, lo, hi):
    cols = spectral_columns(lo, hi)
    rows = []
    blocks = sorted(plant_df["block"].dropna().unique())
    for block in blocks:
        pos = plant_df[
            (plant_df["evaluation"] == evaluation) &
            (plant_df["block"] == block) &
            (plant_df["group"] == group)
        ]
        ctl = plant_df[
            (plant_df["evaluation"] == evaluation) &
            (plant_df["block"] == block) &
            (plant_df["group"] == "Control")
        ]
        if len(pos) and len(ctl):
            diff = pos[cols].to_numpy().mean() - ctl[cols].to_numpy().mean()
            rows.append((block, float(diff)))
    return rows

def transform_matrix(X, feature_set):
    X = np.asarray(X, dtype=float)
    if feature_set == "Raw_full_350_2500":
        return X
    if feature_set == "Raw_VIS_rededge_520_749":
        i0, i1 = 520-350, 749-350+1
        return X[:, i0:i1]
    if feature_set == "D1_rededge_650_749":
        d1 = savgol_filter(X, window_length=21, polyorder=2, deriv=1,
                           delta=1.0, axis=1, mode="interp")
        i0, i1 = 650-350, 749-350+1
        return d1[:, i0:i1]
    raise ValueError(feature_set)

def make_model(name):
    if name == "LogisticRegression":
        return LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=5000
        )
    if name == "SVM_RBF":
        return SVC(C=1.0, gamma="scale", kernel="rbf",
                   class_weight="balanced")
    if name == "PLS_DA":
        return PLSRegression(n_components=5, scale=False, max_iter=500)
    raise ValueError(name)

def fit_predict_lobo(X, y, blocks, sample_ids, model_name):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    blocks = np.asarray(blocks)
    sample_ids = np.asarray(sample_ids)
    records = []
    for b in sorted(np.unique(blocks)):
        train = blocks != b
        test = blocks == b
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])
        model = make_model(model_name)
        model.fit(Xtr, y[train])
        if model_name == "LogisticRegression":
            score = model.decision_function(Xte)
            pred = (score >= 0).astype(int)
        elif model_name == "SVM_RBF":
            score = model.decision_function(Xte)
            pred = (score >= 0).astype(int)
        else:
            score = model.predict(Xte).ravel()
            pred = (score >= 0.5).astype(int)
        for sid, yy, pp, ss in zip(sample_ids[test], y[test], pred, score):
            records.append((sid, b, int(yy), int(pp), float(ss)))
    return pd.DataFrame(records, columns=["sample_id","block","true_label","pred_label","score"])

def pooled_metrics(pred):
    y = pred["true_label"].to_numpy()
    p = pred["pred_label"].to_numpy()
    s = pred["score"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if tp+fn else np.nan
    spec = tn/(tn+fp) if tn+fp else np.nan
    return {
        "balanced_accuracy": (sens+spec)/2,
        "sensitivity": sens,
        "specificity": spec,
        "roc_auc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }
