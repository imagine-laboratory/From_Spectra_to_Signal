\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from common import SPECTRAL_COLS, add_analysis_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/01_qc_final/banana_spectral_analysis_primary.csv")
    ap.add_argument("--outdir", default="results/02_exploration")
    args = ap.parse_args()
    d = add_analysis_labels(pd.read_csv(args.input, low_memory=False))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Compact summaries used to diagnose evaluation/group/block structure.
    rows = []
    for (e,g), x in d.groupby(["evaluation","group"]):
        mean = x[SPECTRAL_COLS].mean().to_numpy()
        rows.append({"evaluation":e,"group":g,"n":len(x),
                     "mean_reflectance":float(np.nanmean(mean)),
                     "spectral_rms":float(np.sqrt(np.nanmean(mean**2)))})
    pd.DataFrame(rows).to_csv(out/"exploratory_group_summary.csv", index=False)

    # Treatment-control RMS contrast by evaluation.
    rows=[]
    for e in sorted(d.evaluation.unique()):
        ctl=d[(d.evaluation==e)&(d.group=="Control")][SPECTRAL_COLS].mean().to_numpy()
        for g in ["FOC_R1_only","Radopholus_only","Coinoculation"]:
            x=d[(d.evaluation==e)&(d.group==g)][SPECTRAL_COLS].mean().to_numpy()
            delta=x-ctl
            rows.append({"evaluation":e,"group":g,
                         "rms_difference_to_control":float(np.sqrt(np.mean(delta**2)))})
    pd.DataFrame(rows).to_csv(out/"exploratory_control_contrasts.csv",index=False)

    # Raw-reflectance PCA: mean centered, no variance scaling.
    X=d[SPECTRAL_COLS].to_numpy(float)
    pca=PCA(n_components=10, svd_solver="full")
    scores=pca.fit_transform(X)
    score_df=d[["sample_id","plant_id","evaluation","block","treatment","group","leaf_number","position"]].copy()
    for i in range(scores.shape[1]): score_df[f"PC{i+1}"]=scores[:,i]
    score_df.to_csv(out/"exploratory_pca_scores.csv",index=False)
    pd.DataFrame({"component":range(1,11),"explained_variance_ratio":pca.explained_variance_ratio_}).to_csv(
        out/"exploratory_pca_variance.csv",index=False)
    print("Phase 2 exploration complete.")

if __name__=="__main__":
    main()
