\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import SPECTRAL_COLS, add_analysis_labels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/01_qc_final/banana_spectral_analysis_primary.csv")
    ap.add_argument("--outdir",default="results/03_e5_sensitivity")
    args=ap.parse_args()
    d=add_analysis_labels(pd.read_csv(args.input,low_memory=False))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    e5=d[d.evaluation==5].copy()
    global_centroid=e5[SPECTRAL_COLS].mean().to_numpy()
    rows=[]
    for b,x in e5.groupby("block"):
        centroid=x[SPECTRAL_COLS].mean().to_numpy()
        others=e5[e5.block!=b][SPECTRAL_COLS].mean().to_numpy()
        rows.append({
            "block":b,"n":len(x),
            "centroid_rms_vs_other_blocks":float(np.sqrt(np.mean((centroid-others)**2))),
            "median_R1850":float(x["WL_1850"].median())
        })
    diag=pd.DataFrame(rows).sort_values("centroid_rms_vs_other_blocks",ascending=False)
    diag.to_csv(out/"E5_block_diagnostics.csv",index=False)

    # The diagnostic identifies E5 Block 7 as the acquisition/calibration anomaly.
    excluded=d[(d.evaluation==5)&(d.block==7)].copy()
    primary=d[~((d.evaluation==5)&(d.block==7))].copy()
    excluded.to_csv(out/"E5_block7_excluded_records.csv",index=False)
    primary.to_csv(out/"banana_spectral_analysis_temporal_primary.csv",index=False)
    print({"input_n":len(d),"E5B7_excluded":len(excluded),"temporal_primary_n":len(primary)})

if __name__=="__main__":
    main()
