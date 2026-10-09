\
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from common import (
    SPECTRAL_COLS, REGIONS, TASKS, add_analysis_labels, plant_level_mean,
    spectral_columns, exact_signflip_p, holm
)

def maxT_bandwise(block_diffs):
    # block_diffs: n_blocks x n_wavelengths
    D=np.asarray(block_diffs,float)
    observed=np.abs(D.mean(axis=0))
    null_max=[]
    for s in product((-1.0,1.0),repeat=D.shape[0]):
        null_max.append(np.abs(D*np.asarray(s)[:,None]).mean(axis=0).max())
    null_max=np.asarray(null_max)
    p=np.array([(null_max >= x-1e-15).mean() for x in observed])
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/03_e5_sensitivity/banana_spectral_analysis_temporal_primary.csv")
    ap.add_argument("--outdir",default="results/04_temporal_separability")
    args=ap.parse_args()
    d=add_analysis_labels(pd.read_csv(args.input,low_memory=False))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    plant,meta=plant_level_mean(d)
    meta.to_csv(out/"plant_level_mean_spectra_metadata.csv",index=False)

    region_rows=[]; block_rows=[]; max_rows=[]
    for e in sorted(plant.evaluation.unique()):
      for g in TASKS:
        pvals=[]; temp=[]
        for region,(lo,hi) in REGIONS.items():
          cols=spectral_columns(lo,hi); vals=[]; blocks=[]
          for b in sorted(plant.block.unique()):
            a=plant[(plant.evaluation==e)&(plant.block==b)&(plant.group==g)]
            c=plant[(plant.evaluation==e)&(plant.block==b)&(plant.group=="Control")]
            if len(a) and len(c):
              diff=float(a[cols].to_numpy().mean()-c[cols].to_numpy().mean())
              vals.append(diff); blocks.append(int(b))
              block_rows.append([e,g,b,region,diff])
          p=exact_signflip_p(vals); pvals.append(p)
          temp.append([e,g,len(vals),",".join(map(str,blocks)),region,
                       np.mean(vals),np.std(vals,ddof=1),p])
        hp=holm(pvals)
        for row,h in zip(temp,hp): region_rows.append(row+[h])

        # Exact maxT across all 2,151 wavelengths.
        bd=[]
        for b in sorted(plant.block.unique()):
          a=plant[(plant.evaluation==e)&(plant.block==b)&(plant.group==g)]
          c=plant[(plant.evaluation==e)&(plant.block==b)&(plant.group=="Control")]
          if len(a) and len(c):
            bd.append(a[SPECTRAL_COLS].mean().to_numpy()-c[SPECTRAL_COLS].mean().to_numpy())
        if bd:
          D=np.vstack(bd); meanD=D.mean(axis=0); padj=maxT_bandwise(D)
          max_rows.append({
              "evaluation":e,"group":g,"n_blocks":D.shape[0],
              "rms_mean_difference":float(np.sqrt(np.mean(meanD**2))),
              "n_bands_maxT_p_lt_0_05":int((padj<0.05).sum()),
              "wavelength_max_abs_mean_difference":int(350+np.argmax(np.abs(meanD))),
              "signed_difference_at_max":float(meanD[np.argmax(np.abs(meanD))]),
              "min_maxT_adjusted_p":float(padj.min())
          })

    pd.DataFrame(region_rows,columns=[
      "evaluation","group","n_blocks","blocks","spectral_region","mean_signed_difference",
      "sd_block_difference","exact_signflip_p","holm_p_5_regions"
    ]).to_csv(out/"temporal_region_tests.csv",index=False)
    pd.DataFrame(block_rows,columns=[
      "evaluation","group","block","spectral_region","mean_signed_difference"
    ]).to_csv(out/"temporal_block_region_differences.csv",index=False)
    pd.DataFrame(max_rows).to_csv(out/"temporal_bandwise_maxT_summary.csv",index=False)
    print("Temporal separability complete.")

if __name__=="__main__":
    main()
