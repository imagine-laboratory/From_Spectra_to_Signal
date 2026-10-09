\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from common import SPECTRAL_COLS,WAVELENGTHS,TASKS,add_analysis_labels

TRANSFORMS=["raw","d1","d2"]

def transform(X,name):
    if name=="raw": return X
    if name=="d1": return savgol_filter(X,21,2,deriv=1,delta=1.0,axis=1,mode="interp")
    if name=="d2": return savgol_filter(X,21,2,deriv=2,delta=1.0,axis=1,mode="interp")
    raise ValueError(name)

def scenario_effect(d,e,g,cfg,tname):
    x=d[(d.evaluation==e)&(d.configuration==cfg)&d.group.isin([g,"Control"])].copy()
    X=transform(x[SPECTRAL_COLS].to_numpy(float),tname)
    x["_row"]=np.arange(len(x))
    diffs=[]
    for b in sorted(x.block.unique()):
        ip=x[(x.block==b)&(x.group==g)].index
        ic=x[(x.block==b)&(x.group=="Control")].index
        if len(ip) and len(ic):
            pp=transform(d.loc[ip,SPECTRAL_COLS].to_numpy(float),tname).mean(axis=0)
            cc=transform(d.loc[ic,SPECTRAL_COLS].to_numpy(float),tname).mean(axis=0)
            diffs.append(pp-cc)
    if len(diffs)<2: return None
    D=np.vstack(diffs)
    mean=D.mean(axis=0); sd=D.std(axis=0,ddof=1)
    dz=np.divide(mean,sd,out=np.zeros_like(mean),where=sd>0)
    consistency=np.maximum((D>0).mean(axis=0),(D<0).mean(axis=0))
    return D,dz,consistency

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/03_e5_sensitivity/banana_spectral_analysis_temporal_primary.csv")
    ap.add_argument("--outdir",default="results/06_spectral_regions")
    args=ap.parse_args()
    d=add_analysis_labels(pd.read_csv(args.input,low_memory=False))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    configs=sorted(d.configuration.unique())
    summaries=[]; flags={}

    for g in TASKS:
      for t in TRANSFORMS:
        flags[(g,t)]=[]
        for e in sorted(d.evaluation.unique()):
          for cfg in configs:
            z=scenario_effect(d,e,g,cfg,t)
            if z is None: continue
            D,dz,cons=z
            stable=(np.abs(dz)>=1)&(cons>=0.75)
            flags[(g,t)].append(stable.astype(int))
            summaries.append({
              "transform":t,"group":g,"evaluation":e,"configuration":cfg,"n_blocks":D.shape[0],
              "n_wavelengths_abs_dz_ge1_consistency_ge075":int(stable.sum()),
              "mean_abs_dz_all_wavelengths":float(np.mean(np.abs(dz))),
              "max_abs_dz":float(np.max(np.abs(dz)))
            })
    pd.DataFrame(summaries).to_csv(out/"spectral_transform_scenario_summary.csv",index=False)

    # Rank fixed 25-nm bins by recurrence across usable scenarios.
    ranks=[]
    for (g,t),arrs in flags.items():
      A=np.vstack(arrs); recurrence=A.mean(axis=0)
      bins=[]
      for start in range(350,2501,25):
        end=min(start+24,2500); ix=(WAVELENGTHS>=start)&(WAVELENGTHS<=end)
        bins.append((start,end,float(recurrence[ix].mean()),float(recurrence[ix].max()),
                     int(A[:,ix].sum(axis=0).max()),A.shape[0]))
      bins.sort(key=lambda x:(x[2],x[3]),reverse=True)
      for rank,b in enumerate(bins[:25],1):
        ranks.append([rank,g,t,*b])
    pd.DataFrame(ranks,columns=["rank","group","transform","start_nm","end_nm",
      "mean_recurrence_fraction","max_recurrence_fraction","max_scenarios_flagged","n_scenarios"]
    ).to_csv(out/"spectral_region_recurrence_top25nm.csv",index=False)
    print("Spectral-region exploration complete.")

if __name__=="__main__":
    main()
