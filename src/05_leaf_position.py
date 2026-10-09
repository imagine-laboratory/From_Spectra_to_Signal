\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import REGIONS,TASKS,SPECTRAL_COLS,add_analysis_labels,spectral_columns,exact_signflip_p,holm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/03_e5_sensitivity/banana_spectral_analysis_temporal_primary.csv")
    ap.add_argument("--outdir",default="results/05_leaf_position")
    args=ap.parse_args()
    d=add_analysis_labels(pd.read_csv(args.input,low_memory=False))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    configs=sorted(d.configuration.unique())
    rows=[]
    for e in sorted(d.evaluation.unique()):
      for g in TASKS:
        family=[]
        for cfg in configs:
          for region,(lo,hi) in REGIONS.items():
            cols=spectral_columns(lo,hi); vals=[]; blocks=[]
            for b in sorted(d.block.unique()):
              a=d[(d.evaluation==e)&(d.block==b)&(d.group==g)&(d.configuration==cfg)]
              c=d[(d.evaluation==e)&(d.block==b)&(d.group=="Control")&(d.configuration==cfg)]
              if len(a) and len(c):
                vals.append(float(a[cols].to_numpy().mean()-c[cols].to_numpy().mean()))
                blocks.append(int(b))
            p=exact_signflip_p(vals)
            leaf=int(cfg.split("_")[0][1:]); pos=cfg.split("_",1)[1]
            family.append([e,g,cfg,leaf,pos,len(vals),",".join(map(str,blocks)),region,
                           np.mean(vals),np.std(vals,ddof=1),p])
        p30=holm([r[-1] for r in family])
        # within config: 5 regions
        h5={}
        for cfg in configs:
          idx=[i for i,r in enumerate(family) if r[2]==cfg]
          adj=holm([family[i][-1] for i in idx])
          for i,a in zip(idx,adj): h5[i]=a
        # across 6 configs within region
        h6={}
        for region in REGIONS:
          idx=[i for i,r in enumerate(family) if r[7]==region]
          adj=holm([family[i][-1] for i in idx])
          for i,a in zip(idx,adj): h6[i]=a
        for i,r in enumerate(family):
          rows.append(r+[p30[i],h5[i],h6[i]])

    cols=["evaluation","group","configuration","leaf","position","n_blocks","blocks",
          "spectral_region","mean_signed_difference","sd_block_difference","exact_signflip_p",
          "holm_p_30_config_x_region","holm_p_5_regions_within_config","holm_p_6_configs_within_region"]
    result=pd.DataFrame(rows,columns=cols)
    result.to_csv(out/"leaf_position_region_tests.csv",index=False)

    e4=result[(result.evaluation==4)&(result.group=="FOC_R1_only")&(result.spectral_region=="VIS_400_700")].copy()
    e4["paired_effect_dz"]=e4.mean_signed_difference/e4.sd_block_difference
    # Directional consistency recomputed from raw block contrasts.
    cons=[]
    for _,r in e4.iterrows():
      cfg=r.configuration; vals=[]
      for b in sorted(d.block.unique()):
        a=d[(d.evaluation==4)&(d.block==b)&(d.group=="FOC_R1_only")&(d.configuration==cfg)]
        c=d[(d.evaluation==4)&(d.block==b)&(d.group=="Control")&(d.configuration==cfg)]
        if len(a) and len(c):
          cols2=spectral_columns(400,700)
          vals.append(float(a[cols2].to_numpy().mean()-c[cols2].to_numpy().mean()))
      cons.append(max(np.mean(np.array(vals)>0),np.mean(np.array(vals)<0)))
    e4["sign_consistency"]=cons
    e4.rename(columns={
      "mean_signed_difference":"mean_VIS_difference","sd_block_difference":"sd_block_difference",
      "exact_signflip_p":"raw_p","holm_p_5_regions_within_config":"holm5_within_config",
      "holm_p_6_configs_within_region":"holm6_across_configs","holm_p_30_config_x_region":"holm30_global"
    })[["configuration","leaf","position","n_blocks","mean_VIS_difference","sd_block_difference",
        "paired_effect_dz","sign_consistency","raw_p","holm5_within_config",
        "holm6_across_configs","holm30_global"]].to_csv(out/"FOC_E4_visible_by_leaf_position.csv",index=False)
    print("Leaf-position analysis complete.")

if __name__=="__main__":
    main()
