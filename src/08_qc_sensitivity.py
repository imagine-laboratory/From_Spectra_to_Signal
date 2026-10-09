\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import (
    SPECTRAL_COLS,REGIONS,add_analysis_labels,plant_level_mean,spectral_columns,
    exact_signflip_p,holm,transform_matrix,fit_predict_lobo,pooled_metrics
)

def scenario_sets(master):
    observed=master.record_status.eq("observed")
    dup=master.qc_exclusion_reason.fillna("").str.contains("duplicate")
    low=master.qc_exclusion_reason.fillna("").str.contains("low_signal")
    e5b7=(master.evaluation==5)&(master.block==7)
    primary=master.qc_primary_included.astype(bool)
    return {
      "S0_Primary_temporal": master[primary & ~e5b7].copy(),
      "S1_Reinclude_E5B7": master[primary].copy(),
      "S2_Reinclude_duplicates": master[observed & ~low & ~e5b7].copy(),
      "S3_Reinclude_low_signal": master[observed & ~dup & ~e5b7].copy(),
      "S4_All_observed_noE5B7": master[observed & ~e5b7].copy(),
      "S5_All_observed": master[observed].copy(),
    }

def e4_vis_test(df):
    plant,_=plant_level_mean(add_analysis_labels(df))
    vals=[]
    cols=spectral_columns(400,700)
    for b in sorted(plant.block.unique()):
      a=plant[(plant.evaluation==4)&(plant.block==b)&(plant.group=="FOC_R1_only")]
      c=plant[(plant.evaluation==4)&(plant.block==b)&(plant.group=="Control")]
      if len(a) and len(c):
        vals.append(float(a[cols].to_numpy().mean()-c[cols].to_numpy().mean()))
    p=exact_signflip_p(vals)
    # Holm across all five E4 FOC regions.
    ps=[]
    for _,(lo,hi) in REGIONS.items():
      v=[]; cc=spectral_columns(lo,hi)
      for b in sorted(plant.block.unique()):
        a=plant[(plant.evaluation==4)&(plant.block==b)&(plant.group=="FOC_R1_only")]
        c=plant[(plant.evaluation==4)&(plant.block==b)&(plant.group=="Control")]
        if len(a) and len(c): v.append(float(a[cc].to_numpy().mean()-c[cc].to_numpy().mean()))
      ps.append(exact_signflip_p(v))
    return np.mean(vals),p,holm(ps)[0]

def selected_ml(df,e,task,level,feature,model):
    d=add_analysis_labels(df)
    if level=="whole_plant":
      keys=["plant_id","evaluation","block","treatment","group"]
      x=d.groupby(keys,as_index=False)[SPECTRAL_COLS].mean()
      x["sample_id"]=x.plant_id.astype(str)+"_E"+x.evaluation.astype(str)+"_whole"
      x=x[x.evaluation==e]
    else:
      x=d[(d.evaluation==e)&(d.configuration==level)].copy()
    x=x[x.group.isin(["Control",task])].copy()
    y=(x.group==task).astype(int).to_numpy()
    X=transform_matrix(x[SPECTRAL_COLS].to_numpy(float),feature)
    p=fit_predict_lobo(X,y,x.block.to_numpy(),x.sample_id.astype(str).to_numpy(),model)
    return pooled_metrics(p),len(p),p.block.nunique()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--master",default="results/01_qc_final/banana_spectral_master_qc_final.csv")
    ap.add_argument("--outdir",default="results/08_qc_sensitivity")
    args=ap.parse_args()
    master=pd.read_csv(args.master,low_memory=False)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    sets=scenario_sets(master)
    pd.DataFrame([{"set":k,"n_spectra":len(v)} for k,v in sets.items()]).to_csv(
      out/"sensitivity_QC_scenarios.csv",index=False)

    rr=[]; mm=[]
    selected=[
      ("FOC_E1_H6_Margen_full_PLS","FOC_R1_only",1,"H6_Margen","Raw_full_350_2500","PLS_DA"),
      ("FOC_E4_whole_VIS_SVM","FOC_R1_only",4,"whole_plant","Raw_VIS_rededge_520_749","SVM_RBF"),
      ("FOC_E5_whole_full_Logistic","FOC_R1_only",5,"whole_plant","Raw_full_350_2500","LogisticRegression"),
    ]
    for name,df in sets.items():
      delta,p,h=e4_vis_test(df)
      rr.append({"set":name,"evaluation":4,"group":"FOC_R1_only","region":"VIS_400_700",
                 "mean_diff":delta,"raw_p":p,"holm5_p":h})
      for analysis,task,e,level,feat,model in selected:
        met,n,nb=selected_ml(df,e,task,level,feat,model)
        mm.append({"set":name,"analysis":analysis,"task":task,"evaluation":e,
                   "level":level,"representation":feat,"model":model,
                   "dataset_n":len(df),"n_test_obs":n,"n_blocks":nb,**met})
    pd.DataFrame(rr).to_csv(out/"sensitivity_temporal_region_tests.csv",index=False)
    pd.DataFrame(mm).to_csv(out/"sensitivity_selected_ml_models.csv",index=False)
    print("QC sensitivity complete.")

if __name__=="__main__":
    main()
