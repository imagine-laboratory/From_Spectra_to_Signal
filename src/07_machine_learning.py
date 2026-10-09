\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import (
    SPECTRAL_COLS,TASKS,add_analysis_labels,transform_matrix,fit_predict_lobo,pooled_metrics
)

FEATURES=["Raw_full_350_2500","Raw_VIS_rededge_520_749","D1_rededge_650_749"]
MODELS=["LogisticRegression","SVM_RBF","PLS_DA"]

def run_dataset(x,task,feature,model):
    x=x[x.group.isin(["Control",task])].copy()
    if x.empty: return None,None
    y=(x.group==task).astype(int).to_numpy()
    if len(np.unique(y))<2 or x.block.nunique()<2: return None,None
    X=transform_matrix(x[SPECTRAL_COLS].to_numpy(float),feature)
    pred=fit_predict_lobo(X,y,x.block.to_numpy(),x.sample_id.astype(str).to_numpy(),model)
    met=pooled_metrics(pred)
    return pred,met

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/03_e5_sensitivity/banana_spectral_analysis_temporal_primary.csv")
    ap.add_argument("--outdir",default="results/07_machine_learning")
    args=ap.parse_args()
    d=add_analysis_labels(pd.read_csv(args.input,low_memory=False))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    result=[]; oof=[]
    for e in sorted(d.evaluation.unique()):
      for task in TASKS:
        for cfg in sorted(d.configuration.unique()):
          x=d[(d.evaluation==e)&(d.configuration==cfg)].copy()
          for feature in FEATURES:
            for model in MODELS:
              p,m=run_dataset(x,task,feature,model)
              if p is None: continue
              row={"evaluation":e,"task":task,"configuration":cfg,
                   "leaf":int(cfg.split("_")[0][1:]),"position":cfg.split("_",1)[1],
                   "feature_set":feature,"model":model,"n":len(p),
                   "n_blocks":p.block.nunique(),
                   "n_control":int((p.true_label==0).sum()),
                   "n_positive":int((p.true_label==1).sum()),**m}
              result.append(row)
              p=p.copy()
              p["evaluation"]=e;p["task"]=task;p["configuration"]=cfg
              p["feature_set"]=feature;p["model"]=model
              # Recover plant IDs from sample IDs through merge.
              idmap=x[["sample_id","plant_id"]].drop_duplicates()
              p=p.merge(idmap,on="sample_id",how="left")
              oof.append(p[["evaluation","task","configuration","feature_set","model",
                            "sample_id","plant_id","block","true_label","pred_label","score"]])

    res=pd.DataFrame(result)
    res.to_csv(out/"ml_leave_one_block_out_results.csv",index=False)
    pd.concat(oof,ignore_index=True).to_csv(out/"ml_out_of_fold_predictions.csv",index=False)

    # Reproduce the cross-pipeline summaries used in the manuscript.
    agg=(res.groupby(["task","feature_set","model"],as_index=False)
           .agg(mean_BA=("balanced_accuracy","mean"),
                median_BA=("balanced_accuracy","median"),
                mean_AUC=("roc_auc","mean"),
                median_AUC=("roc_auc","median")))
    agg.to_csv(out/"ml_model_feature_aggregate.csv",index=False)

    robust=(res.groupby(["task","evaluation","configuration"],as_index=False)
              .agg(median_balanced_accuracy=("balanced_accuracy","median"),
                   min_balanced_accuracy=("balanced_accuracy","min"),
                   max_balanced_accuracy=("balanced_accuracy","max"),
                   median_auc=("roc_auc","median"),
                   mean_balanced_accuracy=("balanced_accuracy","mean")))
    robust.to_csv(out/"ml_configuration_robustness_summary.csv",index=False)

    pairs=[]
    for task in TASKS:
        r=res[res.task==task]
        for a in ["Raw_VIS_rededge_520_749","D1_rededge_650_749"]:
            A=r[r.feature_set==a][["evaluation","configuration","model","balanced_accuracy"]].rename(
                columns={"balanced_accuracy":"BA_A"})
            B=r[r.feature_set=="Raw_full_350_2500"][["evaluation","configuration","model","balanced_accuracy"]].rename(
                columns={"balanced_accuracy":"BA_B"})
            m=A.merge(B,on=["evaluation","configuration","model"],how="inner")
            diff=m.BA_A-m.BA_B
            pairs.append({
                "task":task,"feature_A":a,"feature_B":"Raw_full_350_2500",
                "n_matched":len(m),"A_wins":int((diff>1e-12).sum()),
                "B_wins":int((diff<-1e-12).sum()),
                "ties":int((np.abs(diff)<=1e-12).sum()),
                "mean_BA_diff_A_minus_B":float(diff.mean()),
                "median_BA_diff":float(diff.median())
            })
    pd.DataFrame(pairs).to_csv(out/"ml_feature_set_pairwise_comparison.csv",index=False)

    # Highest observed configuration-specific pipeline at each evaluation (exploratory only).
    best=(res.sort_values(["task","evaluation","balanced_accuracy"],ascending=[True,True,False])
             .groupby(["task","evaluation"],as_index=False).head(1))
    best.to_csv(out/"ml_best_observed_by_evaluation_exploratory.csv",index=False)

    # Whole-plant representation: mean the six valid spectra within plant/evaluation first.
    keys=["plant_id","evaluation","block","treatment","group"]
    whole=d.groupby(keys,as_index=False)[SPECTRAL_COLS].mean()
    whole["sample_id"]=whole["plant_id"].astype(str)+"_E"+whole["evaluation"].astype(str)+"_whole"
    wr=[]
    for e in sorted(whole.evaluation.unique()):
      for task in TASKS:
        x=whole[whole.evaluation==e].copy()
        for feature in FEATURES:
          for model in MODELS:
            p,m=run_dataset(x,task,feature,model)
            if p is None: continue
            wr.append({"evaluation":e,"task":task,"feature_set":feature,"model":model,
                       "n":len(p),"n_blocks":p.block.nunique(),
                       "n_control":int((p.true_label==0).sum()),
                       "n_positive":int((p.true_label==1).sum()),**m})
    pd.DataFrame(wr).to_csv(out/"ml_whole_plant_leave_one_block_out_results.csv",index=False)

    # Task-specific best observed tables, matching the archived reporting convention.
    for task,stem in [("FOC_R1_only","FOC"),("Radopholus_only","Radopholus"),("Coinoculation","Coinoculation")]:
        z=(res[res.task==task].sort_values(["evaluation","balanced_accuracy"],ascending=[True,False])
             .groupby("evaluation",as_index=False).head(1))
        z[["evaluation","configuration","feature_set","model","balanced_accuracy",
           "sensitivity","specificity","roc_auc","n","n_blocks"]].to_csv(
             out/f"ml_{stem}_best_observed_by_evaluation.csv",index=False)

    print("Machine-learning analysis complete.")
    print("Fixed hyperparameters:")
    print("  LogisticRegression: C=1, L2, solver=liblinear, class_weight=balanced")
    print("  RBF-SVM: C=1, gamma=scale, class_weight=balanced")
    print("  PLS-DA: 5 latent components; threshold=0.5")
    print("  No hyperparameter tuning; no nested CV; no held-out block used for model selection.")

if __name__=="__main__":
    main()
