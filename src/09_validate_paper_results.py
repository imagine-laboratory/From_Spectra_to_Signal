\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def close(x,y,tol): return abs(float(x)-float(y)) <= tol

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",default="results")
    args=ap.parse_args()
    r=Path(args.results)
    checks=[]

    q=pd.read_csv(r/"01_qc_final/qc_summary.csv").iloc[0]
    checks += [
      ("curated expected slots", q.expected_slots==2880, q.expected_slots, 2880),
      ("observed spectra", q.observed==2873, q.observed, 2873),
      ("QC-primary spectra", q.primary_n==2761, q.primary_n, 2761),
    ]
    t=pd.read_csv(r/"03_e5_sensitivity/banana_spectral_analysis_temporal_primary.csv",low_memory=False)
    checks.append(("temporal-primary spectra",len(t)==2703,len(t),2703))

    reg=pd.read_csv(r/"04_temporal_separability/temporal_region_tests.csv")
    x=reg[(reg.evaluation==4)&(reg.group=="FOC_R1_only")&(reg.spectral_region=="VIS_400_700")].iloc[0]
    checks += [
      ("E4 FOC VIS deltaR",close(x.mean_signed_difference,0.00441,1e-5),x.mean_signed_difference,0.00441),
      ("E4 FOC VIS exact p",close(x.exact_signflip_p,0.0078125,1e-12),x.exact_signflip_p,0.0078125),
      ("E4 FOC VIS Holm p",close(x.holm_p_5_regions,0.0390625,1e-12),x.holm_p_5_regions,0.0390625),
    ]

    wp=pd.read_csv(r/"07_machine_learning/ml_whole_plant_leave_one_block_out_results.csv")
    e4=wp[(wp.evaluation==4)&(wp.task=="FOC_R1_only")&
          (wp.feature_set=="Raw_VIS_rededge_520_749")&(wp.model=="SVM_RBF")].iloc[0]
    e5=wp[(wp.evaluation==5)&(wp.task=="FOC_R1_only")&
          (wp.feature_set=="Raw_full_350_2500")&(wp.model=="LogisticRegression")].iloc[0]
    checks += [
      ("E4 whole-plant VIS SVM BA",close(e4.balanced_accuracy,0.703125,5e-4),e4.balanced_accuracy,0.703125),
      ("E5 whole-plant full logistic BA",close(e5.balanced_accuracy,0.799,0.002),e5.balanced_accuracy,0.799),
    ]

    failed=[c for c in checks if not c[1]]
    for name,ok,got,expected in checks:
      print(("PASS" if ok else "FAIL"),name,"got=",got,"expected=",expected)
    if failed:
      raise SystemExit(f"{len(failed)} frozen-result checks failed.")
    print("All frozen paper-result checks passed.")

if __name__=="__main__":
    main()
