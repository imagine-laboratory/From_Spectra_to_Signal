\
from pathlib import Path
import pandas as pd
import pytest

RESULTS=Path("results")

@pytest.mark.skipif(not (RESULTS/"04_temporal_separability/temporal_region_tests.csv").exists(),
                    reason="Run the pipeline first.")
def test_e4_confirmatory_vis_result():
    d=pd.read_csv(RESULTS/"04_temporal_separability/temporal_region_tests.csv")
    x=d[(d.evaluation==4)&(d.group=="FOC_R1_only")&(d.spectral_region=="VIS_400_700")].iloc[0]
    assert abs(x.mean_signed_difference-0.00441)<1e-5
    assert abs(x.exact_signflip_p-0.0078125)<1e-12
    assert abs(x.holm_p_5_regions-0.0390625)<1e-12

@pytest.mark.skipif(not (RESULTS/"07_machine_learning/ml_whole_plant_leave_one_block_out_results.csv").exists(),
                    reason="Run the pipeline first.")
def test_key_ml_results():
    d=pd.read_csv(RESULTS/"07_machine_learning/ml_whole_plant_leave_one_block_out_results.csv")
    e4=d[(d.evaluation==4)&(d.task=="FOC_R1_only")&
         (d.feature_set=="Raw_VIS_rededge_520_749")&(d.model=="SVM_RBF")].iloc[0]
    e5=d[(d.evaluation==5)&(d.task=="FOC_R1_only")&
         (d.feature_set=="Raw_full_350_2500")&(d.model=="LogisticRegression")].iloc[0]
    assert abs(e4.balanced_accuracy-0.703125)<5e-4
    assert abs(e5.balanced_accuracy-0.799)<0.002
