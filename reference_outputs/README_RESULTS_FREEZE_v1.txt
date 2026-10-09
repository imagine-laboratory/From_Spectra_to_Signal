FOCR1 — FINAL RESULTS FREEZE v1

Purpose
-------
This package defines the numerical and interpretive results that should be treated as frozen
for manuscript drafting. Changes should be made only if a demonstrable analytical error is found.

Central paper story
-------------------
A reproducible spectral response associated with FOC R1 inoculation first reaches confirmatory
regional support at Evaluation 4, primarily in the visible region (400–700 nm), and this result
is robust to every tested QC scenario.

Primary result
--------------
FOC R1-only vs control, Evaluation 4, VIS 400–700 nm:
- mean reflectance difference: +0.00441
- approximately +9.9% relative to control mean VIS reflectance
- exact block sign-flip p = 0.0078125
- Holm-adjusted p = 0.0390625 across five predefined spectral regions
- result remains significant in all six QC sensitivity scenarios

Important constraints
---------------------
- No single wavelength survives maxT family-wise correction across 2,151 bands.
- No global multivariate comparison reaches p < 0.05.
- Leaf 6 contributes most to the E4 FOC visible response, but no single leaf-position
  configuration survives the full 30-test correction.
- The 520–749 nm window is the most defensible FOC feature region, but feature-region rankings
  remain exploratory.
- Machine learning is secondary evidence. Isolated best-performing pipelines are not
  independently validated optima.
- Do not use "confirmed infection" unless pathology confirmation becomes available; use
  inoculated plants / inoculation status.
- Radopholus-only and coinoculation do not have confirmatory inferential spectral separation.

Primary figures selected
------------------------
Figure 1: FOC temporal regional response
Figure 2: E4 FOC difference spectrum
Figure 3: E4 leaf-position contribution

Supplementary figures selected
------------------------------
E5 Block 7 QC anomaly
FOC E4 QC sensitivity
FOC spectral-region recurrence
FOC ML robustness
Selected ML QC sensitivity

Files
-----
FOCR1_final_results_freeze_v1.xlsx
final_claims_registry.csv
paper_figure_plan.csv
paper_table_plan.csv
analysis_freeze_metadata.csv
figures_main/
figures_supplementary/
source_tables/

Next step
---------
Draft the manuscript from this frozen result set.
