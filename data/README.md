# Data input

Place the curated dataset from `00. FOCR1_curated_data_package` here as:

`banana_spectral_master_clean.csv`

The repository intentionally does not bundle the biological dataset because its
public-release status must be decided by the data owners.

The curated file is the earliest preserved machine-readable analysis artifact.
It already contains the traceable ID/metadata corrections, the three reconstructed
split spectra, and the low-signal QC flag. Therefore this repository reproduces the
paper analysis **from the curated dataset forward**; it does not claim to reconstruct
the unavailable original instrument `.asd` files.
