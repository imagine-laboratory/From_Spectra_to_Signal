\
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import SPECTRAL_COLS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/banana_spectral_master_clean.csv")
    ap.add_argument("--outdir", default="results/01_qc_final")
    args = ap.parse_args()

    d = pd.read_csv(args.input, low_memory=False)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    observed = d["record_status"].eq("observed")

    # Exact spectrum duplicates are recomputed directly over all 2,151 reflectance values.
    duplicate = pd.Series(False, index=d.index)
    duplicate.loc[observed] = d.loc[observed].duplicated(subset=SPECTRAL_COLS, keep=False)

    # Low-signal designation was established during curated-data reconstruction and preserved
    # as a traceable flag. The original ASD files are unavailable, so this flag is not re-inferred.
    low_signal = d.get("spectral_qc_flag", pd.Series("", index=d.index)).fillna("").astype(str).str.contains("suspected_low_signal")
    missing = ~observed

    # Stable duplicate group IDs.
    dup_group = pd.Series(pd.NA, index=d.index, dtype="Int64")
    if duplicate.any():
        sig = pd.util.hash_pandas_object(d.loc[duplicate, SPECTRAL_COLS], index=False)
        codes, _ = pd.factorize(sig, sort=True)
        dup_group.loc[duplicate] = codes + 1

    reason = pd.Series(pd.NA, index=d.index, dtype="object")
    reason.loc[missing] = "missing_measurement"
    reason.loc[low_signal & ~duplicate & observed] = "suspected_low_signal"
    reason.loc[duplicate & ~low_signal & observed] = "exact_duplicate_spectrum_unresolvable"
    reason.loc[duplicate & low_signal & observed] = "suspected_low_signal_and_exact_duplicate"

    q = d.copy()
    q["qc_primary_included"] = observed & ~duplicate & ~low_signal
    q["qc_final_decision"] = np.where(q["qc_primary_included"], "include_primary", "exclude_primary")
    q["qc_exclusion_reason"] = reason
    q["exact_duplicate_group"] = dup_group

    q.to_csv(out/"banana_spectral_master_qc_final.csv", index=False)
    q.loc[q.qc_primary_included].to_csv(out/"banana_spectral_analysis_primary.csv", index=False)
    q.loc[~q.qc_primary_included].to_csv(out/"qc_final_exclusions.csv", index=False)

    dup_cols = ["sample_id","plant_id","evaluation","block","treatment","leaf_number",
                "position","canonical_asd_id","source_file","source_rows","exact_duplicate_group"]
    q.loc[duplicate, dup_cols].rename(columns={"exact_duplicate_group":"duplicate_group"}).to_csv(
        out/"qc_exact_duplicate_groups.csv", index=False
    )

    counts = {
        "expected_slots": len(q),
        "observed": int(observed.sum()),
        "missing": int(missing.sum()),
        "duplicate_records": int(duplicate.sum()),
        "duplicate_groups": int(q.loc[duplicate,"exact_duplicate_group"].nunique()),
        "low_signal_records": int(low_signal.sum()),
        "primary_n": int(q.qc_primary_included.sum()),
    }
    pd.DataFrame([counts]).to_csv(out/"qc_summary.csv", index=False)
    print(counts)

if __name__ == "__main__":
    main()
