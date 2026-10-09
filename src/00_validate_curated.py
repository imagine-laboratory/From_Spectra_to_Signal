\
import argparse
from pathlib import Path
import pandas as pd
from common import SPECTRAL_COLS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/banana_spectral_master_clean.csv")
    ap.add_argument("--outdir", default="results/00_curated_validation")
    args = ap.parse_args()
    d = pd.read_csv(args.input, low_memory=False)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    required = {"sample_id","plant_id","evaluation","record_status","block","treatment",
                "leaf_number","position",*SPECTRAL_COLS}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing[:20]}")
    summary = {
        "rows_total": len(d),
        "observed": int((d.record_status == "observed").sum()),
        "missing_measurements": int((d.record_status == "missing_measurement").sum()),
        "spectral_bands": len(SPECTRAL_COLS),
        "n_plants": int(d.plant_id.dropna().nunique()),
        "n_evaluations": int(d.evaluation.dropna().nunique()),
        "n_blocks": int(d.block.dropna().nunique()),
        "n_treatments": int(d.treatment.dropna().nunique()),
    }
    pd.DataFrame([summary]).to_csv(out/"curated_validation.csv", index=False)
    print(summary)

if __name__ == "__main__":
    main()
