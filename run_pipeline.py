\
from pathlib import Path
import subprocess, sys

STAGES = [
    "00_validate_curated.py",
    "01_qc_final.py",
    "02_exploration.py",
    "03_e5_sensitivity.py",
    "04_temporal_separability.py",
    "05_leaf_position.py",
    "06_spectral_regions.py",
    "07_machine_learning.py",
    "08_qc_sensitivity.py",
    "09_validate_paper_results.py",
]

for stage in STAGES:
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([sys.executable, str(Path("src")/stage)], check=True)
