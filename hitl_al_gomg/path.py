import os
from pathlib import Path
import pandas as pd

def table():
    files = [
        [key, value, value.exists()]
        for key, value in globals().items()
        if isinstance(value, Path)
    ]
    return pd.DataFrame(files, columns=["variable", "path", "exists"])

# Determine paths
_here = Path(__file__).resolve()
_repo_root = _here.parent.parent

# Data
_data = _repo_root / "data"

# Add paths to data files and directories here:
training = _data / "train"
testing = _data / "test"
chemspace = _data / "chemspace"

# Add paths to trained model files and directories here:
predictors = _data / "predictors"
simulators = _data / "simulators"
priors = _data / "priors"

# REINVENT
reinvent = _repo_root / "ReinventHITL"
reinventconda = _repo_root / "reinvent-hitl"

# Results
demos = _repo_root / "generations"

if __name__ == "__main__":
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table())
