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

# Models
_models = _repo_root / "HITL_AL_GOMG/models"

# Add paths to data files and directories here:
training = _data / "train"
testing = _data / "test"
chemspace = _data / "chemspace"

# Add paths to trained model files and directories here:
predictors = _models / "predictors"
simulators = _models / "simulators"
priors = _models / "priors"

# REINVENT
reinvent = _repo_root / "HITL_AL_GOMG/Reinvent"
reinventconda = _repo_root / "HITL_AL_GOMG/reinvent.v3.2"

# Results
demos = _repo_root / "generations"

if __name__ == "__main__":
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table())
