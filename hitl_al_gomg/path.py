from pathlib import Path
import pandas as pd

def get_package_root():
    # Find the path of the current file
    return Path(__file__).resolve().parent

def table():
    files = [
        [key, value, value.exists()]
        for key, value in globals().items()
        if isinstance(value, Path)
    ]
    return pd.DataFrame(files, columns=["variable", "path", "exists"])

# Get the package root path dynamically
package_root = get_package_root()

# Path to chembl sample data and trained Reinvent priors
chemspace = package_root / "scoring" / "chemspace"
priors = package_root / "models" / "priors"

if __name__ == "__main__":
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(table())
