import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RAW_PATH = "data/processed/lending_club_subset.csv"
OUTPUT_PATH = "predict/lending_club_template.csv"

def main():
    """Matches the model's schema by loading a single row from data."""
    df = pd.read_csv(RAW_PATH, nrows=1)
    for col in df.columns:
        df[col] = np.nan
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
