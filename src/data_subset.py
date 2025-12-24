import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv.gz"
OUTPUT_PATH = "data/processed/lending_club_subset.csv"

START_DATE = "2015-01-01"
END_DATE = "2018-12-31"
VALID_LOAN_STATUS = {"Fully Paid", "Charged Off"}
MAX_ROWS = 200_000
RANDOM_STATE = 42

def main():
    print("Loading raw dataset...")
    df = pd.read_csv(
        RAW_PATH,
        compression="gzip",
        low_memory=False
    )
    print(f"Initial shape: {df.shape}")

    # Filter by loan status
    df = df[df["loan_status"].isin(VALID_LOAN_STATUS)]
    print(f"After loan_status filtering: {df.shape}")

    # Time filtering
    df["issue_d"] = pd.to_datetime(df["issue_d"])

    df = df[
        (df["issue_d"] >= START_DATE) &
        (df["issue_d"] <= END_DATE)
    ]
    print(f"After time filtering ({START_DATE} -> {END_DATE}): {df.shape}")

    # downsampling
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)
        print(f"After downsampling to {MAX_ROWS}: {df.shape}")

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()