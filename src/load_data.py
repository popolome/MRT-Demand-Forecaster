import os
import pandas as pd

def load_ridership_data() -> pd.DataFrame:
    """
    Load the monthly ridership CSV from data/raw folder.

    Returns:
        DataFrame containing ridership data
    """
    filepath = os.path.join("data", "raw", "monthly_ave_daily_pt_ridership.csv")
    df = pd.read_csv(filepath)

    return df

if __name__ == "__main__":
    df = load_ridership_data()
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nUnique modes: {df['mode'].unique()}")