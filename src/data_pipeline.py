import os
import requests
import pandas as pd
from dotenv import load_dotenv
from utils import validate_date_format, check_raw_data_exists, log, generate_date_range

load_dotenv()
API_KEY = os.getenv("LTA_API_KEY")

BASE_URL = "http://datamall2.mytransport.sg/ltaodataservice"

HEADERS = {
    "AccountKey": API_KEY,
    "accept": "application/json"
}

def fetch_passenger_volume(date: str) -> pd.DataFrame:
    """
    Fetch monthly passenger volume by train station from LTA DataMall.

    Args:
        date: Month to fetch in YYYYMM format e.g. "202501"

    Returns:
        DataFrame containing passenger volume data
    """
    url = f"{BASE_URL}/PV/Train"
    params = {"Date": date}

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["value"])

    return df

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save raw DataFrame to CSV in the data/raw folder.

    Args:
        df: DataFrame to save
        filename: Name of the CSV file e.g. "202501.csv"

    Returns:
        None
    """
    os.makedirs("data/raw", exist_ok=True)
    filepath = os.path.join("data", "raw", filename)
    df.to_csv(filepath, index=False)
    log(f"Saved raw data to {filepath}")

def load_raw_data(filenames: list) -> pd.DataFrame:
    """
    Load and combine multiple monthly CSV files into one DataFrame.

    Args:
        filenames: List of CSV filenames e.g. ["202501.csv", "202502.csv"]

    Returns:
        Combine DataFrame of all months
    """
    dfs = []

    for filename in filenames:
        filepath = os.path.join("data", "raw", filename)
        df = pd.read_csv(filepath)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

if __name__ == "__main__":
    dates = generate_date_range("202301", "202412")

    for date in dates:
        if check_raw_data_exists(f"{date}.csv"):
            log(f"Skipping {date} — already exists")
            continue

        if not validate_date_format(date):
            log(f"Invalid date format: {date} — skipping")
            continue

        log(f"Fetching data for {date}...")
        df = fetch_passenger_volume(date)
        save_raw_data(df, f"{date}.csv")

    log("All data fetched and saved!")