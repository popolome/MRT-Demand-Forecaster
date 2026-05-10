import os
import pandas as pd
from datetime import datetime

def validate_date_format(date: str) -> bool:
    """
    Validate that a date string is in YYYYMM format.

    Args:
        date: Date string to validate e.g. "202501"

    Returns:
        True if valid, False if not
    """
    try:
        datetime.strptime(date, "%Y%m")
        return True
    except ValueError:
        return False
    
def generate_date_range(start_date: str, end_date: str) -> list:
    """
    Generate a list of dates in YYYYMM format between two months.

    Args:
        start_date: Start month in YYYYMM format e.g. "202301"
        end_date: End month in YYYYMM format e.g. "202312"

    Returns:
        List of date strings in YYYYMM format
    """
    start = datetime.strptime(start_date, "%Y%m")
    end = datetime.strptime(end_date, "%Y%m")

    dates = []
    current = start

    while current <= end:
        dates.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return dates

def check_raw_data_exists(filename: str) -> bool:
    """
    Check if a raw data file already exists to avoid re-fetching.

    Args:
        filename: CSV filename to check e.g. "202501.csv"

    Returns:
        True if file exists, False if not
    """
    filepath = os.path.join("data", "raw", filename)
    return os.path.exists(filepath)

def log(message: str) -> None:
    """
    Print a timestamped log message.

    Args:
        message: Message to log

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")