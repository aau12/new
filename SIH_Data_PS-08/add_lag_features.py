from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURE_ENGINEERING_DIR = BASE_DIR / "feature_engineering_data"

DATASETS = {
    "MEO": (
        FEATURE_ENGINEERING_DIR / "MEO_time_features.csv",
        FEATURE_ENGINEERING_DIR / "MEO_lag_features.csv",
    ),
    "GEO": (
        FEATURE_ENGINEERING_DIR / "GEO_time_features.csv",
        FEATURE_ENGINEERING_DIR / "GEO_lag_features.csv",
    ),
}

# Base column patterns to search for (will match variations in spacing)
LAG_COLUMN_PATTERNS = ["x_error", "y_error", "z_error", "satclockerror"]
# Short names for lag column naming
LAG_COLUMN_NAMES = ["x", "y", "z", "clock"]
# Lag steps (in 15-minute intervals)
LAG_STEPS = [1, 2, 4, 8, 16, 24, 48, 96]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for specified columns."""
    df = df.copy()
    
    lag_columns_created = []
    
    for pattern, short_name in zip(LAG_COLUMN_PATTERNS, LAG_COLUMN_NAMES):
        # Find the actual column name that matches the pattern
        matching_cols = [col for col in df.columns if pattern in col]
        
        if not matching_cols:
            print(f"Warning: No column matching '{pattern}' found in dataframe, skipping...")
            continue
        
        # Use the first matching column
        col = matching_cols[0]
            
        for lag_step in LAG_STEPS:
            lag_col_name = f"{short_name}_lag_{lag_step}"
            df[lag_col_name] = df[col].shift(lag_step)
            lag_columns_created.append(lag_col_name)
    
    return df, lag_columns_created


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    """Process a single dataset by adding lag features."""
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    # Convert utc_time to datetime
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    
    # Sort rows by utc_time
    df.sort_values("utc_time", inplace=True)
    
    # Set utc_time as index (required for correct shifting)
    df.set_index("utc_time", inplace=True)
    
    shape_before = df.shape
    
    # Add lag features
    df_with_lags, lag_columns_created = add_lag_features(df)
    
    shape_after = df_with_lags.shape
    
    # Count rows with NaN due to lagging
    # NaN will be present in the first max(LAG_STEPS) rows for lag columns
    nan_count = df_with_lags[lag_columns_created].isna().any(axis=1).sum()
    
    # Reset index to save utc_time as a column
    df_with_lags.reset_index(inplace=True)
    
    # Save to output file
    df_with_lags.to_csv(output_path, index=False)
    
    # Logging
    print(f"--- {label} ---")
    print(f"Number of lag columns created: {len(lag_columns_created)}")
    print(f"Shape before: {shape_before}, Shape after: {shape_after}")
    print(f"Number of rows containing NaN due to lagging: {nan_count}")
    print(f"Output saved to: {output_path}")
    print()


def main() -> None:
    """Process all datasets."""
    for label, (input_path, output_path) in DATASETS.items():
        if not input_path.exists():
            print(f"Warning: Input file '{input_path}' not found, skipping {label}...")
            continue
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
