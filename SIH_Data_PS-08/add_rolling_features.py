from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURE_ENGINEERING_DIR = BASE_DIR / "feature_engineering_data"

DATASETS = {
    "MEO": (
        FEATURE_ENGINEERING_DIR / "MEO_lag_features.csv",
        FEATURE_ENGINEERING_DIR / "MEO_rolling_features.csv",
    ),
    "GEO": (
        FEATURE_ENGINEERING_DIR / "GEO_lag_features.csv",
        FEATURE_ENGINEERING_DIR / "GEO_rolling_features.csv",
    ),
}

VARIABLE_PATTERNS = [
    ("x_error", "x"),
    ("y_error", "y"),
    ("z_error", "z"),
    ("satclockerror", "clock"),
]
ROLLING_WINDOWS = [3, 6, 12, 24]


def add_rolling_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute rolling statistics and slopes for specified variables."""
    df = df.copy()
    created_columns: list[str] = []

    for pattern, short_name in VARIABLE_PATTERNS:
        matching_cols = [col for col in df.columns if pattern in col]
        if not matching_cols:
            print(f"Warning: No column matching '{pattern}' found. Skipping...")
            continue

        col = matching_cols[0]
        series = df[col]

        for window in ROLLING_WINDOWS:
            rolling = series.rolling(window=window, min_periods=window)

            mean_col = f"{short_name}_roll_mean_{window}"
            std_col = f"{short_name}_roll_std_{window}"
            min_col = f"{short_name}_roll_min_{window}"
            max_col = f"{short_name}_roll_max_{window}"
            slope_col = f"{short_name}_roll_slope_{window}"

            df[mean_col] = rolling.mean()
            df[std_col] = rolling.std()
            df[min_col] = rolling.min()
            df[max_col] = rolling.max()
            df[slope_col] = (series - series.shift(window - 1)) / window

            created_columns.extend(
                [mean_col, std_col, min_col, max_col, slope_col]
            )

    return df, created_columns


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    df.sort_values("utc_time", inplace=True)
    df.set_index("utc_time", inplace=True)

    shape_before = df.shape
    df_with_features, created_columns = add_rolling_features(df)
    shape_after = df_with_features.shape

    df_with_features.reset_index(inplace=True)
    df_with_features.to_csv(output_path, index=False)

    print(f"--- {label} ---")
    print(f"Number of rolling features created: {len(created_columns)}")
    print(f"Shape before: {shape_before}, Shape after: {shape_after}")
    print("Preview of first 20 rows (showing NaNs):")
    print(df_with_features.head(20).to_string(index=False))
    print(f"Output saved to: {output_path}\n")


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        if not input_path.exists():
            print(f"Warning: Input file '{input_path}' not found. Skipping {label}...")
            continue
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
