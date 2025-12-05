from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURE_ENGINEERING_DIR = BASE_DIR / "feature_engineering_data"

DATASETS = {
    "MEO": (
        FEATURE_ENGINEERING_DIR / "MEO_ewm_features.csv",
        FEATURE_ENGINEERING_DIR / "MEO_interaction_features.csv",
    ),
    "GEO": (
        FEATURE_ENGINEERING_DIR / "GEO_ewm_features.csv",
        FEATURE_ENGINEERING_DIR / "GEO_interaction_features.csv",
    ),
}

EPS = 1e-6

INTERACTION_COLUMNS = [
    "pos_err_norm",
    "xy_ratio",
    "xz_ratio",
    "yz_ratio",
    "clock_pos_ratio",
]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    x_col = next(col for col in df.columns if "x_error" in col)
    y_col = next(col for col in df.columns if "y_error" in col)
    z_col = next(col for col in df.columns if "z_error" in col)
    clock_col = next(col for col in df.columns if "satclockerror" in col)

    x = df[x_col]
    y = df[y_col]
    z = df[z_col]
    clock = df[clock_col]

    pos_err_norm = np.sqrt(x**2 + y**2 + z**2)

    df["pos_err_norm"] = pos_err_norm
    df["xy_ratio"] = x / (y + EPS)
    df["xz_ratio"] = x / (z + EPS)
    df["yz_ratio"] = y / (z + EPS)
    df["clock_pos_ratio"] = clock / (pos_err_norm + EPS)

    return df


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    df.sort_values("utc_time", inplace=True)
    df.set_index("utc_time", inplace=True)

    shape_before = df.shape
    df_with_features = add_interaction_features(df)
    shape_after = df_with_features.shape

    df_with_features.reset_index(inplace=True)
    df_with_features.to_csv(output_path, index=False)

    print(f"--- {label} ---")
    print(f"Shape before: {shape_before}, Shape after: {shape_after}")
    print(f"Interaction columns created: {INTERACTION_COLUMNS}")
    print("Preview of first 10 rows:")
    print(df_with_features.head(10).to_string(index=False))
    print(f"Output saved to: {output_path}\n")


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        if not input_path.exists():
            print(f"Warning: Input file '{input_path}' not found. Skipping {label}...")
            continue
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
