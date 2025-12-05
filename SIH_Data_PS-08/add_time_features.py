from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "15min_resampled"
FEATURE_ENGINEERING_DIR = BASE_DIR / "feature_engineering_data"
FEATURE_ENGINEERING_DIR.mkdir(exist_ok=True)

DATASETS = {
    "MEO": (INPUT_DIR / "MEO_scaled.csv", FEATURE_ENGINEERING_DIR / "MEO_time_features.csv"),
    "GEO": (INPUT_DIR / "GEO_scaled.csv", FEATURE_ENGINEERING_DIR / "GEO_time_features.csv"),
}
NEW_FEATURES = [
    "hour",
    "minute",
    "dow",
    "doy",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dow"] = df.index.weekday
    df["doy"] = df.index.dayofyear

    df["hour_sin"] = df["hour"].apply(lambda h: math.sin(2 * math.pi * h / 24))
    df["hour_cos"] = df["hour"].apply(lambda h: math.cos(2 * math.pi * h / 24))
    df["doy_sin"] = df["doy"].apply(lambda d: math.sin(2 * math.pi * d / 365))
    df["doy_cos"] = df["doy"].apply(lambda d: math.cos(2 * math.pi * d / 365))

    return df


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    df.sort_values("utc_time", inplace=True)
    df.set_index("utc_time", inplace=True)

    shape_before = df.shape
    df_with_features = add_time_features(df)
    shape_after = df_with_features.shape

    df_with_features.reset_index(inplace=True)
    df_with_features.to_csv(output_path, index=False)

    first_ts = df_with_features["utc_time"].iloc[0]
    last_ts = df_with_features["utc_time"].iloc[-1]

    print(f"--- {label} ---")
    print(f"New features added: {len(NEW_FEATURES)}")
    print(f"First timestamp: {first_ts}")
    print(f"Last timestamp: {last_ts}")
    print(f"Shape before: {shape_before}, Shape after: {shape_after}")
    print()


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
