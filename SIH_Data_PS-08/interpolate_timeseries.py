from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).resolve().parent / "15min_resampled"
DATASETS = {
    "MEO": (INPUT_DIR / "MEO_Zscore_outliers_removed.csv", INPUT_DIR / "MEO_interpolated.csv"),
    "GEO": (INPUT_DIR / "GEO_Zscore_outliers_removed.csv", INPUT_DIR / "GEO_interpolated.csv"),
}


def _count_total_nans(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    df = df.set_index("utc_time")

    before_nans = _count_total_nans(df)

    interpolated = df.interpolate(method="time")

    boundary_actions: list[str] = []
    if not interpolated.empty and interpolated.iloc[0].isna().any():
        interpolated = interpolated.bfill()
        boundary_actions.append("bfill")
    if not interpolated.empty and interpolated.iloc[-1].isna().any():
        interpolated = interpolated.ffill()
        boundary_actions.append("ffill")

    after_nans = _count_total_nans(interpolated)

    result = interpolated.reset_index()
    result.to_csv(output_path, index=False)

    print(f"--- {label} ---")
    print(f"NaNs before interpolation: {before_nans}")
    print(f"NaNs after interpolation: {after_nans}")
    if boundary_actions:
        print("Boundary fills applied: " + ", ".join(boundary_actions))
    else:
        print("Boundary fills applied: none")
    print()


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
