from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMNS = ["x_error", "y_error", "z_error", "satclockerror"]
NORMALIZED_TARGETS = {"".join(col.lower().split("_")): col for col in TARGET_COLUMNS}
INPUT_DIR = Path(__file__).resolve().parent / "15min_resampled"
DATASETS = {
    "MEO": (INPUT_DIR / "MEO_15min_raw.csv", INPUT_DIR / "MEO_Zscore_outliers_removed.csv"),
    "GEO": (INPUT_DIR / "GEO_15min_raw.csv", INPUT_DIR / "GEO_Zscore_outliers_removed.csv"),
}


def _normalize_column_name(name: str) -> str:
    base = name.split("(")[0]
    base = base.strip().lower().replace(" ", "").replace("_", "")
    return base


def _select_numeric_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in df.columns:
        if column.lower() == "utc_time":
            continue
        normalized = _normalize_column_name(column)
        for normalized_target, target in NORMALIZED_TARGETS.items():
            if normalized.startswith(normalized_target) and target not in mapping:
                mapping[target] = column
                break
    missing = [target for target in TARGET_COLUMNS if target not in mapping]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")
    return mapping


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    column_map = _select_numeric_columns(df)
    total_outliers = 0

    print(f"--- {label} ---")
    print(f"Total rows: {len(df)}")

    for target in TARGET_COLUMNS:
        column = column_map[target]
        series = pd.to_numeric(df[column], errors="coerce")
        mean = series.mean()
        std = series.std()

        if pd.isna(std) or std == 0:
            outliers = 0
        else:
            z_scores = (series - mean) / std
            mask = z_scores.abs() > 3
            outliers = int(mask.sum())
            df.loc[mask, column] = pd.NA

        total_outliers += outliers
        print(f"{column}: mean={mean:.6f} std={std:.6f} | outliers replaced: {outliers}")

    print(f"Total outlier values replaced: {total_outliers}\n")

    df.to_csv(output_path, index=False)


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
