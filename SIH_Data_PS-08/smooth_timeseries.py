from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).resolve().parent / "15min_resampled"
DATASETS = {
    "MEO": (INPUT_DIR / "MEO_interpolated.csv", INPUT_DIR / "MEO_smoothed.csv"),
    "GEO": (INPUT_DIR / "GEO_interpolated.csv", INPUT_DIR / "GEO_smoothed.csv"),
}
NUMERIC_COLUMNS = ["x_error", "y_error", "z_error", "satclockerror"]
NORMALIZED_TARGETS = {"".join(col.lower().split("_")): col for col in NUMERIC_COLUMNS}


def _normalize_column_name(name: str) -> str:
    return name.split("(")[0].strip().lower().replace(" ", "").replace("_", "")


def _coalesce_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for normalized_target, target in NORMALIZED_TARGETS.items():
        matching_columns = [
            column
            for column in df.columns
            if _normalize_column_name(column) == normalized_target
        ]
        if not matching_columns:
            continue
        keeper = matching_columns[0]
        for extra in matching_columns[1:]:
            df[keeper] = df[keeper].combine_first(df[extra])
            df.drop(columns=extra, inplace=True)
    return df


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in df.columns:
        normalized = _normalize_column_name(column)
        for normalized_target, target in NORMALIZED_TARGETS.items():
            if normalized.startswith(normalized_target) and target not in mapping:
                mapping[target] = column
                break
    missing = [target for target in NUMERIC_COLUMNS if target not in mapping]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")
    return mapping


def _count_nans(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df = _coalesce_measurement_columns(df)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    df = df.set_index("utc_time")
    before_nans = _count_nans(df)

    col_map = _resolve_columns(df.reset_index())
    columns_to_smooth = list(col_map.values())

    smoothed = df.copy()
    smoothed[columns_to_smooth] = (
        smoothed[columns_to_smooth]
        .rolling(window=3, center=True, min_periods=1)
        .median()
    )

    after_nans = _count_nans(smoothed)

    result = smoothed.reset_index()
    result.to_csv(output_path, index=False)

    first_ts = smoothed.index.min()
    last_ts = smoothed.index.max()

    print(f"--- {label} ---")
    print(f"NaNs before smoothing: {before_nans}")
    print(f"NaNs after smoothing: {after_nans}")
    if first_ts is not None and last_ts is not None:
        print(
            "Timestamp range preserved: "
            f"{first_ts.isoformat(sep=' ')} to {last_ts.isoformat(sep=' ')}"
        )
    else:
        print("Timestamp range preserved: unavailable")
    print()


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
