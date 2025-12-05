from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_DIR_NAME = "15min_resampled"


def log_dataset_stats(
    name: str,
    original_rows: int,
    resampled_rows: int,
    original_first: pd.Timestamp | None,
    original_last: pd.Timestamp | None,
    resampled_first: pd.Timestamp | None,
    resampled_last: pd.Timestamp | None,
    nan_rows: int,
) -> None:
    print(f"--- {name} ---")
    print(f"Original rows: {original_rows}")
    print(f"Rows after resampling: {resampled_rows}")

    if original_first is not None and original_last is not None:
        print(
            "Original time range: "
            f"{original_first.isoformat(sep=' ')} to {original_last.isoformat(sep=' ')}"
        )
    else:
        print("Original time range: unavailable (non-datetime values)")

    if resampled_first is not None and resampled_last is not None:
        print(
            "Resampled time range: "
            f"{resampled_first.isoformat(sep=' ')} to {resampled_last.isoformat(sep=' ')}"
        )
    else:
        print("Resampled time range: unavailable (no valid timestamps)")

    print(f"Rows with NaNs from empty 15-min slots: {nan_rows}")
    print()


def process_dataset(dataset_path: Path, output_path: Path, label: str) -> None:
    df = pd.read_csv(dataset_path)
    original_rows = len(df)

    utc = pd.to_datetime(df["utc_time"], errors="coerce")
    original_first = utc.min() if not utc.isna().all() else None
    original_last = utc.max() if not utc.isna().all() else None

    df["utc_time"] = utc
    df = df.set_index("utc_time")

    resampled = df.resample("15T").mean()
    nan_rows = int(resampled.isna().all(axis=1).sum())

    resampled_first = resampled.index.min() if not resampled.empty else None
    resampled_last = resampled.index.max() if not resampled.empty else None

    resampled_reset = resampled.reset_index()
    resampled_reset.to_csv(output_path, index=False)

    log_dataset_stats(
        label,
        original_rows,
        len(resampled_reset),
        original_first,
        original_last,
        resampled_first,
        resampled_last,
        nan_rows,
    )


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    output_dir = data_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    process_dataset(
        data_dir / "MEO_merged.csv",
        output_dir / "MEO_15min_raw.csv",
        "MEO",
    )

    process_dataset(
        data_dir / "DATA_GEO_Train.csv",
        output_dir / "GEO_15min_raw.csv",
        "GEO",
    )


if __name__ == "__main__":
    main()
