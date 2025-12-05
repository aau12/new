from __future__ import annotations

from pathlib import Path

import pandas as pd


def _count_empty_string_rows(df: pd.DataFrame) -> int:
    def _is_empty(value: object) -> bool:
        return isinstance(value, str) and value.strip() == ""

    return df.applymap(_is_empty).any(axis=1).sum()


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    file1 = data_dir / "DATA_MEO_Train.csv"
    file2 = data_dir / "DATA_MEO_Train2.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    print(f"Rows in {file1.name}: {len(df1)}")
    print(f"Rows in {file2.name}: {len(df2)}")

    combined = pd.concat([df1, df2], ignore_index=True)

    raw_utc = combined["utc_time"]
    missing_utc_mask = raw_utc.isna() | raw_utc.astype(str).str.strip().eq("")
    missing_utc_count = int(missing_utc_mask.sum())
    empty_field_count = int(_count_empty_string_rows(combined))

    combined["utc_time"] = pd.to_datetime(raw_utc, errors="coerce")
    combined.sort_values("utc_time", inplace=True)

    pre_dedup_rows = len(combined)
    combined = combined.drop_duplicates(subset=["utc_time"], keep="first")
    duplicates_removed = pre_dedup_rows - len(combined)

    combined.reset_index(drop=True, inplace=True)

    output_file = data_dir / "MEO_merged.csv"
    combined.to_csv(output_file, index=False)

    print(f"Rows after merge (before dedup): {pre_dedup_rows}")
    print(f"Rows after removing duplicate timestamps: {len(combined)}")
    print(f"Duplicates removed: {duplicates_removed}")

    if missing_utc_count:
        print(f"Rows missing utc_time: {missing_utc_count}")
    else:
        print("Rows missing utc_time: 0")

    if empty_field_count:
        print(f"Rows with empty string fields: {empty_field_count}")
    else:
        print("Rows with empty string fields: 0")

    valid_utc = combined["utc_time"].dropna()
    if not valid_utc.empty:
        print(
            "Time range: "
            f"{valid_utc.min().isoformat(sep=' ')} to {valid_utc.max().isoformat(sep=' ')}"
        )
    else:
        print("Time range: unavailable (all utc_time values are missing)")


if __name__ == "__main__":
    main()
