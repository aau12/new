from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_DIR = Path(__file__).resolve().parent / "15min_resampled"
SCALER_DIR = Path(__file__).resolve().parent / "models" / "scalers"
DATASETS = {
    "MEO": (
        INPUT_DIR / "MEO_smoothed.csv",
        INPUT_DIR / "MEO_scaled.csv",
        SCALER_DIR / "MEO_scaler.pkl",
    ),
    "GEO": (
        INPUT_DIR / "GEO_smoothed.csv",
        INPUT_DIR / "GEO_scaled.csv",
        SCALER_DIR / "GEO_scaler.pkl",
    ),
}
NUMERIC_COLUMNS = ["x_error", "y_error", "z_error", "satclockerror"]
NORMALIZED_TARGETS = {"".join(col.lower().split("_")): col for col in NUMERIC_COLUMNS}


def _normalize_column_name(name: str) -> str:
    base = name.split("(")[0]
    return base.strip().lower().replace(" ", "").replace("_", "")


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in df.columns:
        if column.lower() == "utc_time":
            continue
        normalized = _normalize_column_name(column)
        for normalized_target, target in NORMALIZED_TARGETS.items():
            if normalized.startswith(normalized_target) and target not in mapping:
                mapping[target] = column
                break
    missing = [target for target in NUMERIC_COLUMNS if target not in mapping]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")
    return mapping


def process_dataset(label: str, input_path: Path, output_path: Path, scaler_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    column_map = _resolve_columns(df)
    numeric_columns = [column_map[target] for target in NUMERIC_COLUMNS]

    print(f"--- {label} ---")
    for target, column in zip(NUMERIC_COLUMNS, numeric_columns):
        mean = df[column].mean()
        std = df[column].std()
        print(f"Before scaling {column}: mean={mean:.6f}, std={std:.6f}")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_columns])

    scaled_df = df.copy()
    scaled_df[numeric_columns] = scaled_values
    scaled_df.to_csv(output_path, index=False)

    joblib.dump(scaler, scaler_path)

    after_means = scaled_df[numeric_columns].mean()
    after_stds = scaled_df[numeric_columns].std()
    for column in numeric_columns:
        print(
            f"After scaling {column}: mean={after_means[column]:.4f}, std={after_stds[column]:.4f}"
        )
    print()


def main() -> None:
    SCALER_DIR.mkdir(parents=True, exist_ok=True)
    for label, (input_path, output_path, scaler_path) in DATASETS.items():
        process_dataset(label, input_path, output_path, scaler_path)


if __name__ == "__main__":
    main()
