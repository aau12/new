from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller

INPUT_DIR = Path(__file__).resolve().parent / "15min_resampled"
DATASETS = {
    "MEO": (
        INPUT_DIR / "MEO_smoothed.csv",
        INPUT_DIR / "ADF_MEO_results.csv",
    ),
    "GEO": (
        INPUT_DIR / "GEO_smoothed.csv",
        INPUT_DIR / "ADF_GEO_results.csv",
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


def run_adf(series: pd.Series) -> dict[str, float | str]:
    clean_series = series.dropna()
    if clean_series.empty:
        raise ValueError("Cannot run ADF on empty series after dropping NaNs")
    result = adfuller(clean_series, autolag="AIC")
    adf_statistic, p_value, used_lags, n_obs, critical_values, _ = result
    interpretation = "Stationary" if p_value < 0.05 else "Non-Stationary"
    return {
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "num_lags_used": used_lags,
        "num_observations_used": n_obs,
        "critical_value_1%": critical_values["1%"],
        "critical_value_5%": critical_values["5%"],
        "critical_value_10%": critical_values["10%"],
        "interpretation": interpretation,
    }


def process_dataset(label: str, input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    df = df.set_index("utc_time")

    column_map = _resolve_columns(df.reset_index())

    results = []
    stationary_count = 0
    for target in NUMERIC_COLUMNS:
        column = column_map[target]
        stats = run_adf(df[column])
        if stats["interpretation"] == "Stationary":
            stationary_count += 1
        stats_row = {"variable_name": column}
        stats_row.update(stats)
        results.append(stats_row)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    print(f"ADF completed for {label} dataset")
    print(f"Variables analyzed: {len(NUMERIC_COLUMNS)}")
    print(
        f"Stationary variables: {stationary_count}, Non-stationary: {len(NUMERIC_COLUMNS) - stationary_count}"
    )
    print()


def main() -> None:
    for label, (input_path, output_path) in DATASETS.items():
        process_dataset(label, input_path, output_path)


if __name__ == "__main__":
    main()
