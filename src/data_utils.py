"""Data loading and utility functions for The Liar's Chatroom competition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file, preserving text columns and handling UTF-8 BOM files."""
    return pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)


def load_json(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a JSON file into a DataFrame."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return pd.DataFrame(data)


def load_jsonl(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a JSONL (JSON Lines) file into a DataFrame."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {filepath}: {exc}"
                ) from exc
    return pd.DataFrame(rows)


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load CSV, JSON, or JSONL data based on the file extension."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".jsonl":
        return load_jsonl(path)
    raise ValueError(
        f"Unsupported file format: {suffix}. Supported formats: .csv, .json, .jsonl"
    )


def combine_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    output_col: str = "model_text",
) -> pd.DataFrame:
    """Create one clearly separated model-input field from multiple text columns.

    Missing values are treated as empty strings. Column names are included as
    section headers so the model can distinguish evidence from conversation.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot combine text columns; missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()
    sections = []
    for column in columns:
        values = result[column].fillna("").astype(str).str.strip()
        sections.append("[" + column.upper() + "]\n" + values)
    result[output_col] = sections[0]
    for section in sections[1:]:
        result[output_col] = result[output_col] + "\n\n" + section
    return result


def check_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """Return all rows participating in duplicate records."""
    return df[df.duplicated(subset=subset, keep=False)].sort_values(
        by=subset or df.columns.tolist()
    )


def detect_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Map DataFrame columns to broad, human-readable data types."""
    type_map = {
        "int64": "integer",
        "int32": "integer",
        "float64": "float",
        "float32": "float",
        "object": "string",
        "str": "string",
        "string": "string",
        "string[python]": "string",
        "string[pyarrow]": "string",
        "bool": "boolean",
        "datetime64[ns]": "datetime",
        "datetime64[ns, UTC]": "datetime",
    }
    return {column: type_map.get(str(df[column].dtype), "unknown") for column in df.columns}


def get_column_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get basic statistics for each DataFrame column."""
    stats: Dict[str, Dict] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            stats[column] = {
                "type": "numeric",
                "count": int(series.count()),
                "null_count": int(series.isna().sum()),
                "unique": int(series.nunique()),
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
                "std": float(series.std()) if not series.empty else None,
            }
        else:
            stats[column] = {
                "type": "categorical",
                "count": int(series.count()),
                "null_count": int(series.isna().sum()),
                "unique": int(series.nunique()),
                "top_values": series.value_counts(dropna=False).head(10).to_dict(),
            }
    return stats


def inspect_dataframe(df: pd.DataFrame, verbose: bool = False) -> Dict:
    """Generate a summary of a DataFrame's shape, schema, and missing values."""
    summary = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": detect_schema(df),
        "null_counts": {column: int(df[column].isna().sum()) for column in df.columns},
        "memory_usage": float(df.memory_usage(deep=True).sum() / 1024**2),
    }
    if verbose:
        summary["statistics"] = get_column_statistics(df)
    return summary
