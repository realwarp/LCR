"""Data loading and utility functions for The Liar's Chatroom competition."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    return pd.read_csv(filepath)


def load_json(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        DataFrame with the loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_jsonl(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a JSONL (JSON Lines) file into a pandas DataFrame.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        DataFrame with the loaded data
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load data from a file, auto-detecting the format.
    
    Supports CSV, JSON, and JSONL formats.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame with the loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        return load_csv(filepath)
    elif suffix == '.json':
        return load_json(filepath)
    elif suffix == '.jsonl':
        return load_jsonl(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .json, .jsonl")


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Find duplicate rows in a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates. If None, all columns are used.
        
    Returns:
        DataFrame with duplicate rows
    """
    return df[df.duplicated(subset=subset, keep=False)].sort_values(by=subset or df.columns.tolist())


def detect_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Detect the data type of each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to their detected types
    """
    type_map = {
        'int64': 'integer',
        'int32': 'integer',
        'float64': 'float',
        'float32': 'float',
        'object': 'string',
        'str': 'string',
        'string': 'string',
        'string[python]': 'string',
        'string[pyarrow]': 'string',
        'bool': 'boolean',
        'datetime64[ns]': 'datetime',
        'datetime64[ns, UTC]': 'datetime'
    }
    
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        schema[col] = type_map.get(dtype, 'unknown')
    
    return schema


def get_column_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get basic statistics for each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with statistics for each column
    """
    stats = {}
    
    for col in df.columns:
        col_data = df[col]
        col_type = str(col_data.dtype)
        
        if col_type in ['int64', 'int32', 'float64', 'float32']:
            stats[col] = {
                'type': 'numeric',
                'count': int(col_data.count()),
                'null_count': int(col_data.isnull().sum()),
                'unique': int(col_data.nunique()),
                'min': float(col_data.min()) if not col_data.empty else None,
                'max': float(col_data.max()) if not col_data.empty else None,
                'mean': float(col_data.mean()) if not col_data.empty else None,
                'std': float(col_data.std()) if not col_data.empty else None
            }
        else:
            stats[col] = {
                'type': 'categorical',
                'count': int(col_data.count()),
                'null_count': int(col_data.isnull().sum()),
                'unique': int(col_data.nunique()),
                'top_values': col_data.value_counts().head(10).to_dict()
            }
    
    return stats


def inspect_dataframe(df: pd.DataFrame, verbose: bool = False) -> Dict:
    """Generate a comprehensive summary of a DataFrame.
    
    Args:
        df: Input DataFrame
        verbose: Whether to include detailed information
        
    Returns:
        Dictionary with DataFrame summary
    """
    summary = {
        'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
        'columns': list(df.columns),
        'dtypes': detect_schema(df),
        'null_counts': {col: int(df[col].isnull().sum()) for col in df.columns},
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    if verbose:
        summary['statistics'] = get_column_statistics(df)
    
    return summary
