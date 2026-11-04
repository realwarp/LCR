"""Data inspection utilities for The Liar's Chatroom competition."""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import sys

from .data_utils import load_data, inspect_dataframe


def inspect_file(filepath: str, verbose: bool = False, output_dir: Optional[str] = None) -> Dict:
    """Inspect a data file and generate a profile.
    
    Args:
        filepath: Path to the data file
        verbose: Whether to include detailed statistics
        output_dir: Directory to save the profile (optional)
        
    Returns:
        Dictionary with the data profile
    """
    print(f"Loading file: {filepath}")
    
    try:
        df = load_data(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Generate summary
    profile = inspect_dataframe(df, verbose=verbose)
    
    # Print basic info
    print("\n=== Data Summary ===")
    print(f"Shape: {profile['shape']['rows']} rows x {profile['shape']['cols']} columns")
    print(f"Memory usage: {profile['memory_usage']:.2f} MB")
    
    print("\n=== Column Types ===")
    for col, dtype in profile['dtypes'].items():
        null_count = profile['null_counts'][col]
        print(f"  {col}: {dtype} ({null_count} nulls)")
    
    if verbose:
        print("\n=== Detailed Statistics ===")
        for col, stats in profile['statistics'].items():
            print(f"\n{col}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    # Save profile if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON profile
        json_path = output_path / "data_profile.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"\nProfile saved to: {json_path}")
        
        # Generate markdown report
        md_path = output_path / "data_profile.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Data Profile\n\n")
            f.write(f"## File: {filepath}\n\n")
            f.write(f"- **Rows:** {profile['shape']['rows']}\n")
            f.write(f"- **Columns:** {profile['shape']['cols']}\n")
            f.write(f"- **Memory:** {profile['memory_usage']:.2f} MB\n\n")
            
            f.write("## Columns\n\n")
            f.write("| Column | Type | Nulls |\n")
            f.write("|--------|------|-------|\n")
            for col, dtype in profile['dtypes'].items():
                null_count = profile['null_counts'][col]
                f.write(f"| {col} | {dtype} | {null_count} |\n")
        
        print(f"Markdown report saved to: {md_path}")
    else:
        print("\nProfile generated (use -o to save to file)")
    
    return profile


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect and profile a data file for The Liar's Chatroom competition"
    )
    parser.add_argument(
        "filepath",
        help="Path to the data file (CSV, JSON, or JSONL)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Include detailed statistics"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save the profile (default: same as input file)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output_dir
    
    inspect_file(args.filepath, verbose=args.verbose, output_dir=output_dir)


if __name__ == "__main__":
    main()
