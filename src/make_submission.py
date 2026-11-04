"""Submission generation utilities for The Liar's Chatroom competition."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from .data_utils import load_data


def generate_submission(
    model_path: str,
    test_path: str,
    output_path: str,
    id_col: str = "id",
    text_col: str = "text"
) -> pd.DataFrame:
    """Generate a submission file from a trained model.
    
    Args:
        model_path: Path to the saved model (.pkl)
        test_path: Path to test data
        output_path: Path to save submission CSV
        id_col: Name of ID column in test data
        text_col: Name of text column
        
    Returns:
        Submission DataFrame
    """
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = load_data(test_path)
    
    print(f"Loaded {len(test_df)} test samples")
    
    # Prepare features
    X_test = test_df[text_col].values
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        'label': predictions
    })
    
    # Save submission
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")
    
    # Print preview
    print(f"\nSubmission preview ({len(submission)} rows):")
    print(submission.head(10))
    
    return submission


def verify_submission_format(
    submission_path: str,
    sample_submission_path: str
) -> Dict:
    """Verify that a submission matches the expected format.
    
    Args:
        submission_path: Path to the generated submission
        sample_submission_path: Path to the official sample submission
        
    Returns:
        Dictionary with verification results
    """
    print(f"Loading submission: {submission_path}")
    submission = pd.read_csv(submission_path)
    
    print(f"Loading sample submission: {sample_submission_path}")
    sample = pd.read_csv(sample_submission_path)
    
    results = {
        'correct_columns': list(submission.columns) == list(sample.columns),
        'submission_columns': list(submission.columns),
        'sample_columns': list(sample.columns),
        'row_count_match': len(submission) == len(sample),
        'submission_rows': len(submission),
        'sample_rows': len(sample),
        'id_match': set(submission['id']) == set(sample['id']) if 'id' in submission.columns else False
    }
    
    print("\n=== Format Verification ===")
    print(f"Columns match: {results['correct_columns']}")
    print(f"Row count match: {results['row_count_match']}")
    print(f"IDs match: {results['id_match']}")
    
    if not results['correct_columns']:
        print(f"\nExpected columns: {results['sample_columns']}")
        print(f"Found columns: {results['submission_columns']}")
    
    if not results['row_count_match']:
        print(f"\nExpected rows: {results['sample_rows']}")
        print(f"Found rows: {results['submission_rows']}")
    
    return results


def check_prediction_distribution(
    submission: pd.DataFrame,
    target_col: str = "label"
) -> Dict:
    """Check the distribution of predictions.
    
    Args:
        submission: Submission DataFrame
        target_col: Name of the prediction column
        
    Returns:
        Dictionary with prediction statistics
    """
    value_counts = submission[target_col].value_counts()
    
    stats = {
        'total_predictions': len(submission),
        'unique_labels': len(value_counts),
        'label_distribution': value_counts.to_dict(),
        'most_common': value_counts.index[0],
        'least_common': value_counts.index[-1]
    }
    
    print("\n=== Prediction Distribution ===")
    for label, count in stats['label_distribution'].items():
        percentage = (count / stats['total_predictions']) * 100
        print(f"  Label {label}: {count} ({percentage:.1f}%)")
    
    return stats


def create_sample_submission(
    test_path: str,
    output_path: str,
    id_col: str = "id",
    default_label: int = 0
) -> pd.DataFrame:
    """Create a sample submission from test data.
    
    Useful when the official sample submission isn't available.
    
    Args:
        test_path: Path to test data
        output_path: Path to save submission CSV
        id_col: Name of ID column
        default_label: Default label to use for predictions
        
    Returns:
        Submission DataFrame
    """
    print(f"Loading test data from: {test_path}")
    test_df = load_data(test_path)
    
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        'label': default_label
    })
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(output_file, index=False)
    print(f"Sample submission saved to: {output_file}")
    
    return submission


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate submission for The Liar's Chatroom competition"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to saved model file (.pkl)"
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save submission CSV"
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help="Name of ID column in test data (default: id)"
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Name of text column (default: text)"
    )
    parser.add_argument(
        "--verify",
        default=None,
        help="Path to sample submission for format verification (optional)"
    )
    
    args = parser.parse_args()
    
    # Generate submission
    submission = generate_submission(
        model_path=args.model,
        test_path=args.test,
        output_path=args.output,
        id_col=args.id_col,
        text_col=args.text_col
    )
    
    # Verify format if sample submission provided
    if args.verify:
        verify_submission_format(args.output, args.verify)
    
    # Check prediction distribution
    check_prediction_distribution(submission)


if __name__ == "__main__":
    main()
