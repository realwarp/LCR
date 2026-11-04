"""Baseline training utilities for The Liar's Chatroom competition."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path

from .data_utils import load_data
from .validation import (
    stratified_split,
    group_aware_split,
    stratified_kfold,
    calculate_cv_scores,
    check_group_leakage,
    generate_split_report
)


class BaselineTrainer:
    """Train baseline models for The Liar's Chatroom competition."""
    
    def __init__(
        self,
        text_col: str = "text",
        target_col: str = "label",
        group_col: Optional[str] = None
    ):
        """Initialize the trainer.
        
        Args:
            text_col: Name of the text column
            target_col: Name of the target column
            group_col: Name of the group column for group-aware splits
        """
        self.text_col = text_col
        self.target_col = target_col
        self.group_col = group_col
        self.models = {}
        self.vectorizers = {}
        self.results = {}
    
    def create_word_tfidf_lr(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 50000
    ) -> Pipeline:
        """Create a TF-IDF + LogisticRegression pipeline with word n-grams.
        
        Args:
            ngram_range: Range of n-gram sizes
            max_features: Maximum number of features
            
        Returns:
            Scikit-learn Pipeline
        """
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words='english',
                lowercase=True
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ))
        ])
    
    def create_char_tfidf_lr(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 100000
    ) -> Pipeline:
        """Create a TF-IDF + LogisticRegression pipeline with character n-grams.
        
        Args:
            ngram_range: Range of n-gram sizes
            max_features: Maximum number of features
            
        Returns:
            Scikit-learn Pipeline
        """
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                analyzer='char',
                lowercase=True
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ))
        ])
    
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_name: str,
        model: Pipeline,
        val_strategy: str = "stratified",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """Train a model and evaluate on a validation set.
        
        Args:
            df: Training DataFrame
            model_name: Name to identify the model
            model: Scikit-learn Pipeline
            val_strategy: "stratified" or "group_aware"
            test_size: Fraction for validation
            random_state: Random seed
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Split data
        if val_strategy == "group_aware" and self.group_col:
            train_df, val_df = group_aware_split(
                df, self.target_col, self.group_col, test_size, random_state
            )
        else:
            train_df, val_df = stratified_split(
                df, self.target_col, test_size, random_state
            )
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Check for leakage
        if self.group_col:
            leakage = check_group_leakage(train_df, val_df, self.group_col)
            if leakage['leakage_detected']:
                print(f"WARNING: {leakage['num_leaked_groups']} groups leaked to validation!")
        
        # Prepare data
        X_train = train_df[self.text_col].values
        y_train = train_df[self.target_col].values
        X_val = val_df[self.text_col].values
        y_val = val_df[self.target_col].values
        
        # Train model
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_val, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Cross-validation using the training data
        print("Running 5-fold CV...")
        cv_indices = stratified_kfold(train_df, self.target_col, n_splits=5, random_state=42)
        cv_scores = calculate_cv_scores(model, X_train, y_train, cv_indices)
        print(f"CV Score: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
        
        # Store results
        self.models[model_name] = model
        self.results[model_name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores['mean'],
            'cv_std': cv_scores['std'],
            'cv_scores': cv_scores['all_scores'],
            'val_report': report,
            'train_size': len(train_df),
            'val_size': len(val_df)
        }
        
        return self.results[model_name]
    
    def train_baseline_models(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Train both baseline models and compare.
        
        Args:
            df: Training DataFrame
            output_dir: Directory to save models (optional)
            
        Returns:
            Dictionary with results for both models
        """
        # Word n-gram model
        word_model = self.create_word_tfidf_lr()
        word_results = self.train_and_evaluate(
            df, "word_tfidf_lr", word_model,
            val_strategy="group_aware" if self.group_col else "stratified"
        )
        
        # Character n-gram model
        char_model = self.create_char_tfidf_lr()
        char_results = self.train_and_evaluate(
            df, "char_tfidf_lr", char_model,
            val_strategy="group_aware" if self.group_col else "stratified"
        )
        
        # Save models
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(word_model, output_path / "word_tfidf_lr.pkl")
            joblib.dump(char_model, output_path / "char_tfidf_lr.pkl")
            
            print(f"\nModels saved to: {output_path}")
        
        return {
            "word_tfidf_lr": word_results,
            "char_tfidf_lr": char_results
        }


def train_baseline(
    train_path: str,
    text_col: str = "text",
    target_col: str = "label",
    group_col: Optional[str] = None,
    output_dir: str = "models",
    val_strategy: str = "stratified"
):
    """Main function to train baseline models.
    
    Args:
        train_path: Path to training data
        text_col: Name of text column
        target_col: Name of target column
        group_col: Name of group column for leakage prevention
        output_dir: Directory to save models
        val_strategy: "stratified" or "group_aware"
    """
    print(f"Loading training data from: {train_path}")
    df = load_data(train_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    
    trainer = BaselineTrainer(
        text_col=text_col,
        target_col=target_col,
        group_col=group_col
    )
    
    results = trainer.train_baseline_models(df, output_dir=output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name, results_dict in results.items():
        print(f"\n{model_name}:")
        print(f"  Validation Accuracy: {results_dict['accuracy']:.4f}")
        print(f"  CV Score: {results_dict['cv_mean']:.4f} (+/- {results_dict['cv_std']:.4f})")
    
    # Determine best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model by accuracy: {best_model[0]}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train baseline models for The Liar's Chatroom competition"
    )
    parser.add_argument(
        "train_path",
        help="Path to training data (CSV, JSON, or JSONL)"
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Name of text column (default: text)"
    )
    parser.add_argument(
        "--target-col",
        default="label",
        help="Name of target column (default: label)"
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Name of group column for group-aware splits (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--val-strategy",
        choices=["stratified", "group_aware"],
        default="stratified",
        help="Validation strategy (default: stratified)"
    )
    
    args = parser.parse_args()
    
    train_baseline(
        train_path=args.train_path,
        text_col=args.text_col,
        target_col=args.target_col,
        group_col=args.group_col,
        output_dir=args.output_dir,
        val_strategy=args.val_strategy
    )


if __name__ == "__main__":
    main()
