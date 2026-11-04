"""Model comparison utilities for The Liar's Chatroom competition."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Tuple

from .data_utils import load_data
from .validation import (
    stratified_split,
    group_aware_split,
    calculate_cv_scores
)


class ModelComparator:
    """Compare multiple models with controlled experiments."""
    
    def __init__(
        self,
        text_col: str = "text",
        target_col: str = "label",
        group_col: Optional[str] = None
    ):
        """Initialize the comparator.
        
        Args:
            text_col: Name of the text column
            target_col: Name of the target column
            group_col: Name of the group column for group-aware splits
        """
        self.text_col = text_col
        self.target_col = target_col
        self.group_col = group_col
        self.models = {}
        self.results = {}
    
    def create_word_tfidf_lr(self) -> Pipeline:
        """Word n-gram TF-IDF + LogisticRegression."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                stop_words='english'
            )),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
        ])
    
    def create_char_tfidf_lr(self) -> Pipeline:
        """Character n-gram TF-IDF + LogisticRegression."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(3, 5),
                max_features=100000,
                analyzer='char'
            )),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
        ])
    
    def create_word_tfidf_nb(self) -> Pipeline:
        """Word n-gram TF-IDF + Naive Bayes."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                stop_words='english'
            )),
            ('clf', MultinomialNB())
        ])
    
    def create_word_tfidf_svc(self) -> Pipeline:
        """Word n-gram TF-IDF + Linear SVM."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                stop_words='english'
            )),
            ('clf', LinearSVC(random_state=42, max_iter=1000))
        ])
    
    def create_char_tfidf_svc(self) -> Pipeline:
        """Character n-gram TF-IDF + Linear SVM."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(3, 5),
                max_features=100000,
                analyzer='char'
            )),
            ('clf', LinearSVC(random_state=42, max_iter=1000))
        ])
    
    def create_word_tfidf_rf(self) -> Pipeline:
        """Word n-gram TF-IDF + Random Forest."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,  # Fewer features for RF
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def create_char_tfidf_gb(self) -> Pipeline:
        """Character n-gram TF-IDF + Gradient Boosting."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(3, 5),
                max_features=20000,  # Fewer features for GB
                analyzer='char'
            )),
            ('clf', GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    
    def compare_models(
        self,
        df: pd.DataFrame,
        models_to_compare: Optional[List[str]] = None,
        val_strategy: str = "stratified",
        test_size: float = 0.2,
        random_state: int = 42,
        n_folds: int = 5
    ) -> Dict[str, Dict]:
        """Compare multiple models with the same train/val split.
        
        Args:
            df: Training DataFrame
            models_to_compare: List of model names to compare
            val_strategy: "stratified" or "group_aware"
            test_size: Fraction for validation
            random_state: Random seed
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with results for each model
        """
        # Default models to compare
        all_models = {
            'word_tfidf_lr': self.create_word_tfidf_lr,
            'char_tfidf_lr': self.create_char_tfidf_lr,
            'word_tfidf_nb': self.create_word_tfidf_nb,
            'word_tfidf_svc': self.create_word_tfidf_svc,
            'char_tfidf_svc': self.create_char_tfidf_svc,
            'word_tfidf_rf': self.create_word_tfidf_rf,
            'char_tfidf_gb': self.create_char_tfidf_gb,
        }
        
        if models_to_compare is None:
            models_to_compare = list(all_models.keys())
        
        # Validate model names
        for name in models_to_compare:
            if name not in all_models:
                raise ValueError(f"Unknown model: {name}. Available: {list(all_models.keys())}")
        
        # Split data once for fair comparison
        print(f"\n{'='*60}")
        print("Creating consistent train/val split")
        print(f"{'='*60}")
        
        if val_strategy == "group_aware" and self.group_col:
            train_df, val_df = group_aware_split(
                df, self.target_col, self.group_col, test_size, random_state
            )
        else:
            train_df, val_df = stratified_split(
                df, self.target_col, test_size, random_state
            )
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        X_train = train_df[self.text_col].values
        y_train = train_df[self.target_col].values
        X_val = val_df[self.text_col].values
        y_val = val_df[self.target_col].values
        
        # Evaluate each model
        results = {}
        
        for model_name in models_to_compare:
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            # Create and train model
            model = all_models[model_name]()
            
            print("Fitting model...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"Validation Accuracy: {accuracy:.4f}")
            
            # CV score
            print("Running cross-validation...")
            cv_indices = []
            for fold in range(n_folds):
                train_idx = train_df.index.values
                val_idx = val_df.index.values
                cv_indices.append((train_idx, val_idx))
            
            cv_scores = calculate_cv_scores(model, X_train, y_train, cv_indices)
            print(f"CV Score: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores['mean'],
                'cv_std': cv_scores['std'],
                'train_size': len(train_df),
                'val_size': len(val_df)
            }
            
            # Save model
            self.models[model_name] = model
        
        self.results = results
        
        return results
    
    def print_summary(self):
        """Print a summary of model comparisons."""
        if not self.results:
            print("No results available. Run compare_models() first.")
            return
        
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\n{'Model':<25} {'Val Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
        print("-"*70)
        
        for model_name, results_dict in sorted_results:
            print(f"{model_name:<25} {results_dict['accuracy']:>10.4f} "
                  f"{results_dict['cv_mean']:>10.4f} {results_dict['cv_std']:>10.4f}")
        
        print("="*70)
        print(f"Best model: {sorted_results[0][0]}")
        print("="*70)


def compare_models(
    train_path: str,
    models: Optional[List[str]] = None,
    text_col: str = "text",
    target_col: str = "label",
    group_col: Optional[str] = None,
    val_strategy: str = "stratified"
):
    """Main function to compare models.
    
    Args:
        train_path: Path to training data
        models: List of model names to compare
        text_col: Name of text column
        target_col: Name of target column
        group_col: Name of group column
        val_strategy: "stratified" or "group_aware"
    """
    print(f"Loading training data from: {train_path}")
    df = load_data(train_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    comparator = ModelComparator(
        text_col=text_col,
        target_col=target_col,
        group_col=group_col
    )
    
    results = comparator.compare_models(
        df,
        models_to_compare=models,
        val_strategy=val_strategy
    )
    
    comparator.print_summary()
    
    return results


def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare multiple models for The Liar's Chatroom competition"
    )
    parser.add_argument(
        "train_path",
        help="Path to training data (CSV, JSON, or JSONL)"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        help="Models to compare: word_tfidf_lr, char_tfidf_lr, word_tfidf_nb, "
             "word_tfidf_svc, char_tfidf_svc, word_tfidf_rf, char_tfidf_gb"
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
        "--val-strategy",
        choices=["stratified", "group_aware"],
        default="stratified",
        help="Validation strategy (default: stratified)"
    )
    
    args = parser.parse_args()
    
    compare_models(
        train_path=args.train_path,
        models=args.models,
        text_col=args.text_col,
        target_col=args.target_col,
        group_col=args.group_col,
        val_strategy=args.val_strategy
    )


if __name__ == "__main__":
    main()
