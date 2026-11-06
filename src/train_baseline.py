"""Train word- and character-TF-IDF baselines for The Liar's Chatroom.

Works with the competition schema:
    id, evidence, conversation, liar

It also retains support for generic datasets with text/label columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .data_utils import combine_text_columns, load_data


class BaselineTrainer:
    """Train and evaluate multiclass TF-IDF + Logistic Regression baselines."""

    def __init__(
        self,
        text_col: str = "model_text",
        target_col: str = "liar",
        group_col: Optional[str] = None,
        cv_folds: int = 5,
    ) -> None:
        self.text_col = text_col
        self.target_col = target_col
        self.group_col = group_col
        self.cv_folds = cv_folds
        self.models: Dict[str, Pipeline] = {}
        self.results: Dict[str, Dict] = {}

    @staticmethod
    def create_word_tfidf_lr(
        ngram_range: Tuple[int, int] = (1, 2), max_features: int = 50000
    ) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=ngram_range,
                        max_features=max_features,
                        stop_words="english",
                        lowercase=True,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    @staticmethod
    def create_char_tfidf_lr(
        ngram_range: Tuple[int, int] = (3, 4), max_features: int = 30000
    ) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=ngram_range,
                        max_features=max_features,
                        analyzer="char_wb",
                        lowercase=True,
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=700,
                        random_state=42,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        if self.group_col:
            if self.group_col not in df.columns:
                raise KeyError(f"Group column '{self.group_col}' is not in the dataset.")
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, val_idx = next(
                splitter.split(df, df[self.target_col], groups=df[self.group_col])
            )
        else:
            indices = np.arange(len(df))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=df[self.target_col],
            )
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

    def _cross_validate(self, model: Pipeline, train_df: pd.DataFrame) -> Dict:
        X = train_df[self.text_col].astype(str).to_numpy()
        y = train_df[self.target_col].to_numpy()
        smallest_class = int(pd.Series(y).value_counts().min())
        folds = min(self.cv_folds, smallest_class)
        if folds < 2:
            raise ValueError("Not enough examples per class to run cross-validation.")

        if self.group_col:
            group_count = train_df[self.group_col].nunique()
            folds = min(folds, group_count)
            if folds < 2:
                raise ValueError("At least two distinct groups are required for group CV.")
            cv = GroupKFold(n_splits=folds)
            scores = cross_val_score(
                model,
                X,
                y,
                groups=train_df[self.group_col].to_numpy(),
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
            )
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="accuracy", n_jobs=1
            )

        return {
            "folds": int(folds),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "all_scores": [float(score) for score in scores],
        }

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_name: str,
        model: Pipeline,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        print(f"\n{'=' * 64}\nTraining: {model_name}\n{'=' * 64}")
        train_df, val_df = self._split(df, test_size, random_state)
        print(f"Train size: {len(train_df)}, validation size: {len(val_df)}")

        X_train = train_df[self.text_col].astype(str).to_numpy()
        y_train = train_df[self.target_col].to_numpy()
        X_val = val_df[self.text_col].astype(str).to_numpy()
        y_val = val_df[self.target_col].to_numpy()

        print("Fitting holdout model...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, average="macro")
        report_text = classification_report(y_val, predictions, zero_division=0)
        report_dict = classification_report(
            y_val, predictions, output_dict=True, zero_division=0
        )

        print(f"\nValidation accuracy: {accuracy:.4f}")
        print(f"Validation macro-F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(report_text)

        if self.cv_folds >= 2:
            print(f"Running {self.cv_folds}-fold cross-validation on training split...")
            cv_results = self._cross_validate(model, train_df)
            print(
                f"CV accuracy: {cv_results['mean']:.4f} "
                f"(+/- {cv_results['std']:.4f})"
            )
        else:
            print("Cross-validation skipped (use --cv-folds 3 to enable it).")
            cv_results = {"mean": None, "std": None, "all_scores": [], "folds": 0}

        result = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "cv_mean": cv_results["mean"],
            "cv_std": cv_results["std"],
            "cv_scores": cv_results["all_scores"],
            "cv_folds": cv_results["folds"],
            "val_report": report_dict,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "labels": [str(label) for label in model.classes_],
        }
        self.models[model_name] = model
        self.results[model_name] = result
        return result

    def train_baseline_models(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = "models",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Dict]:
        results = {}
        model_factories = {
            "word_tfidf_lr": self.create_word_tfidf_lr,
            "char_tfidf_lr": self.create_char_tfidf_lr,
        }

        for model_name, factory in model_factories.items():
            model = factory()
            results[model_name] = self.train_and_evaluate(
                df, model_name, model, test_size=test_size, random_state=random_state
            )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            X_full = df[self.text_col].astype(str).to_numpy()
            y_full = df[self.target_col].to_numpy()

            print("\nRefitting both selected baselines on ALL labeled training rows...")
            for model_name, factory in model_factories.items():
                final_model = factory()
                final_model.fit(X_full, y_full)
                self.models[model_name] = final_model
                model_path = output_path / f"{model_name}.pkl"
                joblib.dump(final_model, model_path)
                print(f"Saved: {model_path}")

            metrics_path = output_path / "baseline_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
            print(f"Saved metrics: {metrics_path}")

        return results


def prepare_training_data(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    target_col: Optional[str] = None,
    evidence_col: str = "evidence",
    conversation_col: str = "conversation",
) -> tuple[pd.DataFrame, str, str]:
    """Resolve this competition's columns or fall back to generic text/label data."""
    if target_col is None:
        target_col = "liar" if "liar" in df.columns else "label"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    if text_col:
        if text_col not in df.columns:
            raise KeyError(
                f"Text column '{text_col}' not found. Available columns: {list(df.columns)}"
            )
        prepared = df.copy()
        resolved_text_col = text_col
    elif evidence_col in df.columns and conversation_col in df.columns:
        prepared = combine_text_columns(
            df, [evidence_col, conversation_col], output_col="model_text"
        )
        resolved_text_col = "model_text"
    elif "text" in df.columns:
        prepared = df.copy()
        resolved_text_col = "text"
    else:
        raise KeyError(
            "Could not find model input columns. Expected evidence + conversation, "
            "or a single text column. Use --text-col to specify a custom text column. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows that cannot be used for supervised training.
    prepared = prepared.dropna(subset=[target_col]).copy()
    prepared[resolved_text_col] = prepared[resolved_text_col].fillna("").astype(str)
    prepared[target_col] = prepared[target_col].astype(str).str.strip()
    prepared = prepared[prepared[resolved_text_col].str.strip().ne("")].copy()
    if prepared.empty:
        raise ValueError("No usable rows remain after removing missing labels/empty text.")

    class_counts = prepared[target_col].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two distinct target classes.")
    if class_counts.min() < 2:
        raise ValueError(
            "Each class needs at least two rows for a stratified train/validation split. "
            f"Class counts: {class_counts.to_dict()}"
        )
    return prepared, resolved_text_col, target_col


def train_baseline(
    train_path: str,
    text_col: Optional[str] = None,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    output_dir: str = "models",
    test_size: float = 0.2,
    cv_folds: int = 0,
    evidence_col: str = "evidence",
    conversation_col: str = "conversation",
) -> Dict[str, Dict]:
    print(f"Loading training data from: {train_path}")
    df = load_data(train_path)
    print(f"Loaded {len(df)} rows")
    print(f"Detected columns: {list(df.columns)}")

    df, resolved_text_col, resolved_target_col = prepare_training_data(
        df,
        text_col=text_col,
        target_col=target_col,
        evidence_col=evidence_col,
        conversation_col=conversation_col,
    )

    print(f"\nModel input: {resolved_text_col}")
    print(f"Target: {resolved_target_col}")
    print(f"Usable rows: {len(df)}")
    print("Target distribution:")
    print(df[resolved_target_col].value_counts().sort_index().to_string())
    print(f"Number of classes: {df[resolved_target_col].nunique()}")

    trainer = BaselineTrainer(
        text_col=resolved_text_col,
        target_col=resolved_target_col,
        group_col=group_col,
        cv_folds=cv_folds,
    )
    results = trainer.train_baseline_models(
        df,
        output_dir=output_dir,
        test_size=test_size,
    )

    print(f"\n{'=' * 64}\nBASELINE COMPARISON\n{'=' * 64}")
    for name, result in results.items():
        print(
            f"{name}: accuracy={result['accuracy']:.4f}, "
            f"macro-F1={result['macro_f1']:.4f}, "
            + (f"CV={result['cv_mean']:.4f} +/- {result['cv_std']:.4f}" if result["cv_mean"] is not None else "CV=skipped")
        )
    best_name = max(results, key=lambda name: results[name]["accuracy"])
    print(f"\nBest holdout accuracy: {best_name}")
    print(f"Final models are saved in: {output_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF baselines for The Liar's Chatroom competition."
    )
    parser.add_argument(
        "train_path", help="Training data path (CSV, JSON, or JSONL)."
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Use one existing text column instead of combining evidence and conversation.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column (auto-detects 'liar', then 'label').",
    )
    parser.add_argument(
        "--evidence-col", default="evidence", help="Evidence column name."
    )
    parser.add_argument(
        "--conversation-col", default="conversation", help="Conversation column name."
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Optional group column for group-aware holdout and CV.",
    )
    parser.add_argument(
        "--output-dir", default="models", help="Folder in which trained models are saved."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Validation fraction (default: 0.2)."
    )
    parser.add_argument(
        "--cv-folds", type=int, default=0, help="Optional CV folds; 0 skips CV (default: 0)."
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be greater than 0 and less than 1.")
    if args.cv_folds != 0 and args.cv_folds < 2:
        parser.error("--cv-folds must be 0 (skip) or at least 2.")

    train_baseline(
        train_path=args.train_path,
        text_col=args.text_col,
        target_col=args.target_col,
        group_col=args.group_col,
        output_dir=args.output_dir,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        evidence_col=args.evidence_col,
        conversation_col=args.conversation_col,
    )


if __name__ == "__main__":
    main()
