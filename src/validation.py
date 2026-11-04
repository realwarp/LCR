"""Validation and split utilities for The Liar's Chatroom competition."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/validation split.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )
    return train_df, val_df


def group_aware_split(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a group-aware stratified split to prevent leakage.
    
    This ensures all rows from the same group go to the same split.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        group_col: Name of the group column (e.g., conversation_id)
        test_size: Fraction of groups for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Get unique groups
    groups = df[group_col].unique()
    
    # Stratify by target distribution within groups
    group_targets = df.groupby(group_col)[target_col].first()
    
    train_groups, val_groups = train_test_split(
        groups,
        test_size=test_size,
        stratify=group_targets,
        random_state=random_state
    )
    
    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    
    return train_df, val_df


def stratified_kfold(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold indices.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) for each fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = []
    
    for train_idx, val_idx in skf.split(df, df[target_col]):
        indices.append((train_idx, val_idx))
    
    return indices


def group_aware_kfold(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate group-aware stratified k-fold indices.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        group_col: Name of the group column
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) for each fold
    """
    groups = df[group_col].unique()
    group_targets = df.groupby(group_col)[target_col].first()
    
    # Create group-level stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_indices = []
    for train_group_idx, val_group_idx in skf.split(groups, group_targets):
        train_groups = groups[train_group_idx]
        val_groups = groups[val_group_idx]
        
        train_indices = df[df[group_col].isin(train_groups)].index.values
        val_indices = df[df[group_col].isin(val_groups)].index.values
        
        fold_indices.append((train_indices, val_indices))
    
    return fold_indices


def calculate_cv_scores(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_indices: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, float]:
    """Calculate cross-validation scores.
    
    Args:
        model: Scikit-learn model instance
        X: Feature matrix
        y: Target vector
        cv_indices: List of (train_indices, val_indices)
        
    Returns:
        Dictionary with mean and std of CV scores
    """
    scores = []
    
    for train_idx, val_idx in cv_indices:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'all_scores': scores
    }


def check_group_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str
) -> Dict[str, bool]:
    """Check for group leakage between train and validation sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        group_col: Name of the group column
        
    Returns:
        Dictionary with leakage detection results
    """
    train_groups = set(train_df[group_col].unique())
    val_groups = set(val_df[group_col].unique())
    
    common_groups = train_groups.intersection(val_groups)
    
    return {
        'leakage_detected': len(common_groups) > 0,
        'num_leaked_groups': len(common_groups),
        'leaked_groups': list(common_groups)[:10]  # First 10 for review
    }


def generate_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str
) -> Dict:
    """Generate a report comparing train and validation splits.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        target_col: Name of the target column
        
    Returns:
        Dictionary with split statistics
    """
    report = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'total_size': len(train_df) + len(val_df),
        'train_class_dist': train_df[target_col].value_counts().to_dict(),
        'val_class_dist': val_df[target_col].value_counts().to_dict(),
        'class_balance_check': {
            'train_proportions': (train_df[target_col].value_counts(normalize=True) * 100).to_dict(),
            'val_proportions': (val_df[target_col].value_counts(normalize=True) * 100).to_dict()
        }
    }
    
    return report