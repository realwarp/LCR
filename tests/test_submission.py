"""Tests for submission utilities in The Liar's Chatroom competition toolkit."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.make_submission import (
    generate_submission, verify_submission_format,
    check_prediction_distribution, create_sample_submission
)


class TestSubmission:
    """Test submission generation functionality."""
    
    def test_generate_submission(self, synthetic_train_small, synthetic_test_small, temp_dir):
        """Test generating a submission file."""
        from src.train_baseline import BaselineTrainer
        
        # Save test data to file first
        test_path = temp_dir / 'test.csv'
        synthetic_test_small.to_csv(test_path, index=False)
        
        # Train a model first
        trainer = BaselineTrainer(text_col='text', target_col='label')
        model = trainer.create_word_tfidf_lr()
        
        # Train on synthetic data
        X_train = synthetic_train_small['text'].values
        y_train = synthetic_train_small['label'].values
        model.fit(X_train, y_train)
        
        # Save model
        model_path = temp_dir / 'model.pkl'
        joblib.dump(model, model_path)
        
        # Generate submission
        output_path = temp_dir / 'submission.csv'
        submission = generate_submission(
            model_path=str(model_path),
            test_path=str(test_path),
            output_path=str(output_path)
        )
        
        assert isinstance(submission, pd.DataFrame)
        assert len(submission) == len(synthetic_test_small)
        assert 'id' in submission.columns
        assert 'label' in submission.columns
    
    def test_verify_submission_format_matches(self, sample_submission_format, temp_dir):
        """Test submission format verification when format matches."""
        # Create submission matching sample format (4 rows)
        submission_path = temp_dir / 'submission.csv'
        pd.DataFrame({
            'id': ['conv_000_A', 'conv_000_B', 'conv_000_C', 'conv_000_D'],
            'label': [0, 1, 0, 1]
        }).to_csv(submission_path, index=False)
        
        # Create sample submission
        sample_path = temp_dir / 'sample.csv'
        sample_submission_format.to_csv(sample_path, index=False)
        
        # Verify
        results = verify_submission_format(str(submission_path), str(sample_path))
        
        assert results['correct_columns']
        assert results['row_count_match']
        assert results['id_match']
    
    def test_verify_submission_format_wrong_columns(self, temp_dir):
        """Test submission format verification with wrong columns."""
        # Create submission with wrong columns
        submission_path = temp_dir / 'submission.csv'
        pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'prediction': [0, 1, 0]  # Wrong column name
        }).to_csv(submission_path, index=False)
        
        # Create sample submission
        sample_path = temp_dir / 'sample.csv'
        pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'label': [0, 1, 0]
        }).to_csv(sample_path, index=False)
        
        # Verify
        results = verify_submission_format(str(submission_path), str(sample_path))
        
        assert not results['correct_columns']
    
    def test_check_prediction_distribution(self, sample_submission_format):
        """Test prediction distribution checking."""
        stats = check_prediction_distribution(sample_submission_format)
        
        assert 'total_predictions' in stats
        assert stats['total_predictions'] == 4
        assert 'label_distribution' in stats
        assert 'most_common' in stats
    
    def test_create_sample_submission(self, synthetic_test_small, temp_dir):
        """Test creating a sample submission."""
        # Save test data to file first
        test_path = temp_dir / 'test.csv'
        synthetic_test_small.to_csv(test_path, index=False)
        
        output_path = temp_dir / 'sample_submission.csv'
        submission = create_sample_submission(
            test_path=str(test_path),
            output_path=str(output_path),
            default_label=0
        )
        
        assert isinstance(submission, pd.DataFrame)
        assert len(submission) == len(synthetic_test_small)
        assert submission['label'].nunique() == 1  # All same label
        assert submission['label'].iloc[0] == 0
