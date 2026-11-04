"""Tests for training utilities in The Liar's Chatroom competition toolkit."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.train_baseline import BaselineTrainer


class TestTraining:
    """Test baseline training functionality."""
    
    def test_create_word_tfidf_lr(self):
        """Test word n-gram TF-IDF + LR model creation."""
        trainer = BaselineTrainer()
        model = trainer.create_word_tfidf_lr()
        
        assert len(model.steps) == 2
        assert 'tfidf' in dict(model.steps)
        assert 'clf' in dict(model.steps)
    
    def test_create_char_tfidf_lr(self):
        """Test character n-gram TF-IDF + LR model creation."""
        trainer = BaselineTrainer()
        model = trainer.create_char_tfidf_lr()
        
        assert len(model.steps) == 2
        vectorizer = model.steps[0][1]
        assert vectorizer.analyzer == 'char'
    
    def test_train_and_evaluate_word_model(self, synthetic_train_small):
        """Test training a word n-gram model."""
        trainer = BaselineTrainer(text_col='text', target_col='label')
        model = trainer.create_word_tfidf_lr()
        
        results = trainer.train_and_evaluate(
            synthetic_train_small, 'word_test', model,
            val_strategy='stratified'
        )
        
        assert 'accuracy' in results
        assert 'cv_mean' in results
        assert 'cv_std' in results
        assert results['accuracy'] >= 0  # Accuracy between 0 and 1
        assert results['accuracy'] <= 1
    
    def test_train_and_evaluate_group_aware(self, synthetic_train_small):
        """Test training with group-aware split."""
        trainer = BaselineTrainer(
            text_col='text', target_col='label', group_col='conversation_id'
        )
        model = trainer.create_word_tfidf_lr()
        
        results = trainer.train_and_evaluate(
            synthetic_train_small, 'group_test', model,
            val_strategy='group_aware'
        )
        
        assert results['accuracy'] >= 0
        assert results['accuracy'] <= 1
    
    def test_train_baseline_models(self, synthetic_train_small):
        """Test training both baseline models."""
        trainer = BaselineTrainer(
            text_col='text', target_col='label', group_col='conversation_id'
        )
        
        results = trainer.train_baseline_models(synthetic_train_small)
        
        assert 'word_tfidf_lr' in results
        assert 'char_tfidf_lr' in results
        
        # Both models should have results
        for model_name in ['word_tfidf_lr', 'char_tfidf_lr']:
            assert 'accuracy' in results[model_name]
            assert 'cv_mean' in results[model_name]
    
    def test_train_baseline_models_saves_models(self, synthetic_train_small, temp_dir):
        """Test that trained models are saved."""
        trainer = BaselineTrainer(text_col='text', target_col='label')
        
        output_dir = temp_dir / 'models'
        results = trainer.train_baseline_models(synthetic_train_small, output_dir=str(output_dir))
        
        # Check models were saved
        assert (output_dir / 'word_tfidf_lr.pkl').exists()
        assert (output_dir / 'char_tfidf_lr.pkl').exists()
    
    def test_model_comparison_accuracy_order(self, synthetic_train_small):
        """Test that model comparison produces reasonable results."""
        trainer = BaselineTrainer(text_col='text', target_col='label')
        
        word_model = trainer.create_word_tfidf_lr()
        char_model = trainer.create_char_tfidf_lr()
        
        # Word model results
        word_results = trainer.train_and_evaluate(
            synthetic_train_small, 'word', word_model,
            val_strategy='stratified'
        )
        
        # Char model results
        char_results = trainer.train_and_evaluate(
            synthetic_train_small, 'char', char_model,
            val_strategy='stratified'
        )
        
        # Both should have reasonable accuracy
        assert 0 <= word_results['accuracy'] <= 1
        assert 0 <= char_results['accuracy'] <= 1
