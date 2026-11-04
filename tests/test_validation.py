"""Tests for validation utilities in The Liar's Chatroom competition toolkit."""

import pytest
import pandas as pd
import numpy as np

from src.validation import (
    stratified_split,
    group_aware_split,
    stratified_kfold,
    group_aware_kfold,
    calculate_cv_scores,
    check_group_leakage,
    generate_split_report
)


class TestValidation:
    """Test validation and split functions."""
    
    def test_stratified_split_basic(self, sample_conversation_data):
        """Test basic stratified split."""
        train_df, val_df = stratified_split(sample_conversation_data, 'label', test_size=0.2)
        
        assert len(train_df) == 320  # 80% of 400
        assert len(val_df) == 80  # 20% of 400
        assert len(train_df) + len(val_df) == len(sample_conversation_data)
    
    def test_stratified_split_class_balance(self, sample_conversation_data):
        """Test that stratified split preserves class balance."""
        train_df, val_df = stratified_split(sample_conversation_data, 'label', test_size=0.2)
        
        # Check class proportions are similar
        train_proportion = train_df['label'].mean()
        val_proportion = val_df['label'].mean()
        original_proportion = sample_conversation_data['label'].mean()
        
        assert abs(train_proportion - original_proportion) < 0.1
        assert abs(val_proportion - original_proportion) < 0.1
    
    def test_group_aware_split_basic(self, sample_conversation_data):
        """Test basic group-aware split."""
        train_df, val_df = group_aware_split(
            sample_conversation_data, 'label', 'conversation_id', test_size=0.2
        )
        
        assert len(train_df) == 320
        assert len(val_df) == 80
    
    def test_group_aware_split_no_leakage(self, sample_conversation_data):
        """Test that group-aware split prevents group leakage."""
        train_df, val_df = group_aware_split(
            sample_conversation_data, 'label', 'conversation_id', test_size=0.2
        )
        
        train_groups = set(train_df['conversation_id'].unique())
        val_groups = set(val_df['conversation_id'].unique())
        
        # No groups should be in both splits
        common_groups = train_groups.intersection(val_groups)
        assert len(common_groups) == 0
    
    def test_check_group_leakage_no_leakage(self, sample_conversation_data):
        """Test leakage detection when no leakage exists."""
        train_df, val_df = group_aware_split(
            sample_conversation_data, 'label', 'conversation_id', test_size=0.2
        )
        
        leakage = check_group_leakage(train_df, val_df, 'conversation_id')
        
        assert not leakage['leakage_detected']
        assert leakage['num_leaked_groups'] == 0
    
    def test_check_group_leakage_with_leakage(self, sample_conversation_data):
        """Test leakage detection when leakage exists."""
        train_df = sample_conversation_data.iloc[:200]
        val_df = sample_conversation_data.iloc[190:390]  # Overlap!
        
        leakage = check_group_leakage(train_df, val_df, 'conversation_id')
        
        assert leakage['leakage_detected']
        assert leakage['num_leaked_groups'] > 0
    
    def test_stratified_kfold(self, sample_conversation_data):
        """Test stratified k-fold generation."""
        indices = stratified_kfold(sample_conversation_data, 'label', n_splits=5)
        
        assert len(indices) == 5
        
        for train_idx, val_idx in indices:
            assert len(train_idx) == 320
            assert len(val_idx) == 80
            assert len(set(train_idx) & set(val_idx)) == 0
    
    def test_group_aware_kfold(self, sample_conversation_data):
        """Test group-aware k-fold generation."""
        indices = group_aware_kfold(
            sample_conversation_data, 'label', 'conversation_id', n_splits=5
        )
        
        assert len(indices) == 5
        
        for train_idx, val_idx in indices:
            train_groups = set(sample_conversation_data.iloc[train_idx]['conversation_id'].unique())
            val_groups = set(sample_conversation_data.iloc[val_idx]['conversation_id'].unique())
            
            # No group leakage
            assert len(train_groups & val_groups) == 0
    
    def test_generate_split_report(self, sample_conversation_data):
        """Test split report generation."""
        train_df, val_df = stratified_split(sample_conversation_data, 'label')
        
        report = generate_split_report(train_df, val_df, 'label')
        
        assert 'train_size' in report
        assert 'val_size' in report
        assert 'train_class_dist' in report
        assert 'val_class_dist' in report
        assert 'class_balance_check' in report
    
    def test_generate_split_report_class_balance(self, sample_conversation_data):
        """Test class balance in split report."""
        train_df, val_df = stratified_split(sample_conversation_data, 'label')
        
        report = generate_split_report(train_df, val_df, 'label')
        
        train_props = report['class_balance_check']['train_proportions']
        val_props = report['class_balance_check']['val_proportions']
        
        # Proportions should sum to 100%
        assert abs(sum(train_props.values()) - 100) < 0.1
        assert abs(sum(val_props.values()) - 100) < 0.1
