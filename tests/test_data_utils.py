"""Tests for data utilities in The Liar's Chatroom competition toolkit."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_utils import (
    load_csv, load_json, load_jsonl, load_data,
    check_duplicates, detect_schema, get_column_statistics, inspect_dataframe
)


class TestDataUtils:
    """Test data loading and utility functions."""
    
    def test_load_csv(self, sample_csv_file):
        """Test loading a CSV file."""
        df = load_csv(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 400  # 100 conversations x 4 characters
        assert 'id' in df.columns
        assert 'conversation_id' in df.columns
        assert 'character' in df.columns
        assert 'text' in df.columns
        assert 'label' in df.columns
    
    def test_load_jsonl(self, sample_jsonl_file):
        """Test loading a JSONL file."""
        df = load_jsonl(sample_jsonl_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 400
        assert 'id' in df.columns
    
    def test_load_data_auto_detect_csv(self, sample_csv_file):
        """Test auto-detection of CSV format."""
        df = load_data(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 400
    
    def test_load_data_auto_detect_jsonl(self, sample_jsonl_file):
        """Test auto-detection of JSONL format."""
        df = load_data(sample_jsonl_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 400
    
    def test_load_data_unsupported_format(self, temp_dir):
        """Test error handling for unsupported formats."""
        filepath = temp_dir / "data.txt"
        filepath.write_text("test")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(filepath)
    
    def test_check_duplicates_no_duplicates(self, sample_conversation_data):
        """Test duplicate detection when no duplicates exist."""
        duplicates = check_duplicates(sample_conversation_data)
        assert len(duplicates) == 0
    
    def test_check_duplicates_with_duplicates(self, sample_conversation_data):
        """Test duplicate detection when duplicates exist."""
        # Add duplicates
        df_with_dupes = pd.concat([
            sample_conversation_data,
            sample_conversation_data.iloc[:10].copy()
        ])
        
        duplicates = check_duplicates(df_with_dupes)
        # All 20 rows (original 10 + 10 duplicates) are considered duplicates
        assert len(duplicates) == 20
    
    def test_check_duplicates_by_subset(self, sample_conversation_data):
        """Test duplicate detection by subset of columns."""
        # Make some duplicates in conversation_id
        df = sample_conversation_data.copy()
        df.loc[0:3, 'conversation_id'] = 'conv_dup_001'
        
        # Check for duplicates by conversation_id
        duplicates = check_duplicates(df, subset=['conversation_id'])
        # All rows should be marked as duplicates because they all share conversation_id with at least one other
        assert len(duplicates) > 0
    
    def test_detect_schema(self, sample_conversation_data):
        """Test schema detection."""
        schema = detect_schema(sample_conversation_data)
        
        assert schema['id'] == 'string'
        assert schema['conversation_id'] == 'string'
        assert schema['character'] == 'string'
        assert schema['text'] == 'string'
        assert schema['label'] == 'integer'
    
    def test_get_column_statistics_numeric(self, sample_conversation_data):
        """Test statistics for numeric column."""
        stats = get_column_statistics(sample_conversation_data)
        
        assert 'label' in stats
        assert stats['label']['type'] == 'numeric'
        assert 'count' in stats['label']
        assert 'mean' in stats['label']
    
    def test_get_column_statistics_categorical(self, sample_conversation_data):
        """Test statistics for categorical column."""
        stats = get_column_statistics(sample_conversation_data)
        
        assert 'character' in stats
        assert stats['character']['type'] == 'categorical'
        assert 'unique' in stats['character']
        assert 'top_values' in stats['character']
    
    def test_inspect_dataframe_basic(self, sample_conversation_data):
        """Test basic dataframe inspection."""
        summary = inspect_dataframe(sample_conversation_data)
        
        assert 'shape' in summary
        assert summary['shape']['rows'] == 400
        assert summary['shape']['cols'] == 5
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'null_counts' in summary
    
    def test_inspect_dataframe_verbose(self, sample_conversation_data):
        """Test verbose dataframe inspection."""
        summary = inspect_dataframe(sample_conversation_data, verbose=True)
        
        assert 'statistics' in summary
        for col in sample_conversation_data.columns:
            assert col in summary['statistics']
