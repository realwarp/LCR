"""Tests for data inspection utilities."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.inspect_data import inspect_file


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': ['Hello', 'World', 'Test', 'Data', 'Sample'],
        'label': [0, 1, 0, 1, 0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    
    yield path
    Path(path).unlink()


@pytest.fixture
def sample_jsonl():
    """Create a temporary JSONL file for testing."""
    data = [
        {'id': 1, 'text': 'Hello', 'label': 0},
        {'id': 2, 'text': 'World', 'label': 1},
        {'id': 3, 'text': 'Test', 'label': 0}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
        path = f.name
    
    yield path
    Path(path).unlink()


def test_inspect_csv(sample_csv):
    """Test inspecting a CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = inspect_file(sample_csv, verbose=False, output_dir=tmpdir)
        
        assert profile['shape']['rows'] == 5
        assert profile['shape']['cols'] == 3
        assert len(profile['columns']) == 3


def test_inspect_jsonl(sample_jsonl):
    """Test inspecting a JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = inspect_file(sample_jsonl, verbose=False, output_dir=tmpdir)
        
        assert profile['shape']['rows'] == 3
        assert profile['shape']['cols'] == 3


def test_inspect_without_output(sample_csv):
    """Test inspecting without saving output."""
    profile = inspect_file(sample_csv, verbose=False, output_dir=None)
    
    assert profile['shape']['rows'] == 5
    assert profile['shape']['cols'] == 3


def test_inspect_with_verbose(sample_csv):
    """Test inspecting with verbose output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = inspect_file(sample_csv, verbose=True, output_dir=tmpdir)
        
        assert 'statistics' in profile
        assert len(profile['statistics']) == 3
