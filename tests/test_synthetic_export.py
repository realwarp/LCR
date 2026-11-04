"""Tests for synthetic dataset export."""

import pytest
import json
import pandas as pd
from pathlib import Path
import tempfile

from synthetic.exporter import Exporter


class TestTrainValidationExport:
    """Tests for train/validation export."""
    
    def test_exports_to_csv(self):
        """Test that data is exported to CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            conversations = [[
                {'speaker': 'A', 'message': 'Hello', 'timestamp': 0, 'is_liar': True},
                {'speaker': 'B', 'message': 'Hi', 'timestamp': 1, 'is_liar': False},
                {'speaker': 'C', 'message': 'Hey', 'timestamp': 2, 'is_liar': False},
                {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3, 'is_liar': False},
            ]]
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            }]
            
            output_paths = exporter.export_train_validation(
                conversations, scenarios, output_dir=tmpdir
            )
            
            assert Path(output_paths['train']).exists()
            assert Path(output_paths['validation']).exists()
    
    def test_csv_has_required_columns(self):
        """Test that CSV has required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            conversations = [[
                {'speaker': 'A', 'message': 'Hello', 'timestamp': 0, 'is_liar': True},
                {'speaker': 'B', 'message': 'Hi', 'timestamp': 1, 'is_liar': False},
                {'speaker': 'C', 'message': 'Hey', 'timestamp': 2, 'is_liar': False},
                {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3, 'is_liar': False},
            ]]
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            }]
            
            output_paths = exporter.export_train_validation(
                conversations, scenarios, output_dir=tmpdir
            )
            
            train_df = pd.read_csv(output_paths['train'])
            
            assert 'text' in train_df.columns
            assert 'label' in train_df.columns
            assert 'character' in train_df.columns
    
    def test_balanced_split(self):
        """Test that train/validation split is balanced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            conversations = []
            scenarios = []
            
            for i in range(20):
                conv = [
                    {'speaker': 'A', 'message': f'Hello {i}', 'timestamp': 0, 'is_liar': True},
                    {'speaker': 'B', 'message': f'Hi {i}', 'timestamp': 1, 'is_liar': False},
                    {'speaker': 'C', 'message': f'Hey {i}', 'timestamp': 2, 'is_liar': False},
                    {'speaker': 'D', 'message': f'Hi there {i}', 'timestamp': 3, 'is_liar': False},
                ]
                conversations.append(conv)
                
                scenarios.append({
                    'world': {'name': f'World_{i}'},
                    'liar': 'A',
                    'characters': ['A', 'B', 'C', 'D']
                })
            
            output_paths = exporter.export_train_validation(
                conversations, scenarios, output_dir=tmpdir
            )
            
            train_df = pd.read_csv(output_paths['train'])
            val_df = pd.read_csv(output_paths['validation'])
            
            # Should have both train and val splits
            assert len(train_df) > 0
            assert len(val_df) > 0


class TestGroundTruthExport:
    """Tests for ground truth export."""
    
    def test_exports_to_jsonl(self):
        """Test that ground truth is exported to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D'],
                'claim': {'false_claim': 'Test lie'}
            }]
            
            path = exporter.export_ground_truth(scenarios, output_dir=tmpdir)
            
            assert Path(path).exists()
            
            with open(path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data['liar'] == 'A'
    
    def test_ground_truth_separated(self):
        """Test that ground truth doesn't contain model input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D'],
                'claim': {'false_claim': 'Test lie'}
            }]
            
            path = exporter.export_ground_truth(scenarios, output_dir=tmpdir)
            
            with open(path, 'r') as f:
                lines = f.readlines()
            
            data = json.loads(lines[0])
            
            # Ground truth should have liar info, not text
            assert 'liar' in data
            assert 'text' not in data


class TestQualityReport:
    """Tests for quality report export."""
    
    def test_exports_to_markdown(self):
        """Test that report is exported to Markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            conversations = [[
                {'speaker': 'A', 'message': 'Hello', 'timestamp': 0, 'is_liar': True},
                {'speaker': 'B', 'message': 'Hi', 'timestamp': 1, 'is_liar': False},
                {'speaker': 'C', 'message': 'Hey', 'timestamp': 2, 'is_liar': False},
                {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3, 'is_liar': False},
            ]]
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            }]
            
            path = exporter.export_quality_report(
                conversations, scenarios, output_dir=tmpdir
            )
            
            assert Path(path).exists()
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert '# Synthetic Dataset Quality Report' in content
            assert '## Overview' in content
    
    def test_report_contains_statistics(self):
        """Test that report contains statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            
            conversations = [[
                {'speaker': 'A', 'message': 'Hello', 'timestamp': 0, 'is_liar': True},
                {'speaker': 'B', 'message': 'Hi', 'timestamp': 1, 'is_liar': False},
                {'speaker': 'C', 'message': 'Hey', 'timestamp': 2, 'is_liar': False},
                {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3, 'is_liar': False},
            ]]
            
            scenarios = [{
                'world': {'name': 'TestWorld'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            }]
            
            path = exporter.export_quality_report(
                conversations, scenarios, output_dir=tmpdir
            )
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert 'Total Conversations' in content
            assert 'Total Messages' in content
            assert 'Average Messages' in content
