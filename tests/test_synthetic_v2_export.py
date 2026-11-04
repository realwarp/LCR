"""Tests for V2 exporter."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator
from synthetic.v2.liar_generator import LiarGenerator
from synthetic.v2.conversation_generator import ConversationGenerator
from synthetic.v2.exporter import Exporter


class TestExporterV2:
    """Tests for Exporter V2."""
    
    def test_export_train_validation(self, temp_dir):
        """Test that train/validation CSVs are exported."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create test data
        conversations = [[
            {'speaker': 'A', 'text': 'Hello', 'timestamp': 0, 'is_liar': False},
            {'speaker': 'B', 'text': 'Hi', 'timestamp': 1, 'is_liar': False},
            {'speaker': 'C', 'text': 'Test', 'timestamp': 2, 'is_liar': False},
            {'speaker': 'D', 'text': 'Done', 'timestamp': 3, 'is_liar': True}
        ]]
        
        scenarios = [{
            'scenario_id': 'scenario_000001',
            'scenario_family_id': 'family_0001',
            'characters': ['A', 'B', 'C', 'D']
        }]
        
        ground_truths = [{
            'conversation_id': 'scenario_000001',
            'scenario_id': 'scenario_000001',
            'scenario_family_id': 'family_0001',
            'liar': 'D',
            'claim': {'attribute': 'date', 'true_value': '15 Dec', 'false_value': '16 Dec', 'context': 'Test'},
            'verified_facts': {'date': '15 Dec'}
        }]
        
        output_paths = exporter.export_train_validation(
            conversations, scenarios, ground_truths, str(temp_dir)
        )
        
        assert 'train' in output_paths
        assert 'validation' in output_paths
        assert output_paths['train'].exists()
        assert output_paths['validation'].exists()
    
    def test_export_ground_truth(self, temp_dir):
        """Test that ground truth JSONL is exported."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        ground_truths = [{
            'conversation_id': 'scenario_000001',
            'scenario_id': 'scenario_000001',
            'liar': 'A',
            'claim': {'attribute': 'date', 'true_value': '15 Dec', 'false_value': '16 Dec', 'context': 'Test'},
            'verified_facts': {'date': '15 Dec'}
        }]
        
        path = exporter.export_ground_truth(ground_truths, str(temp_dir))
        
        assert path.exists()
        
        # Check JSONL format
        with open(path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert 'conversation_id' in data
            assert 'claim' in data


class TestExportStatistics:
    """Tests for export statistics."""
    
    def test_export_stats_count(self):
        """Test that export stats track counts correctly."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        conversations = [
            [{'speaker': 'A', 'text': 'Hello', 'timestamp': 0}],
            [{'speaker': 'B', 'text': 'Hi', 'timestamp': 0}],
            [{'speaker': 'C', 'text': 'Test', 'timestamp': 0}]
        ]
        
        scenarios = [{'scenario_id': f'scenario_{i}'} for i in range(3)]
        ground_truths = [{'conversation_id': f'scenario_{i}'} for i in range(3)]
        
        output_paths = exporter.export_all(conversations, scenarios, ground_truths)
        
        stats = exporter.get_stats()
        
        assert stats['conversations_exported'] == 3
        assert stats['messages_exported'] == 3
        assert stats['ground_truth_records'] == 3
        assert stats['scenario_records'] == 3


class TestAccuracyCounts:
    """Tests for Priority 8: Accurate quality report."""
    
    def test_quality_report_counts_reconcile(self, temp_dir):
        """Test that quality report counts match file contents."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create test data
        conversations = []
        for i in range(5):
            conv = []
            for j in range(10):  # 10 messages each
                conv.append({
                    'speaker': chr(ord('A') + j % 4),
                    'text': f'Message {j}',
                    'timestamp': j
                })
            conversations.append(conv)
        
        scenarios = [{
            'scenario_id': f'scenario_{i}',
            'scenario_family_id': f'family_{i % 2}',
            'characters': ['A', 'B', 'C', 'D']
        } for i in range(5)]
        
        ground_truths = [{
            'conversation_id': f'scenario_{i}',
            'scenario_id': f'scenario_{i}',
            'liar': 'A',
            'claim': {'attribute': 'date', 'true_value': '15 Dec', 'false_value': '16 Dec', 'context': 'Test'},
            'verified_facts': {'date': '15 Dec'}
        } for i in range(5)]
        
        output_paths = exporter.export_all(conversations, scenarios, ground_truths, str(temp_dir))
        
        # Read CSVs and count rows
        train_df = pd.read_csv(output_paths['train'])
        val_df = pd.read_csv(output_paths['validation'])
        
        # Total messages should match
        total_csv_rows = len(train_df) + len(val_df)
        total_conversation_messages = sum(len(c) for c in conversations)
        
        assert total_csv_rows == total_conversation_messages, \
            f"CSV rows {total_csv_rows} != conversation messages {total_conversation_messages}"


class TestLeakageSafeSplitting:
    """Tests for Priority 7: Leakage-safe splitting."""
    
    def test_split_by_scenario_family(self, temp_dir):
        """Test that train/validation split is by scenario_family."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create conversations from 2 scenario families
        conversations = []
        scenarios = []
        for i in range(10):
            family_id = f'family_{i // 5}'  # 5 per family
            conversations.append([
                {'speaker': 'A', 'text': 'Hello', 'timestamp': j}
                for j in range(10)
            ])
            scenarios.append({
                'scenario_id': f'scenario_{i:06d}',
                'scenario_family_id': family_id,
                'characters': ['A', 'B', 'C', 'D']
            })
        
        ground_truths = [{
            'conversation_id': s['scenario_id'],
            'scenario_id': s['scenario_id'],
            'scenario_family_id': s['scenario_family_id'],
            'liar': 'A',
            'claim': {'attribute': 'date', 'true_value': '15 Dec', 'false_value': '16 Dec', 'context': 'Test'},
            'verified_facts': {'date': '15 Dec'}
        } for s in scenarios]
        
        output_paths = exporter.export_train_validation(
            conversations, scenarios, ground_truths, str(temp_dir)
        )
        
        # Read CSVs
        train_df = pd.read_csv(output_paths['train'])
        val_df = pd.read_csv(output_paths['validation'])
        
        # Get unique families in each split
        train_families = set(train_df['scenario_family_id'].unique())
        val_families = set(val_df['scenario_family_id'].unique())
        
        # No overlap
        assert len(train_families & val_families) == 0, \
            f"Scenario family overlap: {train_families & val_families}"


class TestExporterUniqueness:
    """Tests for Priority 1: Unique IDs in export."""
    
    def test_no_duplicate_conv_ids(self, temp_dir):
        """Test that exported data has no duplicate conversation IDs."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create conversations with unique IDs
        conversations = []
        scenarios = []
        for i in range(10):
            conversations.append([
                {'speaker': 'A', 'text': 'Hello', 'timestamp': j}
                for j in range(10)
            ])
            scenarios.append({
                'scenario_id': f'scenario_{i:06d}',
                'scenario_family_id': f'family_{i}',
                'characters': ['A', 'B', 'C', 'D']
            })
        
        ground_truths = [{
            'conversation_id': s['scenario_id'],
            'scenario_id': s['scenario_id'],
            'liar': 'A',
            'claim': {'attribute': 'date', 'true_value': '15 Dec', 'false_value': '16 Dec', 'context': 'Test'},
            'verified_facts': {'date': '15 Dec'}
        } for s in scenarios]
        
        output_paths = exporter.export_all(conversations, scenarios, ground_truths, str(temp_dir))
        
        # Read scenarios JSONL and check for duplicates
        scenarios_path = output_paths['scenarios']
        with open(scenarios_path, 'r') as f:
            lines = f.readlines()
            conv_ids = [json.loads(line)['scenario_id'] for line in lines]
        
        assert len(conv_ids) == len(set(conv_ids)), \
            f"Duplicate conversation IDs found: {len(conv_ids) - len(set(conv_ids))} duplicates"
