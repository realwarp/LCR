"""Tests for synthetic dataset generation package."""

import pytest
import random
import json
import pandas as pd
from pathlib import Path
import tempfile
import os

# Import the package modules
from synthetic.config import (
    ConfigManager, WorldConfig, LiarConfig,
    ConversationConfig, LLMConfig, DatasetConfig
)
from synthetic.world_generator import WorldGenerator, create_fictional_world
from synthetic.liar_generator import LiarGenerator, create_liar_scenario
from synthetic.conversation_generator import ConversationGenerator, generate_conversation_sample
from synthetic.validator import Validator, validate_conversation_sample
from synthetic.exporter import Exporter


class TestConfigManager:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config_manager = ConfigManager()
        
        assert config_manager.world.themes is not None
        assert len(config_manager.world.themes) > 0
        assert config_manager.liar.character_names == ['A', 'B', 'C', 'D']
        assert config_manager.conversation.min_messages == 8
        assert config_manager.conversation.max_messages == 20
        assert config_manager.llm.enabled is False
        assert config_manager.dataset.num_examples == 100
    
    def test_yaml_save_and_load(self):
        """Test saving and loading config from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            
            # Create and save config
            config_manager = ConfigManager()
            config_manager.save(str(config_path))
            
            # Verify file exists
            assert config_path.exists()
            
            # Load and verify
            loaded_manager = ConfigManager(str(config_path))
            loaded_manager.load_from_yaml(str(config_path))
            
            assert loaded_manager.dataset.num_examples == 100
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config_manager = ConfigManager()
        config_dict = config_manager.to_dict()
        
        assert 'world' in config_dict
        assert 'liar' in config_dict
        assert 'conversation' in config_dict
        assert 'llm' in config_dict
        assert 'dataset' in config_dict


class TestWorldGenerator:
    """Tests for world generation."""
    
    def test_generate_world(self):
        """Test world generation."""
        generator = WorldGenerator()
        world = generator.generate_world(seed=42)
        
        assert 'name' in world
        assert 'theme' in world
        assert 'location' in world
        assert 'time_period' in world
        assert 'topic' in world
        assert 'description' in world
    
    def test_generate_fact(self):
        """Test fact generation."""
        generator = WorldGenerator()
        fact = generator.generate_fact(seed=42)
        
        assert isinstance(fact, str)
        assert len(fact) > 0
    
    def test_generate_world_and_facts(self):
        """Test combined world and facts generation."""
        generator = WorldGenerator()
        result = generator.generate_world_and_facts(num_facts=5, seed=42)
        
        assert 'world' in result
        assert 'facts' in result
        assert len(result['facts']) == 5
    
    def test_get_all_worlds(self):
        """Test retrieval of all generated worlds."""
        generator = WorldGenerator()
        generator.generate_world(seed=42)
        generator.generate_world(seed=43)
        
        worlds = generator.get_all_worlds()
        assert len(worlds) == 2
    
    def test_convenience_function(self):
        """Test convenience function."""
        result = create_fictional_world(seed=42)
        
        assert 'world' in result
        assert 'facts' in result


class TestLiarGenerator:
    """Tests for liar generation."""
    
    def test_select_liar(self):
        """Test liar selection from characters."""
        generator = LiarGenerator()
        liar = generator.select_liar(['A', 'B', 'C', 'D'], seed=42)
        
        assert liar in ['A', 'B', 'C', 'D']
    
    def test_generate_false_claim(self):
        """Test false claim generation."""
        generator = LiarGenerator()
        claim = generator.generate_false_claim('A', seed=42)
        
        assert 'liar' in claim
        assert 'lie_type' in claim
        assert 'false_claim' in claim
        assert claim['liar'] == 'A'
    
    def test_generate_scenario(self):
        """Test complete scenario generation."""
        generator = LiarGenerator()
        scenario = generator.generate_scenario(seed=42)
        
        assert 'liar' in scenario
        assert 'characters' in scenario
        assert len(scenario['characters']) == 4
        assert scenario['liar'] in scenario['characters']
        assert 'claim' in scenario
    
    def test_convenience_function(self):
        """Test convenience function."""
        result = create_liar_scenario(seed=42)
        
        assert 'liar' in result
        assert 'characters' in result


class TestConversationGenerator:
    """Tests for conversation generation."""
    
    def test_generate_message(self):
        """Test single message generation."""
        generator = ConversationGenerator()
        msg = generator.generate_message(
            speaker='A',
            fact='The sky is blue',
            message_type='fact',
            seed=42
        )
        
        assert 'speaker' in msg
        assert 'message' in msg
        assert msg['speaker'] == 'A'
        assert len(msg['message']) > 0
    
    def test_generate_conversation(self):
        """Test full conversation generation."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        facts = ['The sky is blue', 'Water is wet']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=facts,
            liar='A',
            lie_claim='The sky is green',
            seed=42
        )
        
        assert len(conv) >= 8
        assert len(conv) <= 20
        
        # Check all speakers are valid characters
        speakers = set(msg['speaker'] for msg in conv)
        assert speakers == set(characters)
    
    def test_format_conversation_text(self):
        """Test text formatting."""
        generator = ConversationGenerator()
        conv = generator.generate_conversation(
            characters=['A', 'B'],
            facts=['Test fact'],
            liar='A',
            lie_claim='Test lie',
            seed=42
        )
        
        formatted = generator.format_conversation(conv, format_type='text')
        assert isinstance(formatted, str)
        assert '[A]:' in formatted or '[B]:' in formatted
    
    def test_convenience_function(self):
        """Test convenience function."""
        conv = generate_conversation_sample(
            characters=['A', 'B', 'C', 'D'],
            facts=['Test'],
            liar='A',
            lie_claim='Test lie',
            seed=42
        )
        
        assert len(conv) > 0


class TestValidator:
    """Tests for validation."""
    
    def test_validate_conversation_valid(self):
        """Test validation of valid conversation."""
        validator = Validator()
        
        # Create a conversation with exactly 4 characters and 10 messages
        # Messages must be at least 10 characters long
        conv = [
            {'speaker': 'A', 'message': 'Hello everyone!', 'timestamp': 0, 'is_liar': True},
            {'speaker': 'B', 'message': 'Hi there, good to see you all.', 'timestamp': 1, 'is_liar': False},
            {'speaker': 'C', 'message': 'Hey, how is everyone doing today?', 'timestamp': 2, 'is_liar': False},
            {'speaker': 'D', 'message': 'Hi there, I am doing well.', 'timestamp': 3, 'is_liar': False},
            {'speaker': 'A', 'message': 'How are you all doing today?', 'timestamp': 4, 'is_liar': True},
            {'speaker': 'B', 'message': 'I am fine and ready for the discussion.', 'timestamp': 5, 'is_liar': False},
            {'speaker': 'C', 'message': 'Good to hear, I hope we make progress.', 'timestamp': 6, 'is_liar': False},
            {'speaker': 'D', 'message': 'What about you all?', 'timestamp': 7, 'is_liar': False},
            {'speaker': 'A', 'message': 'I am doing great and excited.', 'timestamp': 8, 'is_liar': True},
            {'speaker': 'B', 'message': 'Nice, let us start the meeting.', 'timestamp': 9, 'is_liar': False},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        # Debug: print issues if validation fails
        if not result['valid']:
            print(f"Issues: {result['issues']}")
            print(f"Checks: {result['checks']}")
        
        assert result['valid'] is True
        assert result['checks']['four_characters'] is True
        assert result['checks']['one_liar'] is True
        assert result['checks']['message_count'] is True
    
    def test_validate_conversation_wrong_characters(self):
        """Test validation with wrong characters."""
        conv = [
            {'speaker': 'A', 'message': 'Test', 'is_liar': False},
            {'speaker': 'B', 'message': 'Test', 'is_liar': True}
        ]
        
        validator = Validator()
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='B'
        )
        
        assert result['valid'] is False
        assert result['checks']['four_characters'] is False
    
    def test_validate_conversation_no_ground_truth_leakage(self):
        """Test that ground truth is not leaked."""
        conv = [
            {'speaker': 'A', 'message': 'I am the liar', 'is_liar': True},
            {'speaker': 'B', 'message': 'A is lying', 'is_liar': False}
        ]
        
        validator = Validator()
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        assert result['valid'] is False
        assert result['checks']['no_ground_truth'] is False
    
    def test_validate_dataset(self):
        """Test dataset-level validation."""
        generator = ConversationGenerator()
        conversations = []
        scenarios = []
        
        for i in range(3):
            conv = [
                {'speaker': 'A', 'message': f'Hello everyone {i}', 'timestamp': 0, 'is_liar': True},
                {'speaker': 'B', 'message': f'Hi there good to see {i}', 'timestamp': 1, 'is_liar': False},
                {'speaker': 'C', 'message': f'Hey how are you today {i}', 'timestamp': 2, 'is_liar': False},
                {'speaker': 'D', 'message': f'Hi there doing well {i}', 'timestamp': 3, 'is_liar': False},
                {'speaker': 'A', 'message': f'How are you all doing {i}', 'timestamp': 4, 'is_liar': True},
                {'speaker': 'B', 'message': f'Good morning have a nice day {i}', 'timestamp': 5, 'is_liar': False},
                {'speaker': 'C', 'message': f'See you later thanks for coming {i}', 'timestamp': 6, 'is_liar': False},
                {'speaker': 'D', 'message': f'Have a wonderful day ahead {i}', 'timestamp': 7, 'is_liar': False},
            ]
            for msg in conv:
                msg['is_liar'] = (msg['speaker'] == 'A')
            
            conversations.append(conv)
            scenarios.append({
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D'],
                'world': {'name': f'World_{i}'},
                'claim': {
                    'false_claim': f'Lie {i}',
                    'lie_type': 'date_lie'
                }
            })
        
        validator = Validator()
        result = validator.validate_dataset(conversations, scenarios)
        
        # Debug: print issues if validation fails
        if not result['valid']:
            print(f"Issues: {result['issues']}")
            print(f"Statistics: {result['statistics']}")
        
        assert result['valid'] is True
        assert result['statistics']['total_conversations'] == 3
    
    def test_validate_scenario(self):
        """Test scenario validation."""
        validator = Validator()
        scenario = {
            'liar': 'A',
            'characters': ['A', 'B', 'C', 'D'],
            'claim': {
                'false_claim': 'The sky is green'
            },
            'world': {'name': 'Test World'}
        }
        
        result = validator.validate_scenario(scenario)
        
        assert result['valid'] is True
        assert result['checks']['four_characters'] is True
    
    def test_convenience_function(self):
        """Test convenience function."""
        generator = ConversationGenerator()
        conv = generator.generate_conversation(
            characters=['A', 'B', 'C', 'D'],
            facts=['Test'],
            liar='A',
            lie_claim='Test lie',
            seed=42
        )
        for msg in conv:
            msg['is_liar'] = (msg['speaker'] == 'A')
        
        result = validate_conversation_sample(conv, ['A', 'B', 'C', 'D'], 'A')
        
        assert 'valid' in result


class TestExporter:
    """Tests for dataset export."""
    
    def test_export_train_validation(self):
        """Test train/validation export."""
        generator = ConversationGenerator()
        conversations = []
        scenarios = []
        
        for i in range(10):
            conv = generator.generate_conversation(
                characters=['A', 'B', 'C', 'D'],
                facts=['Test'],
                liar='A',
                lie_claim='Test lie',
                seed=42 + i
            )
            for msg in conv:
                msg['is_liar'] = (msg['speaker'] == 'A')
            
            conversations.append(conv)
            scenarios.append({
                'world': {'name': f'World_{i}'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            output_paths = exporter.export_train_validation(
                conversations, scenarios, output_dir=tmpdir
            )
            
            assert 'train' in output_paths
            assert 'validation' in output_paths
            assert output_paths['train'].exists()
            assert output_paths['validation'].exists()
            
            # Verify CSV content
            train_df = pd.read_csv(output_paths['train'])
            assert 'text' in train_df.columns
            assert 'label' in train_df.columns
            assert 'character' in train_df.columns
    
    def test_export_ground_truth(self):
        """Test ground truth export."""
        scenarios = [{
            'world': {'name': 'Test'},
            'liar': 'A',
            'characters': ['A', 'B', 'C', 'D'],
            'claim': {'false_claim': 'Test lie'}
        }]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            path = exporter.export_ground_truth(scenarios, output_dir=tmpdir)
            
            assert path.exists()
            
            # Verify JSONL content
            with open(path, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1
            
            data = json.loads(lines[0])
            assert 'liar' in data
            assert data['liar'] == 'A'
    
    def test_export_quality_report(self):
        """Test quality report export."""
        generator = ConversationGenerator()
        conversations = []
        scenarios = []
        
        for i in range(5):
            conv = generator.generate_conversation(
                characters=['A', 'B', 'C', 'D'],
                facts=['Test'],
                liar='A',
                lie_claim='Test lie',
                seed=42 + i
            )
            for msg in conv:
                msg['is_liar'] = (msg['speaker'] == 'A')
            
            conversations.append(conv)
            scenarios.append({
                'world': {'name': f'World_{i}'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D']
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            path = exporter.export_quality_report(
                conversations, scenarios, output_dir=tmpdir
            )
            
            assert path.exists()
            
            # Verify Markdown content
            with open(path, 'r') as f:
                content = f.read()
            assert '# Synthetic Dataset Quality Report' in content
            assert '## Overview' in content
            assert '## Lie Type Distribution' in content
    
    def test_export_all(self):
        """Test complete export."""
        generator = ConversationGenerator()
        conversations = []
        scenarios = []
        
        for i in range(5):
            conv = generator.generate_conversation(
                characters=['A', 'B', 'C', 'D'],
                facts=['Test'],
                liar='A',
                lie_claim='Test lie',
                seed=42 + i
            )
            for msg in conv:
                msg['is_liar'] = (msg['speaker'] == 'A')
            
            conversations.append(conv)
            scenarios.append({
                'world': {'name': f'World_{i}'},
                'liar': 'A',
                'characters': ['A', 'B', 'C', 'D'],
                'claim': {
                    'false_claim': f'Lie {i}',
                    'lie_type': 'date_lie'
                }
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Exporter()
            output_paths = exporter.export_all(
                conversations, scenarios, output_dir=tmpdir
            )
            
            # Exporter should return exactly 5 paths: train, validation, ground_truth, scenarios, quality_report
            assert len(output_paths) == 5
            assert 'train' in output_paths
            assert 'validation' in output_paths
            assert 'ground_truth' in output_paths
            assert 'scenarios' in output_paths
            assert 'quality_report' in output_paths
            
            for path in output_paths.values():
                assert path.exists()


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_seed_produces_same_results(self):
        """Test that same seed produces same results."""
        generator = LiarGenerator()
        
        result1 = generator.generate_scenario(seed=42)
        result2 = generator.generate_scenario(seed=42)
        
        assert result1['liar'] == result2['liar']
        assert result1['claim']['false_claim'] == result2['claim']['false_claim']
    
    def test_seed_produces_different_results(self):
        """Test that different seeds produce different results."""
        generator = LiarGenerator()
        
        results = []
        for seed in [42, 43, 44]:
            result = generator.generate_scenario(seed=seed)
            results.append(result)
        
        # At least some should differ
        assert any(
            results[0]['liar'] != results[i]['liar']
            for i in range(1, 3)
        )


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_generation_pipeline(self):
        """Test complete generation pipeline."""
        from synthetic.generate import generate_synthetic_dataset
        
        result = generate_synthetic_dataset(
            num_examples=10,
            output_dir=None,
            random_seed=42,
            llm_enabled=False
        )
        
        # Should generate at least some valid conversations
        assert result['num_conversations'] > 0
        assert result['num_scenarios'] > 0
        assert result['validation']['valid'] is True
        
        for path in result['output_paths'].values():
            assert Path(path).exists()
        
        assert result['validation']['valid'] is True
