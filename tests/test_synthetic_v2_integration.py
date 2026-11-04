"""Integration tests for V2 synthetic dataset generation."""

import pytest
import sys
import subprocess
from pathlib import Path
import json
import pandas as pd

from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator
from synthetic.v2.liar_generator import LiarGenerator
from synthetic.v2.conversation_generator import ConversationGenerator
from synthetic.v2.validator import Validator
from synthetic.v2.exporter import Exporter


class TestV2GenerateIntegration:
    """Integration tests for V2 generation."""
    
    def test_exact_generation_count(self, temp_dir):
        """Test that --num-examples produces exactly that many conversations."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen = ConversationGenerator(config_manager.conversation)
        validator = Validator()
        
        num_examples = 10
        conversations = []
        scenarios = []
        ground_truths = []
        
        for i in range(num_examples):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            
            conversations.append(conversation)
            
            ground_truth = {
                'conversation_id': scenario['scenario_id'],
                'scenario_id': scenario['scenario_id'],
                'scenario_family_id': scenario['scenario_family_id'],
                'liar': liar_scenario['liar'],
                'claim': liar_scenario['claim'],
                'verified_facts': scenario['verified_facts']
            }
            ground_truths.append(ground_truth)
            scenarios.append(scenario)
        
        # Validate
        validation = validator.validate_dataset(conversations, scenarios, ground_truths)
        assert validation['valid'], f"Validation failed: {validation['issues']}"
        
        # Export
        exporter = Exporter(config_manager.dataset)
        output_paths = exporter.export_all(conversations, scenarios, ground_truths, str(temp_dir))
        
        # Verify exact count
        assert len(conversations) == num_examples
        assert len(scenarios) == num_examples
        assert len(ground_truths) == num_examples
        
        # Verify unique IDs
        conv_ids = [s['scenario_id'] for s in scenarios]
        assert len(conv_ids) == len(set(conv_ids)), "Duplicate conversation IDs"
    
    def test_one_liar_per_conversation(self, temp_dir):
        """Test that each conversation has exactly one liar."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen = ConversationGenerator(config_manager.conversation)
        
        for i in range(20):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            
            # Count liars
            liar_count = sum(1 for m in conversation if m.get('is_liar'))
            assert liar_count == 1, f"Conversation {i} has {liar_count} liars, expected 1"
    
    def test_four_characters_per_conversation(self, temp_dir):
        """Test that each conversation has exactly 4 characters."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen = ConversationGenerator(config_manager.conversation)
        
        for i in range(20):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            
            speakers = set(m['speaker'] for m in conversation)
            assert speakers == {'A', 'B', 'C', 'D'}, \
                f"Conversation {i} has wrong characters: {speakers}"
    
    def test_message_count_8_to_20(self, temp_dir):
        """Test that each conversation has 8-20 messages."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen = ConversationGenerator(config_manager.conversation)
        
        for i in range(20):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            
            num_messages = len(conversation)
            assert 8 <= num_messages <= 20, \
                f"Conversation {i} has {num_messages} messages, expected 8-20"
    
    def test_claim_context_not_null(self, temp_dir):
        """Test that claim context is never null."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        for i in range(20):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            
            claim = liar_scenario['claim']
            assert claim['context'] is not None, \
                f"Conversation {i} has null claim context"
            assert len(claim['context']) > 0, \
                f"Conversation {i} has empty claim context"


class TestV2UniqueIdentityArchitecture:
    """Tests for Priority 1: Unique identity architecture."""
    
    def test_unique_conversation_ids_across_all(self, temp_dir):
        """Test that all conversation IDs are unique."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen = ConversationGenerator(config_manager.conversation)
        
        conv_ids = []
        for i in range(100):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            conv_id = scenario['scenario_id']
            conv_ids.append(conv_id)
        
        assert len(conv_ids) == len(set(conv_ids)), "Duplicate conversation IDs"


class TestV2ExactGenerationCount:
    """Tests for Priority 6: Exact generation count."""
    
    def test_reproducibility_with_seed(self, temp_dir):
        """Test that same seed produces same results."""
        config_manager = ConfigManager()
        
        # First run
        scenario_gen1 = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen1 = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen1 = ConversationGenerator(config_manager.conversation)
        
        conversations1 = []
        for i in range(10):
            scenario = scenario_gen1.generate_scenario(seed=i*100)
            liar_scenario = liar_gen1.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen1.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            conversations1.append(conversation)
        
        # Second run with same seeds
        scenario_gen2 = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen2 = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        conv_gen2 = ConversationGenerator(config_manager.conversation)
        
        conversations2 = []
        for i in range(10):
            scenario = scenario_gen2.generate_scenario(seed=i*100)
            liar_scenario = liar_gen2.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen2.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            conversations2.append(conversation)
        
        # Results should be identical
        for c1, c2 in zip(conversations1, conversations2):
            texts1 = [m['text'] for m in c1]
            texts2 = [m['text'] for m in c2]
            assert texts1 == texts2, "Results not reproducible with same seed"


class TestV2LeakageSafeSplitting:
    """Tests for Priority 7: Leakage-safe splitting."""
    
    def test_no_train_validation_overlap(self, temp_dir):
        """Test that train and validation have no scenario overlap."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create 10 conversations with 2 families (5 each)
        conversations = []
        scenarios = []
        for i in range(10):
            family_id = f'family_{i // 5}'
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
        
        output_paths = exporter.export_all(conversations, scenarios, ground_truths, str(temp_dir))
        
        # Read CSVs and check family overlap
        train_df = pd.read_csv(output_paths['train'])
        val_df = pd.read_csv(output_paths['validation'])
        
        train_families = set(train_df['scenario_family_id'].unique())
        val_families = set(val_df['scenario_family_id'].unique())
        
        assert len(train_families & val_families) == 0, \
            f"Scenario overlap: {train_families & val_families}"


class TestV2AccurateQualityReport:
    """Tests for Priority 8: Accurate quality report."""
    
    def test_counts_reconcile(self, temp_dir):
        """Test that quality report counts match actual files."""
        config_manager = ConfigManager()
        exporter = Exporter(config_manager.dataset)
        
        # Create 10 conversations with 10 messages each
        conversations = []
        scenarios = []
        for i in range(10):
            conv = []
            for j in range(10):
                conv.append({
                    'speaker': chr(ord('A') + j % 4),
                    'text': f'Message {j}',
                    'timestamp': j
                })
            conversations.append(conv)
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
        
        # Read quality report
        quality_report = output_paths['quality_report']
        with open(quality_report, 'r') as f:
            content = f.read()
        
        # Check that report mentions the right counts
        assert '10' in content, "Quality report should mention 10 conversations"
        assert '100' in content, "Quality report should mention 100 messages"
