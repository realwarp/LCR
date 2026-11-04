"""Tests for V2 validator."""

import pytest
from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator
from synthetic.v2.liar_generator import LiarGenerator
from synthetic.v2.conversation_generator import ConversationGenerator
from synthetic.v2.validator import Validator


class TestValidatorV2:
    """Tests for Validator V2."""
    
    def test_valid_conversation_passes(self):
        """Test that a valid conversation passes validation."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=['A', 'B', 'C', 'D'],
            expected_liar=liar_scenario['liar']
        )
        
        assert validation['valid'], f"Validation failed: {validation['issues']}"
    
    def test_message_count_validation(self):
        """Test that message count is validated."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=['A', 'B', 'C', 'D'],
            expected_liar=liar_scenario['liar']
        )
        
        # Should pass message count check
        assert validation['checks']['message_count'], \
            f"Message count failed: {validation['issues']}"


class TestValidationChecks:
    """Tests for validation checks."""
    
    def test_four_characters_check(self):
        """Test that 4 character check works."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=['A', 'B', 'C', 'D'],
            expected_liar=liar_scenario['liar']
        )
        
        assert validation['checks']['four_characters'], "Four character check failed"
    
    def test_one_liar_check(self):
        """Test that one liar check works."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=['A', 'B', 'C', 'D'],
            expected_liar=liar_scenario['liar']
        )
        
        assert validation['checks']['one_liar'], f"One liar check failed: {validation['issues']}"
    
    def test_no_ground_truth_leakage_check(self):
        """Test that ground truth leakage is checked."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=['A', 'B', 'C', 'D'],
            expected_liar=liar_scenario['liar']
        )
        
        assert validation['checks']['no_ground_truth'], \
            f"Ground truth leakage found: {validation['issues']}"


class TestDatasetValidation:
    """Tests for dataset-level validation."""
    
    def test_dataset_validation_passes(self):
        """Test that valid dataset passes validation."""
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
        
        # Generate 10 conversations
        conversations = []
        scenarios = []
        ground_truths = []
        
        for i in range(10):
            scenario = scenario_gen.generate_scenario(seed=i*100)
            liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=i*100+1)
            conversation = conv_gen.generate_conversation(
                scenario=scenario,
                liar_scenario=liar_scenario,
                seed=i*100+2
            )
            conversations.append(conversation)
            
            # Build ground truth
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
        
        validation = validator.validate_dataset(
            conversations, scenarios, ground_truths
        )
        
        assert validation['valid'], f"Dataset validation failed: {validation['issues']}"
