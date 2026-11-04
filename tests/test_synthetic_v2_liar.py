"""Tests for V2 liar generator."""

import pytest
from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator
from synthetic.v2.liar_generator import LiarGenerator


class TestLiarGeneratorV2:
    """Tests for LiarGenerator V2."""
    
    def test_selects_one_liar(self):
        """Test that exactly one liar is selected."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        assert 'liar' in liar_scenario
        assert liar_scenario['liar'] in ['A', 'B', 'C', 'D']
    
    def test_claim_has_context(self):
        """Test that claim has non-null context."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        claim = liar_scenario['claim']
        
        # Priority 3: Claim context must not be null
        assert 'context' in claim
        assert claim['context'] is not None
        assert len(claim['context']) > 0
    
    def test_claim_has_true_false_values(self):
        """Test that claim has true and false values."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        claim = liar_scenario['claim']
        
        assert 'attribute' in claim
        assert 'true_value' in claim
        assert 'false_value' in claim
        assert 'lie_type' in claim
        
        # True and false values should be different
        assert claim['true_value'] != claim['false_value']
    
    def test_attribute_values_from_verified_facts(self):
        """Test that attribute values come from verified facts."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        claim = liar_scenario['claim']
        
        # The attribute should exist in verified facts
        attr = claim['attribute']
        assert attr in scenario['verified_facts']
        
        # True value should match verified fact
        assert claim['true_value'] == scenario['verified_facts'][attr]


class TestFactGroundedLiarGeneration:
    """Tests for Priority 3: Fact-grounded liar generation."""
    
    def test_liar_modifies_verified_fact(self):
        """Test that liar modifies a verified fact."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        claim = liar_scenario['claim']
        attr = claim['attribute']
        
        # False value should be a variant of true value
        true_val = claim['true_value']
        false_val = claim['false_value']
        
        # They should be related (same attribute type)
        assert true_val != false_val
    
    def test_claim_attribute_is_verified_fact_attr(self):
        """Test that claim attribute is one of the verified fact attributes."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        claim = liar_scenario['claim']
        attr = claim['attribute']
        
        # Attribute should be in verified facts
        assert attr in scenario['verified_facts']
        assert attr in scenario['fact_attributes']


class TestGroundTruthFormat:
    """Tests for ground truth format."""
    
    def test_ground_truth_structure(self):
        """Test that ground truth has correct structure."""
        config_manager = ConfigManager()
        scenario_gen = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        liar_gen = LiarGenerator(
            config_manager.liar,
            config_manager.dataset
        )
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        # Build ground truth like exporter does
        ground_truth = {
            'conversation_id': scenario['scenario_id'],
            'scenario_id': scenario['scenario_id'],
            'liar': liar_scenario['liar'],
            'claim': liar_scenario['claim'],
            'verified_facts': scenario['verified_facts']
        }
        
        # Check required fields
        assert 'conversation_id' in ground_truth
        assert 'scenario_id' in ground_truth
        assert 'liar' in ground_truth
        assert 'claim' in ground_truth
        assert 'verified_facts' in ground_truth
        
        # Check claim structure
        claim = ground_truth['claim']
        assert 'attribute' in claim
        assert 'true_value' in claim
        assert 'false_value' in claim
        assert 'context' in claim
