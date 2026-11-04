"""Tests for V2 scenario generator."""

import pytest
from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator


class TestScenarioGenerator:
    """Tests for ScenarioGenerator."""
    
    def test_generates_scenario(self):
        """Test that scenario is generated with all required fields."""
        config_manager = ConfigManager()
        generator = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        
        scenario = generator.generate_scenario(seed=42)
        
        assert 'scenario_id' in scenario
        assert 'scenario_family_id' in scenario
        assert 'scenario_family_name' in scenario
        assert 'topic' in scenario
        assert 'context' in scenario
        assert 'verified_facts' in scenario
        assert 'fact_attributes' in scenario
    
    def test_scenario_has_verified_facts(self):
        """Test that scenario has verified facts with attributes."""
        config_manager = ConfigManager()
        generator = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        
        scenario = generator.generate_scenario(seed=42)
        
        verified_facts = scenario['verified_facts']
        assert verified_facts
        assert len(verified_facts) > 0
        
        # Each attribute should have a value
        for attr, value in verified_facts.items():
            assert value is not None
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_scenario_family_reuse(self):
        """Test that scenario families can be reused."""
        config_manager = ConfigManager()
        generator = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        
        # Generate multiple scenarios
        scenarios = [generator.generate_scenario(seed=i*100) for i in range(10)]
        
        # Check we have scenario families
        family_ids = [s['scenario_family_id'] for s in scenarios]
        assert len(set(family_ids)) <= len(scenarios)


class TestFactFirstScenarioGeneration:
    """Tests for Priority 2: Fact-first scenario generation."""
    
    def test_verified_facts_before_conversation(self):
        """Test that verified facts exist before conversation generation."""
        config_manager = ConfigManager()
        generator = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        
        scenario = generator.generate_scenario(seed=42)
        
        # Verified facts should be generated
        assert 'verified_facts' in scenario
        
        # Each fact should be a valid string
        for attr, value in scenario['verified_facts'].items():
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_scenario_family_consistency(self):
        """Test that scenario family has consistent attributes."""
        config_manager = ConfigManager()
        generator = ScenarioGenerator(
            config_manager.world,
            config_manager.dataset
        )
        
        # Generate a scenario
        scenario = generator.generate_scenario(seed=42)
        
        family_id = scenario['scenario_family_id']
        family_name = scenario['scenario_family_name']
        
        # Verify we can retrieve family info
        families = generator.get_all_families()
        family_info = next((f for f in families if f['family_id'] == family_id), None)
        
        assert family_info is not None
        assert family_info['family_name'] == family_name


class TestScenarioFamilyCount:
    """Tests for scenario family requirements."""
    
    def test_eight_scenario_families(self):
        """Test that we have exactly 8 scenario families."""
        config_manager = ConfigManager()
        
        expected_families = [
            'college_event', 'travel_schedule', 'meeting',
            'restaurant_reservation', 'project_deadline', 'sports_result',
            'shopping', 'delivery'
        ]
        
        actual_families = [f.name for f in config_manager.world.scenario_families]
        
        assert len(actual_families) == 8, f"Expected 8 families, got {len(actual_families)}"
        
        for family in expected_families:
            assert family in actual_families
