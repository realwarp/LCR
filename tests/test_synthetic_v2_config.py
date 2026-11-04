"""Tests for V2 configuration."""

import pytest
from synthetic.v2.config import ConfigManager, WorldConfig, LiarConfig, DatasetConfig


class TestScenarioFamily:
    """Tests for scenario family configuration."""
    
    def test_has_eight_families(self):
        """Test that we have the required scenario families."""
        config = WorldConfig()
        
        family_names = [f.name for f in config.scenario_families]
        
        required_families = [
            'college_event', 'travel_schedule', 'meeting',
            'restaurant_reservation', 'project_deadline', 'sports_result',
            'shopping', 'delivery'
        ]
        
        for family in required_families:
            assert family in family_names, f"Missing family: {family}"
    
    def test_has_attribute_templates(self):
        """Test that families have attribute templates."""
        config = WorldConfig()
        
        for family in config.scenario_families:
            assert 'attribute_templates' in family.__dict__
            assert family.attribute_templates
            assert 'date' in family.attribute_templates
            assert 'time' in family.attribute_templates


class TestDatasetConfigV2:
    """Tests for DatasetConfig V2."""
    
    def test_unique_id_generation(self):
        """Test that IDs are generated uniquely."""
        config = DatasetConfig()
        config.next_conversation_id = 1
        config.next_scenario_id = 1
        
        conv_ids = [config.generate_conversation_id() for _ in range(10)]
        scenario_ids = [config.generate_scenario_id() for _ in range(10)]
        
        # Check uniqueness
        assert len(conv_ids) == len(set(conv_ids))
        assert len(scenario_ids) == len(set(scenario_ids))
        
        # Check format
        assert conv_ids[0] == 'conv_000001'
        assert scenario_ids[0] == 'scenario_000001'


class TestLiarConfigV2:
    """Tests for LiarConfig V2."""
    
    def test_lie_types_map_to_attributes(self):
        """Test that lie types map to attribute names."""
        config = LiarConfig()
        
        lie_types = config.lie_types
        
        # Check required lie types
        required_types = [
            'date_lie', 'time_lie', 'location_lie', 'topic_lie',
            'duration_lie', 'attendees_lie', 'budget_lie', 'quantity_lie'
        ]
        
        for lie_type in required_types:
            assert lie_type in lie_types, f"Missing lie type: {lie_type}"


class TestConfigManagerV2:
    """Tests for ConfigManager V2."""
    
    def test_to_dict_preserves_families(self):
        """Test that to_dict preserves scenario families."""
        config_manager = ConfigManager()
        
        config_dict = config_manager.to_dict()
        
        assert 'world' in config_dict
        assert 'scenario_families' in config_dict['world']


class TestUniqueIdentityArchitecture:
    """Tests for Priority 1: Unique identity architecture."""
    
    def test_conversation_id_format(self):
        """Test conversation ID format."""
        config = DatasetConfig()
        
        conv_id = config.generate_conversation_id()
        
        # Check format
        assert conv_id.startswith('conv_')
        assert len(conv_id) == 11  # conv_ + 6 digits
    
    def test_scenario_id_format(self):
        """Test scenario ID format."""
        config = DatasetConfig()
        
        scenario_id = config.generate_scenario_id()
        
        # Check format
        assert scenario_id.startswith('scenario_')
        assert len(scenario_id) == 15  # scenario_ (9) + 6 digits
    
    def test_scenario_family_id_format(self):
        """Test scenario family ID format."""
        config = DatasetConfig()
        
        family_id = config.generate_scenario_family_id()
        
        # Check format
        assert family_id.startswith('family_')
        assert len(family_id) == 11  # family_ (7) + 4 digits
