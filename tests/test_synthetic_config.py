"""Tests for synthetic dataset configuration."""

import pytest
import yaml
from pathlib import Path
import tempfile

from synthetic.config import ConfigManager, WorldConfig, LiarConfig


class TestWorldConfig:
    """Tests for WorldConfig."""
    
    def test_default_values(self):
        """Test default theme values."""
        config = WorldConfig()
        
        # Check that 'mystery' appears in at least one theme (as substring)
        has_mystery = any('mystery' in theme for theme in config.themes)
        assert has_mystery is True
        
        # Check that 'library' appears in at least one location (as substring)
        has_library = any('library' in location for location in config.locations)
        assert has_library is True
        
        # Check that time_periods contains era-related values
        has_era = any('era' in period or '1920s' in period for period in config.time_periods)
        assert has_era is True


class TestLiarConfig:
    """Tests for LiarConfig."""
    
    def test_default_characters(self):
        """Test default character names."""
        config = LiarConfig()
        
        assert config.character_names == ['A', 'B', 'C', 'D']
        assert 'date_lie' in config.lie_types
        assert 'location_lie' in config.lie_types


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""
    
    def test_load_partial_yaml(self):
        """Test loading YAML with partial config."""
        yaml_content = """
dataset:
  num_examples: 200
llm:
  enabled: true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name
        
        try:
            manager = ConfigManager()
            manager.load_from_yaml(temp_path)
            
            assert manager.dataset.num_examples == 200
            assert manager.llm.enabled is True
            # Default values should still exist for other fields
            assert len(manager.world.themes) > 0
        finally:
            Path(temp_path).unlink()
