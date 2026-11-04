"""Tests for synthetic world and fact generation."""

import pytest
from synthetic.world_generator import WorldGenerator, create_fictional_world


class TestWorldGenerator:
    """Tests for WorldGenerator."""
    
    def test_world_generation_unique_names(self):
        """Test that generated world names are unique."""
        generator = WorldGenerator()
        
        names = set()
        for i in range(10):
            world = generator.generate_world(seed=i)
            names.add(world['name'])
        
        # Most names should be unique
        assert len(names) >= 8
    
    def test_fact_generation多样性(self):
        """Test that generated facts are diverse."""
        generator = WorldGenerator()
        
        facts = set()
        for i in range(20):
            fact = generator.generate_fact(seed=i)
            facts.add(fact)
        
        # Most facts should be unique
        assert len(facts) >= 15
    
    def test_world_contains_required_fields(self):
        """Test that world has all required fields."""
        generator = WorldGenerator()
        world = generator.generate_world(seed=42)
        
        required_fields = ['name', 'theme', 'location', 'time_period', 'topic', 'description']
        for field in required_fields:
            assert field in world
            assert world[field] is not None


class TestWorldGeneratorEdgeCases:
    """Edge case tests for WorldGenerator."""
    
    def test_empty_config(self):
        """Test generator with minimal config."""
        from synthetic.config import WorldConfig
        config = WorldConfig()
        config.themes = ['test']
        config.locations = ['place']
        config.time_periods = ['era']
        config.topics = ['topic']
        
        generator = WorldGenerator(config)
        world = generator.generate_world(seed=42)
        
        assert world['theme'] == 'test'
        assert world['location'] == 'place'
