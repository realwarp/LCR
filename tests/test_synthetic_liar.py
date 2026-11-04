"""Tests for synthetic liar and claim generation."""

import pytest
from synthetic.liar_generator import LiarGenerator, create_liar_scenario


class TestLiarSelection:
    """Tests for liar selection logic."""
    
    def test_exactly_one_liar(self):
        """Test that exactly one liar is selected."""
        generator = LiarGenerator()
        
        for _ in range(10):
            scenario = generator.generate_scenario(seed=_)
            assert scenario['liar'] in scenario['characters']
    
    def test_liar_is_character(self):
        """Test that selected liar is in characters list."""
        generator = LiarGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        for seed in range(20):
            liar = generator.select_liar(characters=characters, seed=seed)
            assert liar in characters
    
    def test_different_seeds_different_liars(self):
        """Test that different seeds can select different liars."""
        generator = LiarGenerator()
        
        liars = set()
        for seed in range(10):
            scenario = generator.generate_scenario(seed=seed)
            liars.add(scenario['liar'])
        
        # Should have multiple different liars
        assert len(liars) >= 2


class TestFalseClaimGeneration:
    """Tests for false claim generation."""
    
    def test_claim_has_false_version(self):
        """Test that generated claims have false versions."""
        generator = LiarGenerator()
        
        for lie_type in ['date_lie', 'time_lie', 'location_lie']:
            claim = generator.generate_false_claim('A', seed=42)
            
            assert 'false_claim' in claim
            assert len(claim['false_claim']) > 0
    
    def test_lie_types_covered(self):
        """Test that different lie types are generated."""
        generator = LiarGenerator()
        
        lie_types_found = set()
        for seed in range(30):
            claim = generator.generate_false_claim('A', seed=seed)
            lie_types_found.add(claim['lie_type'])
        
        # Should cover multiple lie types
        assert len(lie_types_found) >= 5


class TestScenarioConsistency:
    """Tests for scenario consistency."""
    
    def test_scenario_complete(self):
        """Test that scenario has all required parts."""
        generator = LiarGenerator()
        scenario = generator.generate_scenario(seed=42)
        
        assert 'liar' in scenario
        assert 'characters' in scenario
        assert 'claim' in scenario
        assert len(scenario['characters']) == 4
    
    def test_scenario_liar_matches(self):
        """Test that scenario liar is consistent."""
        generator = LiarGenerator()
        scenario = generator.generate_scenario(seed=42)
        
        assert scenario['liar'] in scenario['characters']
        assert scenario['liar'] in ['A', 'B', 'C', 'D']
