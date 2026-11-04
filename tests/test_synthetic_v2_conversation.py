"""Tests for V2 conversation generator."""

import pytest
from synthetic.v2.config import ConfigManager
from synthetic.v2.scenario_generator import ScenarioGenerator
from synthetic.v2.liar_generator import LiarGenerator
from synthetic.v2.conversation_generator import ConversationGenerator


class TestConversationGeneratorV2:
    """Tests for ConversationGenerator V2."""
    
    def test_four_characters(self):
        """Test that conversation has exactly 4 characters."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        speakers = set(msg['speaker'] for msg in conversation)
        assert speakers == {'A', 'B', 'C', 'D'}
    
    def test_message_count_range(self):
        """Test that message count is within range."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        num_messages = len(conversation)
        assert 8 <= num_messages <= 20, f"Message count {num_messages} outside range [8, 20]"
    
    def test_liar_makes_false_claim(self):
        """Test that liar makes at least one false claim."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        liar = liar_scenario['liar']
        liar_messages = [m for m in conversation if m['speaker'] == liar]
        
        # At least one liar message should be a false claim
        has_false_claim = any(
            m.get('fact_type') == 'false_claim' or m.get('is_liar_message')
            for m in liar_messages
        )
        assert has_false_claim, "Liar does not make a false claim"
    
    def test_non_liars_consistent(self):
        """Test that non-liars only use verified facts."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        verified_facts = scenario['verified_facts']
        liar = liar_scenario['liar']
        
        # Non-liar messages should reference verified facts
        for msg in conversation:
            if msg['speaker'] != liar:
                # Non-liar should not make false claims
                assert not msg.get('is_liar_message'), f"Non-liar {msg['speaker']} makes false claim"


class TestCoherentConversationGeneration:
    """Tests for Priority 4: Coherent conversation generation."""
    
    def test_all_characters_speak(self):
        """Test that every character speaks at least once."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        speakers = set(msg['speaker'] for msg in conversation)
        assert speakers == {'A', 'B', 'C', 'D'}, "Not all characters speak"
    
    def test_no_unrelated_factual_claims(self):
        """Test that conversation doesn't have unrelated factual claims."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        # All messages should reference verified facts or be fillers
        for msg in conversation:
            fact_type = msg.get('fact_type', 'unknown')
            assert fact_type in ['verified', 'false_claim', 'context', 'filler'], \
                f"Message has unknown fact type: {fact_type}"
    
    def test_message_content_length(self):
        """Test that messages have sufficient content."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        # All messages should be > 10 characters
        for msg in conversation:
            text = str(msg.get('text', ''))
            assert len(text.strip()) > 10, f"Message too short: {text}"


class TestLiarDistribution:
    """Tests for Priority 5: Label semantics."""
    
    def test_one_positive_character_per_conversation(self):
        """Test that exactly one character has positive label."""
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
        
        scenario = scenario_gen.generate_scenario(seed=42)
        liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=42)
        
        conversation = conv_gen.generate_conversation(
            scenario=scenario,
            liar_scenario=liar_scenario,
            seed=42
        )
        
        # Exactly one character should have is_liar=True
        liar = liar_scenario['liar']
        
        for msg in conversation:
            is_liar = msg.get('is_liar', False)
            expected = (msg['speaker'] == liar)
            assert is_liar == expected, \
                f"Mismatch: {msg['speaker']} is_liar={is_liar}, expected={expected}"
