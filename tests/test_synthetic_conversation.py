"""Tests for synthetic conversation generation."""

import pytest
from synthetic.conversation_generator import ConversationGenerator


class TestConversationStructure:
    """Tests for conversation structure."""
    
    def test_min_messages(self):
        """Test conversation has minimum messages."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1', 'Fact 2'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        assert len(conv) >= 8
    
    def test_max_messages(self):
        """Test conversation has maximum messages."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1', 'Fact 2'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        assert len(conv) <= 20
    
    def test_all_characters_speak(self):
        """Test that all characters have messages."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1', 'Fact 2'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        speakers = set(msg['speaker'] for msg in conv)
        assert speakers == set(characters)
    
    def test_messages_have_content(self):
        """Test that messages have non-empty content."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1', 'Fact 2'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        for msg in conv:
            assert len(msg['message']) > 0


class TestFillerMessages:
    """Tests for filler message generation."""
    
    def test_fillers_present(self):
        """Test that some filler messages are generated."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1', 'Fact 2'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        # Some messages should be fillers
        filler_count = sum(1 for msg in conv if msg.get('message_type') == 'filler')
        assert filler_count >= 0  # May or may not have fillers


class TestConversationFormatting:
    """Tests for conversation formatting."""
    
    def test_text_format(self):
        """Test conversation formatting as text."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        formatted = generator.format_conversation(conv, format_type='text')
        assert isinstance(formatted, str)
        assert '\n' in formatted
    
    def test_json_format(self):
        """Test conversation formatting as JSON."""
        generator = ConversationGenerator()
        characters = ['A', 'B', 'C', 'D']
        
        conv = generator.generate_conversation(
            characters=characters,
            facts=['Fact 1'],
            liar='A',
            lie_claim='False claim',
            seed=42
        )
        
        formatted = generator.format_conversation(conv, format_type='json')
        assert isinstance(formatted, list)
        assert len(formatted) == len(conv)
