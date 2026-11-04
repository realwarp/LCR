"""Conversation template generator."""

import random
from typing import List, Dict, Tuple
from .config import ConversationConfig


class ConversationGenerator:
    """Generates conversation templates based on prompts."""
    
    def __init__(self, config: ConversationConfig = None):
        self.config = config or ConversationConfig()
        self._templates = self._build_templates()
    
    def _build_templates(self) -> Dict[str, List[str]]:
        """Build conversation templates based on scenario types."""
        return {
            'scenario introductions': [
                "I'm {name}, and I'm here to discuss {topic}.",
                "Welcome everyone! I'm {name} from {group}.",
                "Thanks for gathering us, {name} speaking.",
                "Hello, I'm {name}. I have some thoughts about {topic}."
            ],
            'fact presentation': [
                "Based on what I've read, {fact}.",
                "I remember clearly that {fact}.",
                "According to my research, {fact}.",
                "It's well-known that {fact}.",
                "The evidence shows that {fact}.",
                "Surely you know that {fact}.",
                "Everyone agrees that {fact}.",
                "I'm certain that {fact}.",
                "Let me tell you about {fact}.",
                "Have you heard that {fact}?"
            ],
            'questioning': [
                "Can you explain why {fact}?",
                "How do you know that {fact}?",
                "Are you sure about {fact}?",
                "Where did you hear about {fact}?",
                "Can you prove that {fact}?",
                "Why do you think {fact}?"
            ],
            'agreement': [
                "I agree, {fact} makes sense.",
                "Yes, I've also observed that {fact}.",
                "That aligns with my understanding of {fact}.",
                "I think you're right about {fact}.",
                "Indeed, {fact} is well-documented."
            ],
            'disagreement': [
                "I disagree, {fact} doesn't seem right.",
                "I'm not convinced about {fact}.",
                "That contradicts what I know about {fact}.",
                "I have doubts about {fact}.",
                "Let's reconsider {fact}."
            ],
            'filler': [
                "Let me think about that.",
                "I need a moment to consider.",
                "That's an interesting point.",
                "I hadn't thought of it that way.",
                "Could you elaborate?",
                "I see what you mean.",
                "Interesting.",
                "I understand.",
                "Go on.",
                "Thank you for sharing."
            ],
            'sign-offs': [
                "I'll share my notes with everyone.",
                "Let's schedule a follow-up.",
                "I'll review the materials again.",
                "Thanks for the discussion.",
                "I look forward to your feedback."
            ]
        }
    
    def generate_message(
        self,
        speaker: str,
        fact: str,
        message_type: str = 'fact',
        seed: int = None
    ) -> Dict:
        """Generate a single conversation message."""
        rng = random.Random(seed)
        
        templates = self._templates.get(message_type, self._templates['fact presentation'])
        template = rng.choice(templates)
        
        # Handle placeholders
        message = template.format(name=speaker, fact=fact)
        
        return {
            'speaker': speaker,
            'message': message,
            'message_type': message_type,
            'fact': fact,
            'timestamp': None
        }
    
    def generate_conversation(
        self,
        characters: List[str],
        facts: List[str],
        liar: str,
        lie_claim: str,
        seed: int = None
    ) -> List[Dict]:
        """Generate a complete conversation."""
        rng = random.Random(seed)
        
        num_messages = rng.randint(
            self.config.min_messages,
            self.config.max_messages
        )
        
        conversation = []
        
        # Distribute messages among characters
        character_sequence = []
        for i in range(num_messages):
            character_sequence.append(rng.choice(characters))
        
        # Build conversation with balanced structure
        message_types = ['fact', 'questioning', 'agreement', 'disagreement']
        
        for i, speaker in enumerate(character_sequence):
            # Determine message content
            if speaker == liar:
                # Liar uses the false claim
                fact = lie_claim
                message_type = 'fact'
            else:
                # Truth-tellers use actual facts
                fact = rng.choice(facts)
                message_type = rng.choice(message_types)
            
            # Add filler occasionally
            if rng.random() < self.config.filler_probability:
                filler_templates = self._templates['filler']
                filler = rng.choice(filler_templates)
                conversation.append({
                    'speaker': speaker,
                    'message': filler,
                    'message_type': 'filler',
                    'fact': None,
                    'timestamp': i
                })
            
            # Add main message
            main_msg = self.generate_message(
                speaker=speaker,
                fact=fact,
                message_type=message_type,
                seed=rng.randint(0, 10000)
            )
            main_msg['timestamp'] = i
            conversation.append(main_msg)
        
        return conversation
    
    def format_conversation(
        self,
        conversation: List[Dict],
        format_type: str = 'text'
    ) -> str:
        """Format conversation for output."""
        if format_type == 'text':
            lines = []
            for msg in conversation:
                lines.append(f"[{msg['speaker']}]: {msg['message']}")
            return '\n'.join(lines)
        elif format_type == 'json':
            return conversation
        else:
            raise ValueError(f"Unknown format type: {format_type}")


def generate_conversation_sample(
    characters: List[str],
    facts: List[str],
    liar: str,
    lie_claim: str,
    seed: int = None
) -> List[Dict]:
    """Convenience function to generate a conversation."""
    generator = ConversationGenerator()
    return generator.generate_conversation(
        characters=characters,
        facts=facts,
        liar=liar,
        lie_claim=lie_claim,
        seed=seed
    )
