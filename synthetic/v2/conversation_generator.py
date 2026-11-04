"""V2 conversation generator with controlled templates."""

import random
from typing import List, Dict
from .config import ConversationConfig


class ConversationGenerator:
    """Generates coherent conversations based on verified facts."""
    
    def __init__(self, config: ConversationConfig = None):
        self.config = config or ConversationConfig()
        self._generations: List[Dict] = []
    
    def generate_conversation(self, scenario: Dict, liar_scenario: Dict, 
                              seed: int = None) -> List[Dict]:
        """Generate a conversation based on scenario facts."""
        rng = random.Random(seed)
        
        characters = liar_scenario['characters']
        liar = liar_scenario['liar']
        claim = liar_scenario['claim']
        
        # Extract verified facts
        verified_facts = scenario.get('verified_facts', {})
        
        # Build message pool based on facts
        messages = self._build_message_pool(
            verified_facts=verified_facts,
            claim=claim,
            liar=liar,
            characters=characters
        )
        
        # Select messages to form conversation
        conversation = self._select_messages(messages, characters, rng)
        
        # Add filler messages if needed
        conversation = self._add_fillers(conversation, characters, rng)
        
        # Ensure message count is within bounds
        conversation = self._adjust_message_count(conversation, rng)
        
        # Mark liar
        for msg in conversation:
            msg['is_liar'] = (msg['speaker'] == liar)
        
        self._generations.append({
            'conversation': conversation,
            'scenario_id': scenario['scenario_id'],
            'liar': liar
        })
        
        return conversation
    
    def _build_message_pool(self, verified_facts: Dict, claim: Dict, 
                           liar: str, characters: List[str]) -> List[Dict]:
        """Build a pool of possible messages based on facts."""
        messages = []
        
        attr = claim['attribute']
        true_value = claim['true_value']
        false_value = claim['false_value']
        lie_type = claim['lie_type']
        context = claim['context']
        
        # Truth-tellers use verified facts
        for attr_name, value in verified_facts.items():
            templates = [
                f"I remember the {attr_name} is {value}.",
                f"The {attr_name} is definitely {value}.",
                f"According to our planning, {attr_name} is {value}.",
                f"We confirmed the {attr_name} is {value}.",
                f"The schedule shows {attr_name} as {value}."
            ]
            for template in templates:
                for speaker in characters:
                    if speaker != liar:
                        messages.append({
                            'speaker': speaker,
                            'text': template,
                            'fact_type': 'verified',
                            'fact_attr': attr_name,
                            'fact_value': value,
                            'is_liar': False
                        })
        
        # Liar uses the false claim
        liar_templates = [
            f"I'm sure the {attr} is {false_value}.",
            f"The {attr} should be {false_value}.",
            f"I recall the {attr} is {false_value}.",
            f"I read that the {attr} is {false_value}.",
            f"I believe the {attr} is {false_value}."
        ]
        for template in liar_templates:
            messages.append({
                'speaker': liar,
                'text': template,
                'fact_type': 'false_claim',
                'fact_attr': attr,
                'fact_value': false_value,
                'is_liar_message': True,
                'is_liar': True  # Add is_liar flag directly
            })
        
        # Context statement
        context_templates = [
            f"We need to finalize the {attr} for the {context}.",
            f"The {attr} is important for our planning.",
            f"We must confirm the {attr} soon.",
            f"Let's verify the {attr} together.",
            f"The {attr} needs to be set."
        ]
        for template in context_templates:
            for speaker in characters:
                messages.append({
                    'speaker': speaker,
                    'text': template,
                    'fact_type': 'context',
                    'fact_attr': attr,
                    'fact_value': context,
                    'is_liar': (speaker == liar)  # Liar context has is_liar
                })
        
        return messages
    
    def _select_messages(self, messages: List[Dict], characters: List[str],
                        rng: random.Random) -> List[Dict]:
        """Select messages to form a coherent conversation."""
        conversation = []
        
        # Ensure each character speaks at least once
        for char in characters:
            char_messages = [m for m in messages if m['speaker'] == char]
            if char_messages:
                msg = rng.choice(char_messages)
                conversation.append(msg)
                messages.remove(msg)
        
        # Add more messages to reach minimum
        min_messages = self.config.min_messages
        while len(conversation) < min_messages and messages:
            # Try to alternate speakers
            current_speakers = [m['speaker'] for m in conversation]
            for char in characters:
                char_messages = [m for m in messages if m['speaker'] == char]
                if char_messages:
                    msg = rng.choice(char_messages)
                    conversation.append(msg)
                    messages.remove(msg)
                    break
            
            if len(conversation) >= min_messages:
                break
        
        return conversation
    
    def _add_fillers(self, conversation: List[Dict], characters: List[str],
                    rng: random.Random) -> List[Dict]:
        """Add filler messages if needed."""
        if rng.random() > self.config.filler_probability:
            return conversation
        
        # Add a few filler messages
        fillers = [
            "Let's think about this.",
            "I need a moment.",
            "That's an interesting point.",
            "Could you clarify?",
            "I see.",
            "Interesting.",
            "I understand.",
            "What do you think?"
        ]
        
        for _ in range(2):
            filler = rng.choice(fillers)
            speaker = rng.choice(characters)
            conversation.append({
                'speaker': speaker,
                'text': filler,
                'fact_type': 'filler',
                'fact_attr': None,
                'fact_value': None
            })
        
        return conversation
    
    def _adjust_message_count(self, conversation: List[Dict],
                             rng: random.Random) -> List[Dict]:
        """Adjust conversation to be within message bounds."""
        # Shuffle to randomize order
        rng.shuffle(conversation)
        
        # Trim to max if needed
        max_messages = self.config.max_messages
        if len(conversation) > max_messages:
            conversation = conversation[:max_messages]
        
        # Add more if below minimum
        min_messages = self.config.min_messages
        while len(conversation) < min_messages:
            # Re-add some messages
            msg = rng.choice(conversation)
            conversation.append(msg.copy())
        
        return conversation
    
    def get_all_generations(self) -> List[Dict]:
        """Return all conversation generations."""
        return self._generations.copy()
    
    def get_generation_count(self) -> int:
        """Get count of generations."""
        return len(self._generations)


def generate_conversation(scenario: Dict, liar_scenario: Dict, 
                         seed: int = None) -> List[Dict]:
    """Convenience function to generate a conversation."""
    generator = ConversationGenerator()
    return generator.generate_conversation(scenario, liar_scenario, seed=seed)
