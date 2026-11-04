"""Liar selection and false claim generation."""

import random
from typing import List, Dict, Tuple
from .config import LiarConfig


class LiarGenerator:
    """Handles liar selection and false claim generation."""
    
    def __init__(self, config: LiarConfig = None):
        self.config = config or LiarConfig()
        self._liar_assignments: List[Dict] = []
    
    def select_liar(self, characters: List[str] = None, seed: int = None) -> str:
        """Select exactly one liar from the characters."""
        rng = random.Random(seed)
        chars = characters or self.config.character_names
        liar = rng.choice(chars)
        return liar
    
    def generate_false_claim(self, liar: str, context: str = None, seed: int = None) -> Dict:
        """Generate a false claim for the selected liar."""
        rng = random.Random(seed)
        
        lie_type = rng.choice(self.config.lie_types)
        
        # Base truths
        truths = {
            'date_lie': [
                'The meeting is scheduled for Monday',
                'The project deadline is next Friday',
                'The event starts at 9 AM',
                'The report was due yesterday',
                'The conference is in December'
            ],
            'time_lie': [
                'The train departs at 8:00 PM',
                'The movie begins at 7:30 PM',
                'The store closes at 6:00 PM',
                'The appointment is at 2:00 PM',
                'The alarm goes off at 6:30 AM'
            ],
            'location_lie': [
                'The restaurant is on Main Street',
                'The office is downtown',
                'The party is at Sarah\'s house',
                'The gym is near the park',
                'The airport is in the south'
            ],
            'identity_lie': [
                'The manager is John Smith',
                'The author wrote 10 books',
                'The scientist discovered gravity',
                'The artist painted the Mona Lisa',
                'The inventor created the lightbulb'
            ],
            'quantity_lie': [
                'There are 100 people attending',
                'The box weighs 50 pounds',
                'The project costs $1000',
                'The room holds 50 people',
                'The recipe needs 3 cups'
            ],
            'negation_lie': [
                'I did not see the document',
                'She has not finished the task',
                'They were not informed',
                'We did not receive the email',
                'He has not completed the work'
            ],
            'status_lie': [
                'The project is progressing well',
                'The results are promising',
                'The budget is sufficient',
                'The team is ready',
                'The deadline is achievable'
            ]
        }
        
        # Get truth and modify it
        base_truth = rng.choice(truths.get(lie_type, truths['status_lie']))
        
        # Generate false version based on lie type
        if lie_type == 'date_lie':
            false_versions = [
                base_truth.replace('Monday', 'Tuesday'),
                base_truth.replace('Friday', 'Thursday'),
                base_truth.replace('Monday', 'Wednesday'),
                base_truth.replace('next Friday', 'tomorrow'),
                base_truth.replace('yesterday', 'tomorrow')
            ]
        elif lie_type == 'time_lie':
            false_versions = [
                base_truth.replace('8:00 PM', '6:00 PM'),
                base_truth.replace('7:30 PM', '8:30 PM'),
                base_truth.replace('6:00 PM', '8:00 PM'),
                base_truth.replace('2:00 PM', '4:00 PM'),
                base_truth.replace('6:30 AM', '7:30 AM')
            ]
        elif lie_type == 'location_lie':
            false_versions = [
                base_truth.replace('Main Street', 'Oak Street'),
                base_truth.replace('downtown', 'uptown'),
                base_truth.replace('Sarah\'s house', 'Mike\'s apartment'),
                base_truth.replace('near the park', 'across from the school'),
                base_truth.replace('in the south', 'in the north')
            ]
        elif lie_type == 'identity_lie':
            false_versions = [
                base_truth.replace('John Smith', 'Jane Doe'),
                base_truth.replace('10 books', '5 books'),
                base_truth.replace('discovered gravity', 'formulated relativity'),
                base_truth.replace('painted the Mona Lisa', 'sculpted David'),
                base_truth.replace('invented the lightbulb', 'patented the telephone')
            ]
        elif lie_type == 'quantity_lie':
            false_versions = [
                base_truth.replace('100', '50'),
                base_truth.replace('50 pounds', '30 pounds'),
                base_truth.replace('$1000', '$500'),
                base_truth.replace('50 people', '30 people'),
                base_truth.replace('3 cups', '5 cups')
            ]
        elif lie_type == 'negation_lie':
            # Remove "not" or "have not"
            false_versions = [
                base_truth.replace(' did not ', ' '),
                base_truth.replace(' has not ', ' has '),
                base_truth.replace(' were not ', ' were '),
                base_truth.replace(' did not ', ' did '),
                base_truth.replace(' has not ', ' has ')
            ]
        else:  # status_lie
            false_versions = [
                base_truth.replace('progressing well', 'facing delays'),
                base_truth.replace('promising', 'concerning'),
                base_truth.replace('sufficient', 'insufficient'),
                base_truth.replace('ready', 'unprepared'),
                base_truth.replace('achievable', 'impossible')
            ]
        
        false_claim = rng.choice(false_versions) if false_versions else base_truth
        
        assignment = {
            'liar': liar,
            'lie_type': lie_type,
            'false_claim': false_claim,
            'context': context,
            'seed': seed
        }
        
        self._liar_assignments.append(assignment)
        return assignment
    
    def generate_scenario(self, characters: List[str] = None, seed: int = None) -> Dict:
        """Generate a complete scenario with one liar."""
        rng = random.Random(seed)
        
        chars = characters or self.config.character_names
        liar = self.select_liar(characters=chars, seed=rng.randint(0, 10000))
        claim = self.generate_false_claim(liar, seed=rng.randint(0, 10000))
        
        scenario = {
            'liar': liar,
            'characters': chars,
            'claim': claim,
            'truth': claim['false_claim'].replace(' not ', ' ').replace(' not', '')  # Rough truth
        }
        
        return scenario
    
    def get_all_assignments(self) -> List[Dict]:
        """Return all liar assignments."""
        return self._liar_assignments.copy()


def create_liar_scenario(seed: int = None) -> Dict:
    """Convenience function to create a liar scenario."""
    generator = LiarGenerator()
    return generator.generate_scenario(seed=seed)
