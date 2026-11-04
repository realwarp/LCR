"""V2 fact-grounded liar generation."""

import random
from typing import List, Dict
from .config import LiarConfig, DatasetConfig


class LiarGenerator:
    """Generates fact-grounded liar scenarios."""
    
    def __init__(self, config: LiarConfig = None, dataset_config: DatasetConfig = None):
        self.config = config or LiarConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self._generations: List[Dict] = []
    
    def generate_liar_scenario(self, scenario: Dict, seed: int = None) -> Dict:
        """Generate a fact-grounded liar scenario tied to verified facts."""
        rng = random.Random(seed)
        
        characters = self.config.character_names
        
        # Select exactly one liar
        liar = rng.choice(characters)
        
        # Select a fact attribute from the scenario
        fact_attrs = scenario.get('verified_facts', {})
        if not fact_attrs:
            # Fallback if no verified facts
            fact_attrs = {'date': '15 December 2026', 'time': '9:00 AM'}
        
        attr_name = rng.choice(list(fact_attrs.keys()))
        true_value = fact_attrs[attr_name]
        
        # Select a lie type based on attribute
        lie_type = self._map_attr_to_lie_type(attr_name)
        
        # Generate false value
        false_value = self._generate_false_value(attr_name, true_value, lie_type, rng)
        
        # Create claim with full context
        context = self._create_claim_context(scenario, attr_name, true_value)
        
        claim = {
            'attribute': attr_name,
            'true_value': true_value,
            'false_value': false_value,
            'lie_type': lie_type,
            'context': context
        }
        
        liar_scenario = {
            'liar': liar,
            'characters': characters,
            'claim': claim,
            'scenario_id': scenario['scenario_id'],
            'scenario_family_id': scenario['scenario_family_id']
        }
        
        self._generations.append(liar_scenario)
        return liar_scenario
    
    def _map_attr_to_lie_type(self, attr_name: str) -> str:
        """Map attribute name to appropriate lie type."""
        type_mapping = {
            'date': 'date_lie',
            'time': 'time_lie',
            'location': 'location_lie',
            'topic': 'topic_lie',
            'duration': 'duration_lie',
            'attendees': 'attendees_lie',
            'budget': 'budget_lie',
            'quantity': 'quantity_lie'
        }
        return type_mapping.get(attr_name, 'status_lie')
    
    def _generate_false_value(self, attr_name: str, true_value: str, 
                              lie_type: str, rng: random.Random) -> str:
        """Generate a false value for the given attribute."""
        # Parse the true value and create a false variant
        
        if attr_name == 'date':
            # Parse "15 December 2026" format
            parts = true_value.split()
            if len(parts) >= 3:
                day = parts[0]
                month = parts[1]
                year = parts[2]
                # Generate different day
                days = ['12', '13', '14', '16', '17', '18']
                new_day = rng.choice([d for d in days if d != day])
                return f"{new_day} {month} {year}"
            return true_value
        
        elif attr_name == 'time':
            # Parse "9:00 AM" format
            parts = true_value.split()
            if len(parts) >= 2:
                time_part = parts[0]
                am_pm = parts[1]
                hour, minute = time_part.split(':')
                # Generate different hour
                hours = ['8', '10', '11', '2', '3', '4', '6', '7']
                new_hour = rng.choice(hours)
                return f"{new_hour}:{minute} {am_pm}"
            return true_value
        
        elif attr_name == 'location':
            # Modify location name
            locations = [
                true_value.replace('Main', 'Side'),
                true_value.replace('Main', 'Back'),
                true_value.replace('Auditorium', 'Seminar Hall'),
                true_value.replace('Hall', 'Room'),
                true_value.replace('Room', 'Lab')
            ]
            return rng.choice(locations)
        
        elif attr_name == 'topic':
            # Modify topic slightly
            topics = [
                true_value.replace('Workshop', 'Demo'),
                true_value.replace('Hackathon', 'Code Sprint'),
                true_value.replace('Review', 'Update'),
                true_value.replace('Discussion', 'Briefing')
            ]
            return rng.choice(topics)
        
        elif attr_name == 'duration':
            # Parse "2 hours" format
            parts = true_value.split()
            if len(parts) >= 2:
                num = parts[0]
                unit = parts[1]
                # Generate different number
                nums = ['1', '3', '4', '5', '6']
                new_num = rng.choice([n for n in nums if n != num])
                return f"{new_num} {unit}"
            return true_value
        
        elif attr_name == 'attendees':
            # Parse numeric value
            nums = ['5', '6', '7', '8', '9', '10', '11', '12', '15', '20']
            return rng.choice([n for n in nums if n != true_value])
        
        elif attr_name == 'budget':
            # Parse "$500" format
            if true_value.startswith('$'):
                val = true_value[1:]
                if val.isdigit():
                    nums = ['600', '800', '1200', '1500', '2000', '2500']
                    new_val = rng.choice(nums)
                    return f"${new_val}"
            return true_value
        
        elif attr_name == 'quantity':
            # Parse numeric value
            nums = ['2', '3', '4', '5', '6', '7']
            return rng.choice([n for n in nums if n != true_value])
        
        else:
            # Fallback: modify the value
            return f"modified_{true_value}"
    
    def _create_claim_context(self, scenario: Dict, attr_name: str, 
                              true_value: str) -> str:
        """Create context for the claim based on scenario facts."""
        family_name = scenario.get('scenario_family_name', 'event')
        topic = scenario.get('topic', 'activity')
        
        return f"In the {family_name} scenario about {topic}, the {attr_name} is {true_value}."
    
    def get_all_generations(self) -> List[Dict]:
        """Return all liar generations."""
        return self._generations.copy()
    
    def get_generation_count(self) -> int:
        """Get count of generations."""
        return len(self._generations)


def create_liar_scenario(scenario: Dict, seed: int = None) -> Dict:
    """Convenience function to create a liar scenario."""
    generator = LiarGenerator()
    return generator.generate_liar_scenario(scenario, seed=seed)
