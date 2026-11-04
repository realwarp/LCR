"""V2 scenario generator with fact-first approach."""

import random
from typing import List, Dict
from .config import WorldConfig, DatasetConfig


class ScenarioGenerator:
    """Generates fact-rich scenarios with verified attributes."""
    
    def __init__(self, config: WorldConfig = None, dataset_config: DatasetConfig = None):
        self.config = config or WorldConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self._generated_scenarios: List[Dict] = []
        self._scenario_families_used: Dict[str, int] = {}  # family_id -> count
    
    def generate_scenario_family(self, seed: int = None) -> Dict:
        """Generate a reusable scenario family."""
        rng = random.Random(seed)
        
        # Select a scenario family
        family = rng.choice(self.config.scenario_families)
        
        # Generate a family ID
        family_id = self.dataset_config.generate_scenario_family_id()
        
        # Store family for reuse
        if family_id not in self._scenario_families_used:
            self._scenario_families_used[family_id] = 0
        
        return {
            'family_id': family_id,
            'family_name': family.name,
            'topic': family.topic,
            'context': family.context,
            'attribute_templates': family.attribute_templates.copy()
        }
    
    def generate_scenario(self, seed: int = None) -> Dict:
        """Generate a single scenario with verified facts."""
        rng = random.Random(seed)
        
        # Generate or select a scenario family
        family = self.generate_scenario_family(seed=rng.randint(0, 10000))
        
        # Select attribute values from templates to create verified facts
        verified_facts = {}
        for attr_name, values in family['attribute_templates'].items():
            value = rng.choice(values)
            verified_facts[attr_name] = value
        
        # Generate scenario ID
        scenario_id = self.dataset_config.generate_scenario_id()
        
        scenario = {
            'scenario_id': scenario_id,
            'scenario_family_id': family['family_id'],
            'scenario_family_name': family['family_name'],
            'topic': family['topic'],
            'context': family['context'],
            'verified_facts': verified_facts,
            'fact_attributes': list(verified_facts.keys()),
            'characters': ['A', 'B', 'C', 'D']
        }
        
        self._generated_scenarios.append(scenario)
        return scenario
    
    def get_scenario_family(self, family_id: str) -> Dict:
        """Get a scenario family by ID."""
        for scenario in self._generated_scenarios:
            if scenario['scenario_family_id'] == family_id:
                return {
                    'family_id': scenario['scenario_family_id'],
                    'family_name': scenario['scenario_family_name'],
                    'topic': scenario['topic'],
                    'context': scenario['context'],
                    'verified_facts': scenario['verified_facts']
                }
        return None
    
    def get_all_families(self) -> List[Dict]:
        """Get all unique scenario families."""
        families = {}
        for scenario in self._generated_scenarios:
            fid = scenario['scenario_family_id']
            if fid not in families:
                families[fid] = {
                    'family_id': fid,
                    'family_name': scenario['scenario_family_name'],
                    'topic': scenario['topic'],
                    'context': scenario['context'],
                    'verified_facts': scenario['verified_facts']
                }
        return list(families.values())
    
    def get_generated_count(self) -> int:
        """Get count of generated scenarios."""
        return len(self._generated_scenarios)


def create_scenario(seed: int = None) -> Dict:
    """Convenience function to create a scenario."""
    generator = ScenarioGenerator()
    return generator.generate_scenario(seed=seed)
