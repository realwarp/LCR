"""Fictional world and fact generator."""

import random
from typing import List, Tuple, Dict
from .config import WorldConfig


class WorldGenerator:
    """Generates fictional worlds with verified facts."""
    
    def __init__(self, config: WorldConfig = None):
        self.config = config or WorldConfig()
        self._worlds: List[Dict] = []
        self._facts: List[str] = []
    
    def generate_world(self, seed: int = None) -> Dict:
        """Generate a single fictional world."""
        rng = random.Random(seed)
        
        theme = rng.choice(self.config.themes)
        location = rng.choice(self.config.locations)
        time_period = rng.choice(self.config.time_periods)
        topic = rng.choice(self.config.topics)
        
        # Generate a world name
        prefixes = ['The', 'A', 'An', 'Legacy of', 'Chronicles of', 'Tales of']
        suffixes = ['Reality', 'Dimension', 'Realm', 'World', 'Domain', 'Epoch']
        name = f"{rng.choice(prefixes)} {theme.title().replace(' ', '')} {rng.choice(suffixes)}"
        
        world = {
            'name': name,
            'theme': theme,
            'location': location,
            'time_period': time_period,
            'topic': topic,
            'description': (
                f"In {time_period}, the characters gather at {location} to discuss "
                f"{topic} within the {theme} setting."
            )
        }
        
        self._worlds.append(world)
        return world
    
    def generate_fact(self, seed: int = None) -> str:
        """Generate a verified factual statement."""
        rng = random.Random(seed)
        
        truth_templates = [
            "The {subject} is {attribute}.",
            "{subject} {verb} {object}.",
            "It is scientifically proven that {subject} {verb}.",
            "{subject} {quantifier} {attribute}.",
            "Research confirms that {subject} {verb} {object}."
        ]
        
        subjects = [
            "water", "gravity", "light", "oxygen", "carbon dioxide",
            "DNA", "the Earth", "the Sun", "atoms", "cells"
        ]
        
        attributes = [
            "essential for life", "composed of molecules", "has mass",
            "travels at a finite speed", "conserved in closed systems",
            "fundamental to physics", "measurable with instruments"
        ]
        
        verbs = [
            "causes objects to fall", "enables photosynthesis", "forms water when combined",
            "is necessary for respiration", "contributes to the greenhouse effect",
            "carries genetic information", "orbits the Sun"
        ]
        
        objects = [
            "in a vacuum", "on Earth", "under standard conditions",
            "in living organisms", "in the atmosphere", "in experiments"
        ]
        
        quantifiers = ["has", "contains", "requires", "produces", "absorbs"]
        
        template = rng.choice(truth_templates)
        
        # Simple template selection to avoid complex combinations
        if "Research confirms" in template:
            fact = template.format(
                subject=rng.choice(subjects).capitalize(),
                verb=rng.choice(verbs),
                object=rng.choice(objects)
            )
        elif "scientifically proven" in template:
            fact = template.format(
                subject=rng.choice(subjects).capitalize(),
                verb=rng.choice(verbs)
            )
        elif "{quantifier}" in template:
            fact = template.format(
                subject=rng.choice(subjects).capitalize(),
                quantifier=rng.choice(quantifiers),
                attribute=rng.choice(attributes)
            )
        elif "{object}" in template:
            fact = template.format(
                subject=rng.choice(subjects).capitalize(),
                verb=rng.choice(verbs),
                object=rng.choice(objects)
            )
        else:
            fact = template.format(
                subject=rng.choice(subjects).capitalize(),
                attribute=rng.choice(attributes)
            )
        
        self._facts.append(fact)
        return fact
    
    def generate_world_and_facts(self, num_facts: int = 5, seed: int = None) -> Dict:
        """Generate a world with multiple verified facts."""
        rng = random.Random(seed)
        
        world = self.generate_world(seed=rng.randint(0, 10000))
        facts = [self.generate_fact(seed=rng.randint(0, 10000)) for _ in range(num_facts)]
        
        return {
            'world': world,
            'facts': facts
        }
    
    def get_all_worlds(self) -> List[Dict]:
        """Return all generated worlds."""
        return self._worlds.copy()
    
    def get_all_facts(self) -> List[str]:
        """Return all generated facts."""
        return self._facts.copy()


def create_fictional_world(seed: int = None) -> Dict:
    """Convenience function to create a world."""
    generator = WorldGenerator()
    return generator.generate_world_and_facts(seed=seed)
