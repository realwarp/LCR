"""Synthetic dataset generation configuration."""

from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class WorldConfig:
    """Configuration for generating fictional worlds."""
    themes: List[str] = None
    locations: List[str] = None
    time_periods: List[str] = None
    topics: List[str] = None
    
    def __post_init__(self):
        if self.themes is None:
            self.themes = [
                'historical mystery', 'sci-fi adventure', 'fantasy quest',
                'detective thriller', 'time travel paradox', 'alien contact',
                'hidden identity', 'forgotten knowledge', 'ancient prophecy'
            ]
        if self.locations is None:
            self.locations = [
                'a hidden library', 'an ancient ruin', 'a spaceship bridge',
                'a secret laboratory', 'an underground bunker', 'a haunted mansion',
                'a futuristic city', 'a mystical forest', 'a desert oasis'
            ]
        if self.time_periods is None:
            self.time_periods = [
                'the Victorian era', 'the future', 'medieval times',
                'the 1920s', 'a post-apocalyptic world', 'an alternate dimension'
            ]
        if self.topics is None:
            self.topics = [
                'science and technology', 'history and archaeology',
                'physics and mathematics', 'biology and chemistry',
                'astronomy and space', 'geology and climate'
            ]


@dataclass
class LiarConfig:
    """Configuration for liar generation."""
    character_names: List[str] = None
    lie_types: List[str] = None
    
    def __post_init__(self):
        if self.character_names is None:
            self.character_names = ['A', 'B', 'C', 'D']
        if self.lie_types is None:
            self.lie_types = [
                'date_lie', 'time_lie', 'location_lie', 'identity_lie',
                'quantity_lie', 'negation_lie', 'status_lie'
            ]


@dataclass
class ConversationConfig:
    """Configuration for conversation generation."""
    min_messages: int = 8
    max_messages: int = 20
    filler_probability: float = 0.15
    message_templates: List[str] = None
    
    def __post_init__(self):
        if self.message_templates is None:
            self.message_templates = [
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
            ]


@dataclass
class LLMConfig:
    """Configuration for optional LLM rewriting."""
    enabled: bool = False
    model: str = "llama3"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_retries: int = 3
    confidence_threshold: float = 0.8
    
    def is_available(self) -> bool:
        """Check if LLM is properly configured."""
        return self.enabled


@dataclass
class DatasetConfig:
    """Main dataset generation configuration."""
    num_examples: int = 100
    output_dir: str = "data/synthetic"
    random_seed: int = 42
    train_ratio: float = 0.8
    
    # Scenario families for balanced splitting
    scenario_families: List[str] = None
    
    # Quality thresholds
    min_message_length: int = 10
    max_message_length: int = 500
    
    def __post_init__(self):
        if self.scenario_families is None:
            self.scenario_families = [
                'family_gathering', 'work_project', 'school_reunion',
                'science_club', 'book_club', 'tech_team', 'art_crew',
                'sports_team', 'volunteer_group', 'study_buddy'
            ]
    
    def get_output_paths(self) -> dict:
        """Get output file paths."""
        base = Path(self.output_dir)
        return {
            'train_csv': str(base / 'train.csv'),
            'validation_csv': str(base / 'validation.csv'),
            'ground_truth_jsonl': str(base / 'ground_truth.jsonl'),
            'scenarios_jsonl': str(base / 'scenarios.jsonl'),
            'quality_report': str(base / 'synthetic_quality_report.md')
        }


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.world = WorldConfig()
        self.liar = LiarConfig()
        self.conversation = ConversationConfig()
        self.llm = LLMConfig()
        self.dataset = DatasetConfig()
    
    def load_from_yaml(self, filepath: str) -> None:
        """Load configuration from YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._apply_config(data)
    
    def _apply_config(self, data: dict) -> None:
        """Apply configuration from dict."""
        if not data:
            return
        
        # Deep merge nested configs
        if 'world' in data:
            self.world = WorldConfig(**data['world'])
        if 'liar' in data:
            self.liar = LiarConfig(**data['liar'])
        if 'conversation' in data:
            conv_data = data['conversation']
            self.conversation = ConversationConfig(**conv_data)
        if 'llm' in data:
            self.llm = LLMConfig(**data['llm'])
        if 'dataset' in data:
            self.dataset = DatasetConfig(**data['dataset'])
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'world': vars(self.world),
            'liar': vars(self.liar),
            'conversation': vars(self.conversation),
            'llm': vars(self.llm),
            'dataset': vars(self.dataset)
        }
    
    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
