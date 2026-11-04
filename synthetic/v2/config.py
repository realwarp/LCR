"""V2 configuration for synthetic dataset generation."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import yaml
from pathlib import Path


@dataclass
class ScenarioFamilyConfig:
    """Configuration for a scenario family."""
    name: str
    topic: str
    context: str
    attribute_templates: Dict[str, List[str]]  # attribute_name -> list of possible values
    
    def generate_attribute(self, attr_name: str, seed: int = None) -> str:
        """Generate a value for an attribute."""
        from random import Random
        rng = Random(seed)
        values = self.attribute_templates.get(attr_name, [])
        if not values:
            return "unknown"
        return rng.choice(values)


@dataclass  
class WorldConfig:
    """Configuration for generating fictional worlds."""
    scenario_families: List[ScenarioFamilyConfig] = None
    
    def __post_init__(self):
        if self.scenario_families is None:
            # Define scenario families with structured attributes
            self.scenario_families = [
                ScenarioFamilyConfig(
                    name='college_event',
                    topic='TechFest 2026',
                    context='Students planning for the annual college technology fest',
                    attribute_templates={
                        'date': ['15 December 2026', '16 December 2026', '17 December 2026'],
                        'time': ['9:00 AM', '10:00 AM', '11:00 AM', '2:00 PM'],
                        'location': ['Main Auditorium', 'Seminar Hall 1', 'Lab Block A'],
                        'topic': ['AI Workshop', 'Hackathon', 'Robotics Demo', 'Code Sprint'],
                        'duration': ['2 hours', '3 hours', '4 hours'],
                        'organizer': ['Tech Club', 'Computer Science Dept', 'Student Council'],
                        'expected_attendees': ['100', '200', '300', '500']
                    }
                ),
                ScenarioFamilyConfig(
                    name='travel_schedule',
                    topic='Business Trip to Mumbai',
                    context='Colleagues coordinating travel plans for a conference',
                    attribute_templates={
                        'date': ['5 December 2026', '6 December 2026', '7 December 2026'],
                        'time': ['8:00 AM', '11:30 AM', '3:00 PM', '7:45 PM'],
                        'location': ['Mumbai Central', 'Chhatrapati Shivaji Airport', 'Taj Hotel'],
                        'topic': ['Client Meeting', 'Product Launch', 'Team Briefing'],
                        'transport': ['Train', 'Flight', 'Taxi'],
                        'duration': ['1 day', '2 days', '3 days'],
                        'accompaniment': ['one colleague', 'two colleagues', 'entire team']
                    }
                ),
                ScenarioFamilyConfig(
                    name='meeting',
                    topic='Weekly Project Review',
                    context='Team members discussing project progress',
                    attribute_templates={
                        'date': ['Monday', 'Wednesday', 'Friday'],
                        'time': ['10:00 AM', '2:00 PM', '4:00 PM'],
                        'location': ['Conference Room A', 'Virtual Meeting', 'Lunch Room'],
                        'topic': [' sprint review', 'budget discussion', 'timeline update'],
                        'duration': ['30 minutes', '45 minutes', '1 hour'],
                        'attendees': ['6', '8', '10', '12'],
                        'agenda': ['milestones', 'blockers', 'deliverables']
                    }
                ),
                ScenarioFamilyConfig(
                    name='restaurant_reservation',
                    topic='Dinner at Italian Bistro',
                    context='Friends planning a weekend dinner',
                    attribute_templates={
                        'date': ['Saturday', 'Sunday'],
                        'time': ['7:00 PM', '8:00 PM', '9:00 PM'],
                        'location': ['Italian Bistro', 'Garden Cafe', 'Rooftop Restaurant'],
                        'topic': [' anniversary dinner', 'birthday celebration', 'reunion'],
                        'party_size': ['2', '4', '6', '8'],
                        'reservation_name': ['John', 'Sarah', 'Mike', 'Emily'],
                        'special_request': ['window seat', 'vegetarian menu', 'private room']
                    }
                ),
                ScenarioFamilyConfig(
                    name='project_deadline',
                    topic='Software Release v2.0',
                    context='Engineering team coordinating release timeline',
                    attribute_templates={
                        'date': ['15 December', '20 December', '25 December'],
                        'time': ['5:00 PM', '6:00 PM', '7:00 PM'],
                        'location': ['Development Lab', 'QA Room', 'Server Room'],
                        'topic': ['bug fixing', 'performance testing', 'deployment'],
                        'duration': ['2 hours', '4 hours', '8 hours'],
                        'team_size': ['4', '6', '8', '10'],
                        'milestone': ['beta release', 'RC1', 'production ready']
                    }
                ),
                ScenarioFamilyConfig(
                    name='sports_result',
                    topic='Inter-College Football Tournament',
                    context='Team discussing match outcomes',
                    attribute_templates={
                        'date': ['12 November', '13 November', '14 November'],
                        'time': ['11:00 AM', '3:00 PM', '5:00 PM'],
                        'location': ['Main Stadium', 'Sports Complex', 'Fields'],
                        'topic': ['quarterfinal', 'semifinal', 'final'],
                        'duration': ['90 minutes', '2 hours', '3 hours'],
                        'teams': ['IT Dept', 'Mechanical', 'Electrical', 'Civil'],
                        'score': ['2-1', '3-0', '1-1']
                    }
                ),
                ScenarioFamilyConfig(
                    name='shopping',
                    topic='Tech Gadgets Purchase',
                    context='Friends shopping for new electronics',
                    attribute_templates={
                        'date': ['this Saturday', 'next Sunday', 'this Friday'],
                        'time': ['10:00 AM', '12:00 PM', '3:00 PM'],
                        'location': ['Electronics Market', 'Shopping Mall', 'Online'],
                        'topic': ['smartphone', 'laptop', 'headphones', 'smartwatch'],
                        'budget': ['$500', '$1000', '$1500', '$2000'],
                        'quantity': ['1', '2', '3'],
                        'store': ['Tech Store', 'Electronics Hub', 'Brand Outlet']
                    }
                ),
                ScenarioFamilyConfig(
                    name='delivery',
                    topic='Package Delivery Coordination',
                    context='Roommates managing delivery schedules',
                    attribute_templates={
                        'date': ['today', 'tomorrow', 'Friday'],
                        'time': ['10:00 AM', '2:00 PM', '6:00 PM'],
                        'location': ['Front Desk', 'Security Gate', 'Apartment'],
                        'topic': ['electronics', 'clothing', 'books', 'food'],
                        'carrier': ['FedEx', 'DHL', 'Amazon', 'Indian Post'],
                        'tracking_number': ['FX123456789', 'DH987654321', 'AM456789123'],
                        'package_size': ['small', 'medium', 'large']
                    }
                )
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
                'date_lie', 'time_lie', 'location_lie', 'topic_lie',
                'duration_lie', 'attendees_lie', 'budget_lie', 'quantity_lie'
            ]


@dataclass
class ConversationConfig:
    """Configuration for conversation generation."""
    min_messages: int = 8
    max_messages: int = 20
    filler_probability: float = 0.15
    min_messages_per_speaker: int = 1
    
    def __post_init__(self):
        if self.min_messages < 4:
            self.min_messages = 4
        if self.max_messages < self.min_messages:
            self.max_messages = self.min_messages


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
    max_generation_attempts: int = 10  # Max attempts per conversation
    
    # Quality thresholds
    min_message_length: int = 10
    max_message_length: int = 500
    
    # IDs tracking
    next_conversation_id: int = 1
    next_scenario_id: int = 1
    next_family_id: int = 1
    
    def __post_init__(self):
        if not (0 < self.train_ratio < 1):
            self.train_ratio = 0.8
        if self.max_generation_attempts < 1:
            self.max_generation_attempts = 10
    
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
    
    def generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        conv_id = f"conv_{self.next_conversation_id:06d}"
        self.next_conversation_id += 1
        return conv_id
    
    def generate_scenario_id(self) -> str:
        """Generate a unique scenario ID."""
        scenario_id = f"scenario_{self.next_scenario_id:06d}"
        self.next_scenario_id += 1
        return scenario_id
    
    def generate_scenario_family_id(self) -> str:
        """Generate a unique scenario family ID."""
        family_id = f"family_{self.next_family_id:04d}"
        self.next_family_id += 1
        return family_id


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
            'world': {'scenario_families': [f.__dict__ for f in self.world.scenario_families]},
            'liar': vars(self.liar),
            'conversation': vars(self.conversation),
            'llm': vars(self.llm),
            'dataset': vars(self.dataset)
        }
    
    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
