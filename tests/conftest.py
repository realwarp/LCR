"""Test fixtures for The Liar's Chatroom competition toolkit."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_conversation_data():
    """Create sample conversation data for testing."""
    np.random.seed(42)
    
    # Create conversation IDs to simulate group structure
    conversation_ids = [f"conv_{i:03d}" for i in range(100)]
    
    data = []
    for conv_id in conversation_ids:
        # Each conversation has 4 characters (A, B, C, D)
        for char in ['A', 'B', 'C', 'D']:
            # Generate text
            text = f"Character {char} says something interesting in conversation {conv_id}. "
            text += "The character makes a claim that may or may not be true. "
            text += "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            
            # Random label (0 for truth, 1 for lie)
            label = np.random.randint(0, 2)
            
            data.append({
                'id': f"{conv_id}_{char}",
                'conversation_id': conv_id,
                'character': char,
                'text': text,
                'label': label
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Create sample test data (no labels)."""
    np.random.seed(42)
    
    conversation_ids = [f"test_conv_{i:03d}" for i in range(20)]
    
    data = []
    for conv_id in conversation_ids:
        for char in ['A', 'B', 'C', 'D']:
            text = f"Test character {char} in {conv_id}. "
            text += "This is test data for submission."
            
            data.append({
                'id': f"{conv_id}_{char}",
                'conversation_id': conv_id,
                'character': char,
                'text': text
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_evidence_data():
    """Create sample evidence data."""
    np.random.seed(42)
    
    evidence = []
    for i in range(50):
        evidence.append({
            'id': f"evidence_{i:03d}",
            'evidence_text': f"Verified fact {i}: The sky is blue and the grass is green. "
                            f"This is supporting evidence for truthfulness verification."
        })
    
    return pd.DataFrame(evidence)


@pytest.fixture
def sample_submission_format():
    """Create sample submission format."""
    return pd.DataFrame({
        'id': ['conv_000_A', 'conv_000_B', 'conv_000_C', 'conv_000_D'],
        'label': [0, 1, 0, 1]
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(sample_conversation_data, temp_dir):
    """Create a sample CSV file."""
    filepath = temp_dir / "train.csv"
    sample_conversation_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_jsonl_file(sample_conversation_data, temp_dir):
    """Create a sample JSONL file."""
    filepath = temp_dir / "train.jsonl"
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in sample_conversation_data.iterrows():
            f.write(row.to_json() + '\n')
    return filepath


@pytest.fixture
def synthetic_train_small():
    """Small synthetic training dataset (20 samples)."""
    np.random.seed(42)
    
    data = {
        'id': [f'train_{i:03d}' for i in range(20)],
        'conversation_id': [f'conv_{i:02d}' for i in range(20)],
        'character': ['A', 'B', 'C', 'D'] * 5,
        'text': [
            'The sun rises in the east and sets in the west.',
            'Water boils at 100 degrees Celsius at sea level.',
            'The Earth is flat and the sky is made of glass.',
            'Gravity pulls objects toward the center of the Earth.',
            '2 + 2 = 4 is a mathematical truth.',
            'The moon is made of cheese and emits its own light.',
            'Humans have 206 bones in their body.',
            'Light travels faster than sound in air.',
            'Penguins are flightless birds that live in Antarctica.',
            'Gold is a yellow metal that does not tarnish.',
            'The Amazon River is the longest river in the world.',
            'Mount Everest is the highest mountain on Earth.',
            'DNA contains genetic information for living organisms.',
            'Photosynthesis converts sunlight into energy.',
            'The human heart pumps blood throughout the body.',
            'Oxygen is essential for human respiration.',
            'The Great Wall of China is visible from space.',
            'The Pacific Ocean is the largest ocean on Earth.',
            'Cows produce milk that humans consume.',
            'Bees make honey from flower nectar.'
        ],
        'label': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_train_medium():
    """Medium synthetic training dataset (200 samples)."""
    np.random.seed(42)
    
    truths = [
        'The capital of France is Paris.',
        'Water freezes at 0 degrees Celsius.',
        'The square root of 16 is 4.',
        'Light travels at approximately 300,000 km/s.',
        'Carbon dioxide is a greenhouse gas.',
        'The periodic table has 118 elements.',
        'The human brain weighs about 1.3 kg.',
        'Octopuses have three hearts.',
        'Bananas are berries botanically.',
        'Honey never spoils.',
        'The Great Barrier Reef is the largest living structure.',
        'Ants can lift objects 50 times their body weight.',
        'Sloths can hold their breath longer than dolphins.',
        'Sharks are fish, not mammals.',
        'Octopuses have blue blood.',
        'The Venus flytrap is a carnivorous plant.',
        'Honeybees can recognize human faces.',
        'Octopuses have distributed intelligence.',
        'The platypus is a mammal that lays eggs.',
        'Komodo dragons have venomous bites.',
        'The Immortal Jellyfish can revert to its juvenile form.',
        'Tardigrades can survive in space.',
        'Bananas are radioactive due to potassium-40.',
        'Octopuses have nine brains.',
        'The hummingbird is the only bird that can fly backward.'
    ]
    
    lies = [
        'The capital of France is London.',
        'Water freezes at 50 degrees Celsius.',
        'The square root of 16 is 5.',
        'Light travels slower than sound.',
        'Carbon dioxide is not a greenhouse gas.',
        'The periodic table has 200 elements.',
        'The human brain weighs 5 kg.',
        'Octopuses have two hearts.',
        'Bananas are not berries.',
        'Honey spoils after one year.',
        'The Great Barrier Reef is invisible from space.',
        'Ants can lift objects 10 times their weight.',
        'Sloths cannot hold their breath.',
        'Sharks are mammals.',
        'Octopuses have red blood.',
        'The Venus flytrap eats only insects.',
        'Honeybees cannot recognize faces.',
        'Octopuses have centralized intelligence.',
        'The platypus lays no eggs.',
        'Komodo dragons have no venom.',
        'The Immortal Jellyfish cannot revert forms.',
        'Tardigrades cannot survive in space.',
        'Bananas are not radioactive.',
        'Octopuses have three brains.',
        'Hummingbirds cannot fly backward.'
    ]
    
    conversation_ids = [f'conv_{i:03d}' for i in range(50)]
    
    data = []
    for i, conv_id in enumerate(conversation_ids):
        for char in ['A', 'B', 'C', 'D']:
            if i % 2 == 0:
                text = truths[i % len(truths)]
                label = 0
            else:
                text = lies[i % len(lies)]
                label = 1
            
            data.append({
                'id': f'{conv_id}_{char}',
                'conversation_id': conv_id,
                'character': char,
                'text': text,
                'label': label
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_test_small():
    """Small synthetic test dataset (10 samples)."""
    np.random.seed(42)
    
    data = {
        'id': [f'test_{i:03d}' for i in range(10)],
        'conversation_id': [f'test_conv_{i:02d}' for i in range(10)],
        'character': ['A', 'B', 'C', 'D', 'A'] * 2,
        'text': [
            'The sun rises in the east.',
            'Water boils at 100°C.',
            'The Earth orbits the Sun.',
            'Gravity attracts all masses.',
            'Oxygen is necessary for life.',
            'The sky is blue because of Rayleigh scattering.',
            'The moon has no atmosphere.',
            'Mars is known as the Red Planet.',
            'Jupiter is the largest planet.',
            'Saturn has prominent rings.'
        ]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_evidence_small():
    """Small synthetic evidence dataset."""
    data = {
        'id': ['ev_001', 'ev_002', 'ev_003', 'ev_004', 'ev_005'],
        'evidence_text': [
            'Scientific measurement confirms water boils at 100°C at sea level.',
            'Astronomical observations verify Earth orbits the Sun.',
            'Atmospheric studies confirm oxygen is necessary for human respiration.',
            'Planetary surveys show Mars has a reddish appearance.',
            'Saturn ring observations confirm their prominent structure.'
        ]
    }
    
    return pd.DataFrame(data)
