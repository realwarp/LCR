"""Main script for V2 synthetic dataset generation."""

import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from .config import ConfigManager, DatasetConfig
from .scenario_generator import ScenarioGenerator
from .liar_generator import LiarGenerator
from .conversation_generator import ConversationGenerator
from .validator import Validator
from .exporter import Exporter


def generate_single_conversation_v2(
    scenario_gen: ScenarioGenerator,
    liar_gen: LiarGenerator,
    conv_gen: ConversationGenerator,
    seed: int = None
) -> Tuple[List[Dict], Dict, Dict]:
    """Generate a single V2 conversation with a liar.
    
    Returns: (conversation, scenario, ground_truth)
    """
    rng = random.Random(seed)
    
    # Step 1: Generate scenario with verified facts
    scenario = scenario_gen.generate_scenario(seed=rng.randint(0, 10000))
    
    # Step 2: Generate fact-grounded liar scenario
    liar_scenario = liar_gen.generate_liar_scenario(scenario, seed=rng.randint(0, 10000))
    
    # Step 3: Generate conversation based on verified facts
    conversation = conv_gen.generate_conversation(
        scenario=scenario,
        liar_scenario=liar_scenario,
        seed=rng.randint(0, 10000)
    )
    
    # Build ground truth record
    ground_truth = {
        'conversation_id': scenario['scenario_id'],
        'scenario_id': scenario['scenario_id'],
        'scenario_family_id': scenario['scenario_family_id'],
        'liar': liar_scenario['liar'],
        'claim': liar_scenario['claim'],
        'verified_facts': scenario['verified_facts']
    }
    
    return conversation, scenario, ground_truth


def generate_synthetic_dataset_v2(
    num_examples: int = 100,
    output_dir: str = None,
    random_seed: int = 42,
    max_attempts: int = 10,
    config_path: str = None
) -> Dict:
    """Generate a complete V2 synthetic dataset.
    
    Returns exactly num_examples valid conversations.
    """
    # Initialize config
    if config_path:
        config_manager = ConfigManager(config_path)
        config_manager.load_from_yaml(config_path)
    else:
        config_manager = ConfigManager()
    
    if output_dir:
        config_manager.dataset.output_dir = output_dir
    config_manager.dataset.random_seed = random_seed
    
    # Override max attempts if provided
    if max_attempts:
        config_manager.dataset.max_generation_attempts = max_attempts
    
    # Set random seed
    random.seed(random_seed)
    
    # Initialize components
    scenario_gen = ScenarioGenerator(
        config_manager.world,
        config_manager.dataset
    )
    liar_gen = LiarGenerator(
        config_manager.liar,
        config_manager.dataset
    )
    conv_gen = ConversationGenerator(config_manager.conversation)
    validator = Validator()
    
    # Generation tracking
    requested = num_examples
    attempted = 0
    accepted = 0
    rejected = 0
    
    conversations = []
    scenarios = []
    ground_truths = []
    
    print(f"Generating {num_examples} valid conversations...")
    print(f"Max attempts per conversation: {config_manager.dataset.max_generation_attempts}")
    
    while accepted < num_examples:
        # Generate a candidate
        attempt_seed = random.randint(0, 10000)
        conversation, scenario, ground_truth = generate_single_conversation_v2(
            scenario_gen=scenario_gen,
            liar_gen=liar_gen,
            conv_gen=conv_gen,
            seed=attempt_seed
        )
        
        attempted += 1
        
        # Validate
        validation = validator.validate_conversation(
            conversation=conversation,
            characters=scenario['characters'],
            expected_liar=scenario['liar']
        )
        
        if validation['valid']:
            conversations.append(conversation)
            scenarios.append(scenario)
            ground_truths.append(ground_truth)
            accepted += 1
            
            if accepted % 20 == 0:
                print(f"  Accepted {accepted}/{num_examples} ({attempted} attempts)")
        else:
            rejected += 1
        
        # Check if we've exceeded max attempts
        if attempted > num_examples * config_manager.dataset.max_generation_attempts:
            print(f"\n⚠️  Exceeded max attempts ({attempted})")
            print(f"  Requested: {requested}, Accepted: {accepted}, Rejected: {rejected}")
            break
    
    print(f"\nGeneration complete:")
    print(f"  Requested: {requested}")
    print(f"  Attempted: {attempted}")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")
    
    # Check if we got exactly what we wanted
    if accepted != num_examples:
        print(f"\n⚠️  WARNING: Only generated {accepted} of {num_examples} requested")
        return None  # Signal failure
    
    # Validate entire dataset
    dataset_validation = validator.validate_dataset(
        conversations, scenarios, ground_truths
    )
    
    if not dataset_validation['valid']:
        print(f"\n⚠️  Dataset validation failed:")
        for issue in dataset_validation['issues'][:5]:
            print(f"  - {issue}")
        return None
    
    # Export dataset
    exporter = Exporter(config_manager.dataset)
    output_paths = exporter.export_all(conversations, scenarios, ground_truths)
    
    print(f"\nExported to:")
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")
    
    # Final stats
    stats = exporter.get_stats()
    print(f"\nExport statistics:")
    print(f"  Conversations: {stats['conversations_exported']}")
    print(f"  Messages: {stats['messages_exported']}")
    print(f"  Ground truth records: {stats['ground_truth_records']}")
    print(f"  Scenario records: {stats['scenario_records']}")
    
    return {
        'num_conversations': accepted,
        'num_scenarios': len(scenarios),
        'num_ground_truth': len(ground_truths),
        'output_paths': output_paths,
        'validation': dataset_validation,
        'generated': accepted,
        'requested': requested,
        'attempted': attempted,
        'rejected': rejected
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset V2 for The Liar\'s Chatroom competition'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of valid conversations to generate (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Output directory (default: data/synthetic)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=10,
        help='Max attempts per conversation (default: 10)'
    )
    
    args = parser.parse_args()
    
    result = generate_synthetic_dataset_v2(
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        random_seed=args.seed,
        max_attempts=args.max_attempts,
        config_path=args.config
    )
    
    if result is None:
        print("\n❌ Generation failed - could not produce valid dataset")
        return 1
    
    if result['generated'] == result['requested']:
        print(f"\n✅ SUCCESS: Generated exactly {result['generated']} valid conversations")
        return 0
    else:
        print(f"\n⚠️  WARNING: Generated {result['generated']} of {result['requested']} requested")
        return 1


if __name__ == '__main__':
    exit(main())
