"""Main script for generating synthetic datasets."""

import random
import argparse
import yaml
from pathlib import Path
from typing import List, Dict

from .config import ConfigManager, DatasetConfig
from .world_generator import WorldGenerator
from .liar_generator import LiarGenerator
from .conversation_generator import ConversationGenerator
from .llm_rewriter import LLMRewriter
from .validator import Validator
from .exporter import Exporter


def generate_single_conversation(
    world_gen: WorldGenerator,
    liar_gen: LiarGenerator,
    conv_gen: ConversationGenerator,
    world: Dict,
    seed: int = None
) -> Dict:
    """Generate a single conversation with a liar."""
    rng = random.Random(seed)
    
    characters = ['A', 'B', 'C', 'D']
    
    # Generate facts for this world
    world_data = world_gen.generate_world_and_facts(num_facts=5, seed=rng.randint(0, 10000))
    
    # Generate liar scenario
    scenario = liar_gen.generate_scenario(
        characters=characters,
        seed=rng.randint(0, 10000)
    )
    
    # Generate conversation
    conversation = conv_gen.generate_conversation(
        characters=characters,
        facts=world_data['facts'],
        liar=scenario['liar'],
        lie_claim=scenario['claim']['false_claim'],
        seed=rng.randint(0, 10000)
    )
    
    # Mark liar in conversation
    for msg in conversation:
        msg['is_liar'] = (msg['speaker'] == scenario['liar'])
    
    return {
        'conversation': conversation,
        'scenario': {
            'world': world_data['world'],
            'liar': scenario['liar'],
            'characters': characters,
            'claim': scenario['claim'],
            'facts': world_data['facts']
        }
    }


def generate_synthetic_dataset(
    num_examples: int = 100,
    output_dir: str = None,
    random_seed: int = 42,
    llm_enabled: bool = False,
    config_path: str = None
) -> Dict:
    """Generate a complete synthetic dataset."""
    # Initialize config
    if config_path:
        config_manager = ConfigManager(config_path)
        config_manager.load_from_yaml(config_path)
    else:
        config_manager = ConfigManager()
    
    if output_dir:
        config_manager.dataset.output_dir = output_dir
    config_manager.dataset.random_seed = random_seed
    
    # Set random seed
    random.seed(random_seed)
    
    # Initialize components
    world_gen = WorldGenerator(config_manager.world)
    liar_gen = LiarGenerator(config_manager.liar)
    conv_gen = ConversationGenerator(config_manager.conversation)
    llm_rewriter = LLMRewriter(config_manager.llm.__dict__)
    validator = Validator()
    
    # Generate conversations
    conversations = []
    scenarios = []
    
    for i in range(num_examples):
        world = world_gen.generate_world(seed=random.randint(0, 10000))
        
        result = generate_single_conversation(
            world_gen=world_gen,
            liar_gen=liar_gen,
            conv_gen=conv_gen,
            world=world,
            seed=random.randint(0, 10000)
        )
        
        conv = result['conversation']
        scenario = result['scenario']
        
        # Optional LLM rewrite
        if llm_enabled:
            conv = llm_rewriter.rewrite_conversation(conv, preserve_liar=True)
        
        # Validate conversation
        validation = validator.validate_conversation(
            conversation=conv,
            characters=scenario['characters'],
            expected_liar=scenario['liar']
        )
        
        if validation['valid']:
            conversations.append(conv)
            scenarios.append(scenario)
        else:
            print(f"Warning: Conversation {i} failed validation: {validation['issues']}")
    
    # Export dataset
    exporter = Exporter(config_manager.dataset)
    
    output_paths = exporter.export_all(conversations, scenarios)
    
    # Final validation
    dataset_validation = validator.validate_dataset(conversations, scenarios)
    
    return {
        'num_conversations': len(conversations),
        'num_scenarios': len(scenarios),
        'output_paths': output_paths,
        'validation': dataset_validation,
        'llm_stats': llm_rewriter.get_stats() if llm_enabled else None
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for The Liar\'s Chatroom competition'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of conversations to generate (default: 100)'
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
        '--llm-enabled',
        action='store_true',
        help='Enable LLM rewriting (requires local Ollama)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_examples} synthetic conversations...")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"LLM enabled: {args.llm_enabled}")
    
    result = generate_synthetic_dataset(
        num_examples=args.num_examples,
        output_dir=args.output_dir,
        random_seed=args.seed,
        llm_enabled=args.llm_enabled,
        config_path=args.config
    )
    
    print(f"\nGenerated {result['num_conversations']} conversations")
    print(f"\nOutput files:")
    for name, path in result['output_paths'].items():
        print(f"  - {name}: {path}")
    
    if result['validation']['valid']:
        print("\n✅ All validations passed!")
    else:
        print(f"\n⚠️ {result['validation']['statistics']['invalid_conversations']} conversations failed validation")
        print("Issues:", result['validation']['issues'][:5])
    
    if result.get('llm_stats'):
        print(f"\nLLM stats: {result['llm_stats']}")
    
    return result


if __name__ == '__main__':
    main()
