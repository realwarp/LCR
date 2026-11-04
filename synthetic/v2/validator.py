"""V2 validation with comprehensive checks."""

from typing import List, Dict, Set
import re


class Validator:
    """Validates synthetic conversation data with V2 checks."""
    
    def __init__(self):
        self._validations: List[Dict] = []
    
    def validate_conversation(
        self,
        conversation: List[Dict],
        characters: List[str],
        expected_liar: str,
        num_messages_min: int = 8,
        num_messages_max: int = 20
    ) -> Dict:
        """Validate a single conversation."""
        validation = {
            'valid': True,
            'issues': [],
            'checks': {}
        }
        
        # Check exactly 4 characters
        chars = set(msg.get('speaker', '') for msg in conversation)
        validation['checks']['four_characters'] = len(chars) == 4
        if not validation['checks']['four_characters']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected 4 characters, got {len(chars)}: {chars}"
            )
        
        # Check character names are A, B, C, D
        expected_chars = set(['A', 'B', 'C', 'D'])
        validation['checks']['correct_characters'] = chars == expected_chars
        if not validation['checks']['correct_characters']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected characters A,B,C,D, got {chars}"
            )
        
        # Check message count
        num_messages = len(conversation)
        validation['checks']['message_count'] = (
            num_messages_min <= num_messages <= num_messages_max
        )
        if not validation['checks']['message_count']:
            validation['valid'] = False
            validation['issues'].append(
                f"Message count {num_messages} outside range [{num_messages_min}, {num_messages_max}]"
            )
        
        # Check exactly one liar with is_liar flag
        liars = set(
            msg.get('speaker') for msg in conversation
            if msg.get('is_liar', False)
        )
        validation['checks']['one_liar'] = len(liars) == 1
        if not validation['checks']['one_liar']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected 1 liar, got {len(liars)}: {liars}"
            )
        
        # Check liar matches expected
        validation['checks']['correct_liar'] = liars == {expected_liar}
        if not validation['checks']['correct_liar']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected liar '{expected_liar}', got {liars}"
            )
        
        # Check message content - must be > 10 chars
        validation['checks']['messages_have_content'] = all(
            len(str(msg.get('text', '')).strip()) > 10
            for msg in conversation
        )
        if not validation['checks']['messages_have_content']:
            validation['valid'] = False
            validation['issues'].append("Some messages lack sufficient content (>10 chars)")
        
        # Check no ground truth leakage in messages
        validation['checks']['no_ground_truth'] = not any(
            'true' in str(msg.get('text', '')).lower() or
            'false' in str(msg.get('text', '')).lower() or
            'lie' in str(msg.get('text', '')).lower() or
            'liar' in str(msg.get('text', '')).lower() or
            'truth' in str(msg.get('text', '')).lower()
            for msg in conversation
        )
        if not validation['checks']['no_ground_truth']:
            validation['valid'] = False
            validation['issues'].append("Ground truth or liar identity revealed in messages")
        
        # Check every character speaks
        speakers = set(msg.get('speaker', '') for msg in conversation)
        validation['checks']['all_speak'] = speakers == expected_chars
        if not validation['checks']['all_speak']:
            validation['valid'] = False
            validation['issues'].append(
                f"Not all characters speak: missing {expected_chars - speakers}"
            )
        
        # Check at least one false claim by liar
        liar_messages = [msg for msg in conversation if msg.get('speaker') == expected_liar]
        has_false_claim = any(
            msg.get('fact_type') == 'false_claim' or
            msg.get('is_liar_message')
            for msg in liar_messages
        )
        validation['checks']['liar_makes_false_claim'] = has_false_claim
        if not validation['checks']['liar_makes_false_claim']:
            validation['valid'] = False
            validation['issues'].append("Liar does not make a false claim")
        
        # Check non-liars don't contradict verified facts
        validation['checks']['non_liar_consistent'] = True
        for msg in conversation:
            if not msg.get('is_liar', False) and msg.get('fact_type') == 'false_claim':
                validation['valid'] = False
                validation['checks']['non_liar_consistent'] = False
                validation['issues'].append(
                    f"Non-liar {msg.get('speaker')} makes false claim"
                )
        
        self._validations.append(validation)
        return validation
    
    def validate_dataset(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """Validate entire dataset with V2 checks."""
        validation = {
            'valid': True,
            'conversation_validations': [],
            'issues': [],
            'statistics': {
                'total_conversations': len(conversations),
                'valid_conversations': 0,
                'invalid_conversations': 0,
                'total_messages': 0,
                'avg_messages_per_conversation': 0
            },
            'checks': {}
        }
        
        total_messages = 0
        conversation_ids = []
        scenario_ids = []
        liar_counts = {}
        lie_type_counts = {}
        message_counts = []
        speaker_counts = {}
        
        for i, (conv, scenario, gt) in enumerate(zip(conversations, scenarios, ground_truth)):
            # Use ground_truth for expected_liar since scenario may not have it
            conv_validation = self.validate_conversation(
                conversation=conv,
                characters=scenario.get('characters', ['A', 'B', 'C', 'D']),
                expected_liar=gt.get('liar', scenario.get('liar', ''))
            )
            
            validation['conversation_validations'].append(conv_validation)
            total_messages += len(conv)
            message_counts.append(len(conv))
            
            conv_id = scenario.get('scenario_id', f'conv_{i}')
            conversation_ids.append(conv_id)
            scenario_ids.append(scenario.get('scenario_family_id', 'unknown'))
            
            if conv_validation['valid']:
                validation['statistics']['valid_conversations'] += 1
            else:
                validation['statistics']['invalid_conversations'] += 1
                validation['issues'].extend(conv_validation['issues'])
            
            # Track statistics
            liar = scenario.get('liar', 'unknown')
            liar_counts[liar] = liar_counts.get(liar, 0) + 1
            
            claim = scenario.get('claim', {})
            lie_type = claim.get('lie_type', 'unknown')
            lie_type_counts[lie_type] = lie_type_counts.get(lie_type, 0) + 1
            
            for msg in conv:
                char = msg.get('speaker', 'unknown')
                speaker_counts[char] = speaker_counts.get(char, 0) + 1
        
        validation['statistics']['total_messages'] = total_messages
        validation['statistics']['avg_messages_per_conversation'] = (
            total_messages / len(conversations) if conversations else 0
        )
        
        # Check 1:1:1 ratio of conversation to scenario to ground truth
        validation['checks']['record_count_match'] = (
            len(conversations) == len(scenarios) == len(ground_truth)
        )
        if not validation['checks']['record_count_match']:
            validation['valid'] = False
            validation['issues'].append(
                f"Record counts don't match: conv={len(conversations)}, "
                f"scenario={len(scenarios)}, gt={len(ground_truth)}"
            )
        
        # Check unique conversation IDs
        validation['checks']['unique_conv_ids'] = len(conversation_ids) == len(set(conversation_ids))
        if not validation['checks']['unique_conv_ids']:
            validation['valid'] = False
            validation['issues'].append(
                f"Duplicate conversation IDs found: "
                f"{len(conversation_ids)} total, {len(set(conversation_ids))} unique"
            )
        
        # Check unique scenario IDs
        validation['checks']['unique_scenario_ids'] = (
            len(scenario_ids) == len(set(scenario_ids))
        )
        if not validation['checks']['unique_scenario_ids']:
            validation['valid'] = False
            validation['issues'].append(
                f"Duplicate scenario IDs found"
            )
        
        # Check balanced liar distribution (25% each, allow 10% tolerance)
        total = sum(liar_counts.values())
        if total > 0:
            for liar, count in liar_counts.items():
                ratio = count / total
                if ratio < 0.15 or ratio > 0.35:
                    validation['valid'] = False
                    validation['issues'].append(
                        f"Liar {liar} ratio {ratio:.2%} outside [15%, 35%]"
                    )
        
        # Check balanced lie types
        total_lies = sum(lie_type_counts.values())
        if total_lies > 0:
            for lie_type, count in lie_type_counts.items():
                ratio = count / total_lies
                if ratio < 0.10 or ratio > 0.35:
                    validation['valid'] = False
                    validation['issues'].append(
                        f"Lie type {lie_type} ratio {ratio:.2%} outside [10%, 35%]"
                    )
        
        # Check scenario family separation (train vs validation)
        train_families = set(scenario_ids[:int(len(scenario_ids) * 0.8)])
        val_families = set(scenario_ids[int(len(scenario_ids) * 0.8):])
        overlap = train_families & val_families
        validation['checks']['no_scenario_overlap'] = len(overlap) == 0
        if not validation['checks']['no_scenario_overlap']:
            validation['valid'] = False
            validation['issues'].append(
                f"Scenario family overlap between train and validation: {overlap}"
            )
        
        # Check ground truth has claim context
        validation['checks']['claim_context_present'] = all(
            gt.get('claim', {}).get('context') for gt in ground_truth
        )
        if not validation['checks']['claim_context_present']:
            validation['valid'] = False
            validation['issues'].append("Some ground truth records have null claim context")
        
        validation['statistics']['liar_distribution'] = liar_counts
        validation['statistics']['lie_type_distribution'] = lie_type_counts
        validation['statistics']['message_count_distribution'] = {
            'min': min(message_counts) if message_counts else 0,
            'max': max(message_counts) if message_counts else 0,
            'avg': validation['statistics']['avg_messages_per_conversation']
        }
        validation['statistics']['speaker_distribution'] = speaker_counts
        
        validation['valid'] = (
            validation['statistics']['invalid_conversations'] == 0 and
            validation['checks'].get('unique_conv_ids', False) and
            validation['checks'].get('record_count_match', False)
        )
        
        self._validations.append(validation)
        return validation
    
    def get_all_validations(self) -> List[Dict]:
        """Return all validations performed."""
        return self._validations.copy()
    
    def get_summary(self) -> Dict:
        """Get summary of all validations."""
        summary = {
            'total_validations': len(self._validations),
            'total_valid': 0,
            'total_invalid': 0,
            'issues': []
        }
        
        for validation in self._validations:
            if validation.get('valid', False):
                summary['total_valid'] += 1
            else:
                summary['total_invalid'] += 1
                summary['issues'].extend(validation.get('issues', []))
        
        return summary


def validate_conversation_sample(
    conversation: List[Dict],
    characters: List[str],
    expected_liar: str
) -> Dict:
    """Convenience function to validate a conversation."""
    validator = Validator()
    return validator.validate_conversation(
        conversation=conversation,
        characters=characters,
        expected_liar=expected_liar
    )
