"""Validation for synthetic conversation data."""

from typing import List, Dict, Set
import re


class Validator:
    """Validates synthetic conversation data."""
    
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
        chars = set(msg['speaker'] for msg in conversation)
        validation['checks']['four_characters'] = len(chars) == 4
        if not validation['checks']['four_characters']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected 4 characters, got {len(chars)}: {chars}"
            )
        
        # Check character names
        expected_chars = set(characters)
        validation['checks']['correct_characters'] = chars == expected_chars
        if not validation['checks']['correct_characters']:
            validation['valid'] = False
            validation['issues'].append(
                f"Expected characters {expected_chars}, got {chars}"
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
        
        # Check exactly one liar
        liars = set(
            msg['speaker'] for msg in conversation
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
        
        # Check message content
        validation['checks']['messages_have_content'] = all(
            len(msg.get('message', '').strip()) > 10
            for msg in conversation
        )
        if not validation['checks']['messages_have_content']:
            validation['valid'] = False
            validation['issues'].append("Some messages lack sufficient content")
        
        # Check no ground truth leakage
        validation['checks']['no_ground_truth'] = not any(
            'true' in msg.get('message', '').lower() or
            'false' in msg.get('message', '').lower() or
            'lie' in msg.get('message', '').lower() or
            'liar' in msg.get('message', '').lower() or
            'truth' in msg.get('message', '').lower()
            for msg in conversation
        )
        if not validation['checks']['no_ground_truth']:
            validation['valid'] = False
            validation['issues'].append("Ground truth or liar identity revealed in messages")
        
        self._validations.append(validation)
        return validation
    
    def validate_dataset(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict]
    ) -> Dict:
        """Validate entire dataset."""
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
            }
        }
        
        total_messages = 0
        
        for conv, scenario in zip(conversations, scenarios):
            conv_validation = self.validate_conversation(
                conversation=conv,
                characters=scenario['characters'],
                expected_liar=scenario['liar']
            )
            
            validation['conversation_validations'].append(conv_validation)
            total_messages += len(conv)
            
            if conv_validation['valid']:
                validation['statistics']['valid_conversations'] += 1
            else:
                validation['statistics']['invalid_conversations'] += 1
                validation['issues'].extend(conv_validation['issues'])
        
        validation['statistics']['total_messages'] = total_messages
        if conversations:
            validation['statistics']['avg_messages_per_conversation'] = (
                total_messages / len(conversations)
            )
        
        validation['valid'] = (
            validation['statistics']['invalid_conversations'] == 0
        )
        
        self._validations.append(validation)
        return validation
    
    def validate_scenario(
        self,
        scenario: Dict
    ) -> Dict:
        """Validate a single scenario."""
        validation = {
            'valid': True,
            'issues': [],
            'checks': {}
        }
        
        # Check required fields
        required = ['liar', 'characters', 'claim', 'world']
        for field in required:
            validation['checks'][f'has_{field}'] = field in scenario
            if not validation['checks'][f'has_{field}']:
                validation['valid'] = False
                validation['issues'].append(f"Missing required field: {field}")
        
        # Check 4 characters
        if 'characters' in scenario:
            validation['checks']['four_characters'] = len(scenario['characters']) == 4
            if not validation['checks']['four_characters']:
                validation['valid'] = False
                validation['issues'].append(
                    f"Expected 4 characters, got {len(scenario['characters'])}"
                )
        
        # Check liar is one of the characters
        if 'characters' in scenario and 'liar' in scenario:
            validation['checks']['liar_in_characters'] = (
                scenario['liar'] in scenario['characters']
            )
            if not validation['checks']['liar_in_characters']:
                validation['valid'] = False
                validation['issues'].append(
                    f"Liar '{scenario['liar']}' not in characters {scenario['characters']}"
                )
        
        # Check claim exists
        if 'claim' in scenario:
            claim = scenario['claim']
            if isinstance(claim, dict):
                validation['checks']['claim_has_false_claim'] = 'false_claim' in claim
                if not validation['checks']['claim_has_false_claim']:
                    validation['valid'] = False
                    validation['issues'].append("Claim is missing 'false_claim' field")
        
        self._validations.append(validation)
        return validation
    
    def validate_no_ground_truth_leakage(
        self,
        conversations: List[List[Dict]],
        ground_truth: List[Dict]
    ) -> Dict:
        """Validate that model input doesn't contain ground truth."""
        validation = {
            'valid': True,
            'issues': [],
            'checks': {}
        }
        
        # Check that labels are not in model input
        validation['checks']['no_labels_in_input'] = True
        for conv in conversations:
            for msg in conv:
                if 'label' in msg or 'is_liar' in msg:
                    validation['valid'] = False
                    validation['checks']['no_labels_in_input'] = False
                    validation['issues'].append(
                        "Label or is_liar field found in conversation message"
                    )
        
        # Check ground truth doesn't contain text that should be model input
        validation['checks']['ground_truth_separated'] = True
        for gt in ground_truth:
            if 'text' in gt and 'label' in gt:
                validation['valid'] = False
                validation['checks']['ground_truth_separated'] = False
                validation['issues'].append(
                    "Ground truth contains both text and label"
                )
        
        self._validations.append(validation)
        return validation
    
    def validate_balanced_lies(
        self,
        conversations: List[List[Dict]],
        scenarios: List[Dict]
    ) -> Dict:
        """Validate that lies are balanced across types."""
        validation = {
            'valid': True,
            'issues': [],
            'checks': {},
            'lie_type_counts': {}
        }
        
        # Count lie types
        lie_type_counts = {}
        for scenario in scenarios:
            claim = scenario.get('claim', {})
            lie_type = claim.get('lie_type', 'unknown')
            lie_type_counts[lie_type] = lie_type_counts.get(lie_type, 0) + 1
        
        validation['checks']['lie_type_counts'] = lie_type_counts
        
        # Check balance (no type should be < 10% or > 40%)
        total = sum(lie_type_counts.values())
        for lie_type, count in lie_type_counts.items():
            ratio = count / total if total > 0 else 0
            if ratio < 0.1 or ratio > 0.4:
                validation['valid'] = False
                validation['issues'].append(
                    f"Lie type '{lie_type}' ratio {ratio:.2%} outside [10%, 40%]"
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
