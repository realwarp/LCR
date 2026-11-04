"""Tests for synthetic validation logic."""

import pytest
from synthetic.validator import Validator


class TestValidationLogic:
    """Tests for validation logic."""
    
    def test_valid_conversation_passes(self):
        """Test that a valid conversation passes validation."""
        validator = Validator()
        
        conv = [
            {'speaker': 'A', 'message': 'Hello', 'timestamp': 0},
            {'speaker': 'B', 'message': 'Hi', 'timestamp': 1},
            {'speaker': 'C', 'message': 'Hey', 'timestamp': 2},
            {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        # Should pass character and liar checks
        assert result['checks']['four_characters'] is True
    
    def test_wrong_number_of_characters_fails(self):
        """Test that wrong character count fails validation."""
        validator = Validator()
        
        conv = [
            {'speaker': 'A', 'message': 'Hello', 'timestamp': 0},
            {'speaker': 'B', 'message': 'Hi', 'timestamp': 1},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        assert result['checks']['four_characters'] is False
    
    def test_wrong_liar_fails(self):
        """Test that wrong liar fails validation."""
        validator = Validator()
        
        conv = [
            {'speaker': 'A', 'message': 'Hello', 'is_liar': True, 'timestamp': 0},
            {'speaker': 'B', 'message': 'Hi', 'is_liar': False, 'timestamp': 1},
            {'speaker': 'C', 'message': 'Hey', 'is_liar': False, 'timestamp': 2},
            {'speaker': 'D', 'message': 'Hi there', 'is_liar': False, 'timestamp': 3},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='B'  # Wrong liar
        )
        
        assert result['checks']['correct_liar'] is False


class TestGroundTruthLeakage:
    """Tests for ground truth leakage detection."""
    
    def test_no_leakage_passes(self):
        """Test that no ground truth leakage passes."""
        validator = Validator()
        
        conv = [
            {'speaker': 'A', 'message': 'The sky is blue', 'is_liar': True, 'timestamp': 0},
            {'speaker': 'B', 'message': 'I think it is', 'is_liar': False, 'timestamp': 1},
            {'speaker': 'C', 'message': 'Interesting', 'is_liar': False, 'timestamp': 2},
            {'speaker': 'D', 'message': 'Sure', 'is_liar': False, 'timestamp': 3},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        assert result['checks']['no_ground_truth'] is True
    
    def test_ground_truth_leakage_detected(self):
        """Test that ground truth leakage is detected."""
        validator = Validator()
        
        conv = [
            {'speaker': 'A', 'message': 'I am lying about the sky', 'is_liar': True, 'timestamp': 0},
            {'speaker': 'B', 'message': 'I know you are lying', 'is_liar': False, 'timestamp': 1},
            {'speaker': 'C', 'message': 'A is the liar', 'is_liar': False, 'timestamp': 2},
            {'speaker': 'D', 'message': 'True', 'is_liar': False, 'timestamp': 3},
        ]
        
        result = validator.validate_conversation(
            conversation=conv,
            characters=['A', 'B', 'C', 'D'],
            expected_liar='A'
        )
        
        assert result['checks']['no_ground_truth'] is False


class TestDatasetValidation:
    """Tests for dataset-level validation."""
    
    def test_empty_dataset_valid(self):
        """Test that empty dataset is considered valid."""
        validator = Validator()
        
        result = validator.validate_dataset([], [])
        
        assert result['valid'] is True
        assert result['statistics']['total_conversations'] == 0
    
    def test_multiple_conversations_validated(self):
        """Test validation of multiple conversations."""
        validator = Validator()
        
        conversations = [[
            {'speaker': 'A', 'message': 'Hello', 'timestamp': 0},
            {'speaker': 'B', 'message': 'Hi', 'timestamp': 1},
            {'speaker': 'C', 'message': 'Hey', 'timestamp': 2},
            {'speaker': 'D', 'message': 'Hi there', 'timestamp': 3},
        ]]
        
        scenarios = [{'liar': 'A', 'characters': ['A', 'B', 'C', 'D']}]
        
        result = validator.validate_dataset(conversations, scenarios)
        
        assert result['statistics']['total_conversations'] == 1
