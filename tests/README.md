# Liar's Chatroom - Tests

This directory contains test fixtures for the competition toolkit.

## Fixtures

- `conftest.py`: Pytest configuration and shared fixtures
- `test_*.py`: Unit tests for each module

## Running Tests

```bash
pytest
```

## Test Results

**39 tests passed** ✅

### Test Categories

- `test_data_utils.py`: Data loading (CSV/JSON/JSONL), duplicate detection, schema detection
- `test_validation.py`: Stratified and group-aware splits, leakage detection, CV scores
- `test_training.py`: Model training and evaluation (TF-IDF + LogisticRegression)
- `test_submission.py`: Submission format verification and generation
- `test_inspect_data.py`: Data inspection and profiling CLI
