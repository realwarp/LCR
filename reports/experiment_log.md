# Experiment Log

This document tracks all experiments run during the competition.

## Setup Complete (2026-11-04)

Initial project setup completed. All tests pass.

| ID | Date | Model | Features | Split Strategy | Train Acc | Val Acc | CV Score | Notes | Files |
|----|------|-------|----------|----------------|-----------|---------|----------|-------|-------|
|    |      |       |          |                |           |         |          |       |       |

## Documentation Instructions

1. **Fill in each row** after completing an experiment
2. **ID**: Sequential number (exp_001, exp_002, etc.)
3. **Date**: ISO format (YYYY-MM-DD)
4. **Model**: Full model name (e.g., LogisticRegression, RandomForest)
5. **Features**: Describe features used (e.g., "word_ngram_1_2, char_ngram_3_5")
6. **Split Strategy**: How data was split (e.g., "stratified_0.8", "group_aware_0.7")
7. **Train Acc**: Training accuracy from final epoch/model
8. **Val Acc**: Validation accuracy on hold-out set
9. **CV Score**: Cross-validation score (mean ± std)
10. **Notes**: Any observations, problems, or next steps
11. **Files**: Link to relevant artifacts (model file, submission file, etc.)

## Experiment Tracking Checklist

- [ ] Run completed
- [ ] All metrics recorded
- [ ] Model file saved
- [ ] Submission file created (optional)
- [ ] Notes written
- [ ] This log updated

## Commands

Run experiments with:

```bash
# Train baseline
python -m src.train_baseline

# Compare models
python -m src.compare_models

# Generate submission
python -m src.make_submission
```

## Tests

Run tests to verify setup:

```bash
pytest
```

Test results: **39 tests passed**

---

## Current Project Status

- ✅ Project directory structure created
- ✅ requirements.txt with dependencies
- ✅ config/competition.example.yaml template
- ✅ src/ module with data_utils.py, validation.py, inspect_data.py, train_baseline.py, compare_models.py, make_submission.py
- ✅ tests/ module with conftest.py and 6 test files
- ✅ README.md documentation
- ✅ reports/experiment_log.md tracking file
- ✅ All tests passing
