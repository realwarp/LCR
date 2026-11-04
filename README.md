# The Liar's Chatroom — Competition Starter Kit

A lightweight Python toolkit for inspecting data, validating strategies, training baselines, and generating submissions for The Liar's Chatroom competition at IIIT Bangalore.

---

## Quick start (Windows PowerShell)

```powershell
# Navigate to project directory
cd C:\Users\ashwi\workplace\liars-chatroom-starter

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Inspect a data file
python -m src.inspect_data data/raw/train.csv

# Run all tests
pytest
```

---

## Project structure

```
liars-chatroom-starter/
├── config/
│   └── competition.example.yaml      # Configuration template
├── data/
│   ├── raw/                          # Raw competition files (git-ignored)
│   └── processed/                    # Processed datasets (git-ignored)
├── models/                           # Trained models (git-ignored)
├── reports/
│   ├── experiment_log.md             # Experiment tracking table
│   └── data_profile.md               # Generated data profile (by inspect_data)
├── src/
│   ├── __init__.py
│   ├── data_utils.py                 # Loading, combining, duplicate detection
│   ├── validation.py                 # Stratified and group-aware splits
│   ├── inspect_data.py               # CLI to profile CSV/JSON/JSONL files
│   ├── train_baseline.py             # TF-IDF + LogisticRegression baseline
│   ├── compare_models.py             # Controlled word vs. character TF-IDF comparison
│   └── make_submission.py            # Generate submission.csv from a saved model
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/                     # Small synthetic test datasets
│   │   └── __init__.py
│   ├── test_data_utils.py
│   ├── test_validation.py
│   ├── test_training.py
│   ├── test_submission.py
│   └── test_inspect_data.py
├── .gitignore
├── requirements.txt
├── README.md
└── experiment_log.md
```

---

## Configuration

Copy `config/competition.example.yaml` to `config/competition.yaml` and fill in paths:

```yaml
paths:
  data_dir: "data/raw"
  models_dir: "models"
  reports_dir: "reports"

files:
  train: "train.csv"
  test: "test.csv"
  evidence: "evidence.csv"
  sample_submission: "sample_submission.csv"

split:
  strategy: "group_aware"   # "stratified" or "group_aware"
  test_size: 0.2
  random_state: 42
  group_column: "conversation_id"  # adjust to your schema

models:
  word_tfidf:
    ngram_range: [1, 2]
    max_features: 50000
  char_tfidf:
    ngram_range: [3, 5]
    max_features: 100000
```

---

## Commands

### Inspect files

```powershell
# CSV file
python -m src.inspect_data data/raw/train.csv

# JSONL file
python -m src.inspect_data data/raw/evidence.jsonl

# Output: data_profile.md + data_profile.json in the same folder
```

### Run baselines

```powershell
# Train baseline TF-IDF + LogisticRegression
python -m src.train_baseline

# Compare word vs. character TF-IDF in a controlled experiment
python -m src.compare_models
```

### Generate submission

```powershell
# Use a saved model to create submission.csv
python -m src.make_submission --model models/word_tfidf_lr_v1.pkl --output submissions/submission_v1.csv
```

### Run tests

```powershell
pytest
```

---

## Validation notes

- **Group-aware splits** reduce conversation leakage when multiple rows belong to the same `conversation_id`.
- **Stratified splits** preserve class proportions and are useful when labels are balanced within groups.
- Run `python -m src.validation.leakage_report` after training to detect group leakage in your splits.
- Always verify that `submission.csv` matches the official sample schema before uploading.

---

## PowerShell examples

```powershell
# List files in data/raw
Get-ChildItem data\raw\

# View profile HTML in browser
Start-Process .\reports\data_profile.md

# Quick test run (skip slow tests)
pytest -m "not slow"

# Run only validation tests
pytest tests/test_validation.py -v
```

---

## Expected inputs (when official data arrives)

| File                  | Expected columns                       |
|-----------------------|----------------------------------------|
| `train.csv`           | `id`, `conversation_id`, `character`, `text`, `label` |
| `test.csv`            | `id`, `conversation_id`, `character`, `text`          |
| `evidence.csv`        | `id`, `evidence_text`                                  |
| `sample_submission`   | `id`, `label`                                          |

The `inspect_data` module will adapt to your actual schema and report detected columns.

---

## License

This toolkit is provided for the The Liar's Chatroom competition at Synergy '26, IIIT Bangalore.
