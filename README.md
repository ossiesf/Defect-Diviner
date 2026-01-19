# Defect Diviner

**Predicting buggy code using ML - and learning why less is more.**

## The Journey

I set out to build a bug predictor with 19 features: code complexity, PR reviews, social signals. What I learned: **most features add noise, not signal.**

| Feature Set | CV ROC AUC |
|-------------|------------|
| All 19 features | 0.66 |
| **Optimal 3 features** | **0.75** |

One feature (`num_bug_fixes`) alone outperforms the full feature set.

Read the full story: [PROJECT_STORY.md](PROJECT_STORY.md)

## Quick Start

```bash
pip install pydriller radon xgboost pandas scikit-learn requests

# Run with optimal model (recommended)
python defect_predictor.py --repos https://github.com/pallets/click --optimal

# Run with all features
python defect_predictor.py --repos https://github.com/pallets/click
```

## How It Works

1. **Mine commits** from GitHub repos using PyDriller
2. **Extract features** - code metrics, history, PR reviews
3. **Train XGBoost** to classify buggy vs clean files
4. **Evaluate** with 5-fold cross-validation

## Key Findings

**What works (the optimal 3):**
```
num_bug_fixes     67%  - Files with past bugs get more bugs
file_age_commits  21%  - Older files are more stable
lines_deleted     12%  - Code churn indicator
```

**What doesn't work (at this data size):**
- Code complexity metrics
- PR review counts
- Day of week / commit hour
- Commit message length
- And 12 other features...

## Results

```
Optimal Model (3 features):
  CV ROC AUC:  0.746 (+/- 0.166)
  Accuracy:    77%

All Features (19):
  CV ROC AUC:  0.658 (+/- 0.246)  # worse!
```

## Project Structure

```
defect_diviner/          # Main package
├── config.py            # Settings, OPTIMAL_FEATURE_COLS
├── github.py            # GitHub API clients
├── features.py          # Feature extraction
├── extraction.py        # Repository mining
├── model.py             # Training & evaluation
└── diagnostics.py       # Repo quality analysis

tests/
├── test_unit.py         # Unit tests
└── test_features.py     # Feature evaluation suite

defect_predictor.py      # CLI entry point
```

## Usage

```bash
# Analyze repos with optimal model
python defect_predictor.py --repos https://github.com/user/repo --optimal

# Diagnose repo quality
python defect_predictor.py --diagnose --repos https://github.com/user/repo

# No GitHub API (regex-only bug detection)
python defect_predictor.py --no-github

# Run tests
python -m pytest tests/ -v

# Feature evaluation report
python tests/test_features.py
```

## Tech Stack

- **PyDriller** - Git repository mining
- **radon** - Python code metrics
- **XGBoost** - Gradient boosting classifier
- **pandas / scikit-learn** - Data pipeline
- **requests** - GitHub API

## Lessons Learned

1. **History beats complexity** - Past bugs predict future bugs better than code metrics
2. **More features ≠ better** - With 1k samples, most features add noise
3. **CV > test split** - Single splits are misleading (0.93 vs 0.66 ROC AUC)
4. **Evaluate rigorously** - Run ablation studies before adding features

## Author

Ossie - Senior SDET exploring ML for software quality

## License

MIT
