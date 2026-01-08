# Defect Diviner

**Predicting buggy code using ML - and learning why code metrics alone don't work.**

## The Journey

I set out to build a bug predictor using code complexity metrics. What I learned: **historical features matter more than code metrics.**

| Approach | ROC AUC |
|----------|---------|
| Code metrics only | 0.54 (random) |
| + Historical features | **0.75** |

Read the full story: [PROJECT_STORY.md](PROJECT_STORY.md)

## Quick Start

```bash
pip install pydriller radon xgboost pandas scikit-learn

python defect_predictor.py --repos https://github.com/pallets/click
```

## How It Works

1. **Mine commits** from GitHub repos using PyDriller
2. **Extract features** for each Python file:
   - Code metrics (complexity, LOC, maintainability)
   - Historical features (contributors, past bugs, churn)
3. **Train XGBoost** to classify buggy vs clean files
4. **Evaluate** with cross-validation

## Key Findings

**Top predictive features:**
1. `num_bug_fixes` - Files with past bugs get more bugs
2. `lines_added` - High churn correlates with defects
3. `num_contributors` - More authors = more coordination issues

**What doesn't work:**
- Code complexity alone (correlation < 0.02)
- File-level metrics for 1-2 line bug fixes

## Results

```
Accuracy: 69%
F1 Score: 0.52
ROC AUC:  0.75
```

## Tech Stack

- **PyDriller** - Git repository mining
- **radon** - Python code metrics
- **XGBoost** - Gradient boosting classifier
- **pandas / scikit-learn** - Data pipeline

## Files

```
defect_predictor.py   # Main pipeline
PROJECT_STORY.md      # Full writeup (blog-ready)
LINKEDIN_POST.md      # Short-form summary
```

## Author

Ossie - Senior SDET exploring ML for software quality

## License

MIT
