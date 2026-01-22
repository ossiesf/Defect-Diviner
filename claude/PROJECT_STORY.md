# Defect Diviner: What I Learned Building a Bug Predictor

## The Idea

I wanted to build an ML model that predicts which code files are likely to contain bugs. The plan seemed straightforward: extract code metrics (complexity, lines of code, maintainability scores) and train a classifier. Simple, right?

## The First Attempt: Code Metrics

I used **radon** to extract metrics from Python files:
- Cyclomatic complexity
- Lines of code (LOC, SLOC)
- Maintainability Index
- Halstead metrics (cognitive complexity)

Data source: I mined bug-fixing commits from GitHub using **PyDriller**. Files before a bug fix = "buggy", files after = "clean".

**Results: Basically random guessing.**

| Metric | Value |
|--------|-------|
| Accuracy | 53% |
| F1 Score | 0.05 |
| ROC AUC | 0.54 |

The models couldn't distinguish buggy from clean code. Why?

## The Problem

When I looked at the actual data, buggy and clean versions of the same file had nearly **identical metrics**. Bug fixes are often just 1-2 line changes - not enough to move the needle on file-level complexity.

Feature correlations with bugs were all under 0.02. The signal simply wasn't there.

## The Insight

I remembered reading that bugs aren't random - they cluster in certain files. Files that had bugs before tend to get more bugs. So I added **historical features**:

- `num_contributors` - how many developers touched this file
- `num_bug_fixes` - past bug fixes in this file
- `lines_added/deleted` - churn in this commit
- `file_age` - how long the file has existed

## The Result

| Model | Accuracy | F1 | ROC AUC |
|-------|----------|-----|---------|
| Code metrics only | 53% | 0.26 | 0.54 |
| **+ Historical features** | **69%** | **0.52** | **0.75** |

Adding historical context improved every metric dramatically. The top predictor? **`num_bug_fixes`** - files with past bugs get more bugs.

## Key Takeaways

1. **Domain knowledge matters more than fancy models.** XGBoost with the right features beat XGBoost with the wrong features.

2. **Code metrics alone don't predict bugs.** Decades of research backs this up - they're useful for code quality assessment, not bug prediction.

3. **History repeats itself.** The best predictor of future bugs is past bugs. This is called "defect density persistence" in the literature.

4. **Sometimes the interesting result is negative.** Understanding *why* something doesn't work is often more valuable than a demo that "works" on cherry-picked data.

## Tech Stack

- **PyDriller** - Git repository mining
- **radon** - Python code metrics
- **XGBoost** - Classification model
- **pandas/scikit-learn** - Data pipeline

## What I'd Do Differently

With more time, I'd explore:
- Code embeddings (CodeBERT) for semantic features
- Developer experience metrics (is this their first commit to this file?)
- Time-based features (day of week, time since last release)

## Try It Yourself

```bash
pip install pydriller radon xgboost pandas scikit-learn
python defect_predictor.py --repos https://github.com/your/repo
```

---

*Built as a portfolio project exploring the intersection of ML and software engineering.*
