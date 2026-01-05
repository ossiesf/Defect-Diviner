# Defect-Diviner - Work In Progress

## Overview

Defect-Diviner analyzes Python source code and predicts the likelihood of defects using complexity metrics and machine learning. The model is trained on real-world bugs from popular open-source Python projects.

**Key Features:**
- Trained on XXX+ real defects from production Python projects
- Uses industry-standard code metrics (cyclomatic complexity, maintainability index, Halstead metrics)
- Built with scikit-learn for robust, interpretable predictions

## Dataset

Training data extracted from [BugsInPy](https://github.com/soarsmu/BugsInPy), a curated database of real bugs from Python projects.

**Metrics extracted using [radon](https://github.com/rubik/radon):**
- Cyclomatic complexity (average, max, total)
- Lines of code (LOC, LLOC, SLOC)
- Maintainability index
- Halstead metrics (volume, difficulty, effort, bugs)
- Comment ratios

## Results

**Model Performance:**
- Best model: [TBD]
- Precision: [TBD]
- Recall: [TBD]
- F1-score: [TBD]

**Top Predictive Features:**
1. [TBD]
2. [TBD]
3. [TBD]

## Repository Structure
```
defect-diviner/
├── data/
│   └── tbd
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── extract_metrics.py
│   └── train_model.py
├── models/
│   └── defect_predictor.pkl
└── README.md
```

## Caveats

- Trained only on Python code
- File-level predictions (not function-level)
- Static analysis only (no temporal features)

## Technical Stack

Python 3.12, scikit-learn, radon, pandas, BugsInPy

## Author

Ossie - Senior SDET with 10+ years experience in test automation and software quality.

## License

MIT