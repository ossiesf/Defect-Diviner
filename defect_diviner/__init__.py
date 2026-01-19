"""
Defect Diviner - Bug Prediction from Code History
==================================================

A machine learning pipeline that predicts which files are likely to contain bugs
based on code metrics AND historical/social features.

Key insight: Code metrics alone (complexity, LOC) don't predict bugs well.
Adding historical features (past bugs, contributors, churn) dramatically improves results.
"""

from .config import (
    DEFAULT_REPOS,
    MAX_COMMITS,
    GITHUB_TOKEN,
    ALL_FEATURE_COLS,
    OPTIMAL_FEATURE_COLS,
)

from .features import (
    FileFeatures,
    extract_code_metrics,
)

from .github import (
    GitHubIssueChecker,
    GitHubPRChecker,
    parse_repo_url,
)

from .extraction import (
    extract_features,
    build_file_history,
)

from .model import (
    train_and_evaluate,
    select_features,
    compare_models,
    full_analysis,
)

from .diagnostics import diagnose_repo

__version__ = "0.2.0"

__all__ = [
    # Config
    "DEFAULT_REPOS",
    "MAX_COMMITS",
    "GITHUB_TOKEN",
    "ALL_FEATURE_COLS",
    "OPTIMAL_FEATURE_COLS",
    # Features
    "FileFeatures",
    "extract_code_metrics",
    # GitHub
    "GitHubIssueChecker",
    "GitHubPRChecker",
    "parse_repo_url",
    # Extraction
    "extract_features",
    "build_file_history",
    # Model
    "train_and_evaluate",
    "select_features",
    "compare_models",
    "full_analysis",
    # Diagnostics
    "diagnose_repo",
]
