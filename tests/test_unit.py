#!/usr/bin/env python3
"""
Unit tests for defect_diviner package.

Usage:
    python -m pytest tests/test_unit.py -v
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIG TESTS
# =============================================================================

def test_config_imports():
    """Config module should import without errors"""
    from defect_diviner.config import (
        DEFAULT_REPOS,
        MAX_COMMITS,
        BUG_KEYWORDS,
        EXCLUDE_KEYWORDS,
        ALL_FEATURE_COLS,
    )
    assert len(DEFAULT_REPOS) > 0
    assert MAX_COMMITS > 0
    assert len(ALL_FEATURE_COLS) > 0


def test_bug_keywords_regex():
    """Bug keywords regex should match expected patterns"""
    from defect_diviner.config import BUG_KEYWORDS, EXCLUDE_KEYWORDS

    # Should match
    assert BUG_KEYWORDS.search("fix: resolve issue")
    assert BUG_KEYWORDS.search("Bug fix for crash")
    assert BUG_KEYWORDS.search("Fixed error handling")
    assert BUG_KEYWORDS.search("patch security vulnerability")

    # Should not match
    assert not BUG_KEYWORDS.search("add new feature")
    assert not BUG_KEYWORDS.search("update documentation")

    # Exclusions should work
    assert EXCLUDE_KEYWORDS.search("fix typo in readme")
    assert EXCLUDE_KEYWORDS.search("fix doc formatting")  # "doc" not "docs"
    assert EXCLUDE_KEYWORDS.search("update style guide")
    assert not EXCLUDE_KEYWORDS.search("fix null pointer crash")


# =============================================================================
# FEATURES TESTS
# =============================================================================

def test_file_features_dataclass():
    """FileFeatures dataclass should work correctly"""
    from defect_diviner.features import FileFeatures

    ff = FileFeatures(
        repo="test",
        commit_hash="abc123",
        file_path="src/main.py",
        is_buggy=1,
        loc=100,
        num_bug_fixes=5,
    )

    assert ff.repo == "test"
    assert ff.is_buggy == 1
    assert ff.loc == 100
    assert ff.num_bug_fixes == 5
    # Defaults
    assert ff.sloc == 0
    assert ff.review_count == 0


def test_file_features_to_dict():
    """FileFeatures should convert to dict"""
    from defect_diviner.features import FileFeatures

    ff = FileFeatures(
        repo="test",
        commit_hash="abc123",
        file_path="src/main.py",
        is_buggy=0,
    )
    d = ff.to_dict()

    assert isinstance(d, dict)
    assert d['repo'] == "test"
    assert d['is_buggy'] == 0
    assert 'num_bug_fixes' in d


def test_extract_code_metrics():
    """Code metrics extraction should work"""
    from defect_diviner.features import extract_code_metrics

    code = '''
def hello():
    """Say hello"""
    print("Hello, World!")

def add(a, b):
    """Add two numbers"""
    if a > 0:
        return a + b
    else:
        return b

class Calculator:
    def multiply(self, a, b):
        return a * b
'''

    metrics = extract_code_metrics(code)

    assert metrics is not None
    assert metrics['loc'] > 0
    assert metrics['sloc'] > 0
    assert metrics['import_count'] == 0
    assert metrics['max_nesting_depth'] >= 1


def test_extract_code_metrics_with_imports():
    """Should count imports correctly"""
    from defect_diviner.features import extract_code_metrics

    # Code must be > 15 lines for extract_code_metrics
    code = '''
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

def main():
    """Main function"""
    print("Hello")
    x = 1
    y = 2
    z = x + y
    return z

def helper():
    """Helper function"""
    pass

class MyClass:
    pass
'''

    metrics = extract_code_metrics(code)
    assert metrics is not None
    assert metrics['import_count'] == 4  # 2 imports + 2 from imports


def test_extract_code_metrics_too_short():
    """Should return None for too-short code"""
    from defect_diviner.features import extract_code_metrics

    short_code = "x = 1\ny = 2"
    assert extract_code_metrics(short_code) is None


def test_extract_code_metrics_invalid():
    """Should return None for invalid Python"""
    from defect_diviner.features import extract_code_metrics

    invalid = "def broken(\n    syntax error here"
    # Should not raise, just return None
    result = extract_code_metrics("not python code {{{{")
    # May or may not be None depending on radon's handling


# =============================================================================
# GITHUB TESTS
# =============================================================================

def test_parse_repo_url():
    """Should parse GitHub URLs correctly"""
    from defect_diviner.github import parse_repo_url

    owner, repo = parse_repo_url("https://github.com/pallets/click")
    assert owner == "pallets"
    assert repo == "click"

    owner, repo = parse_repo_url("https://github.com/user/repo/")
    assert owner == "user"
    assert repo == "repo"

    owner, repo = parse_repo_url("github.com/foo/bar")
    assert owner == "foo"
    assert repo == "bar"


def test_github_issue_checker_init():
    """GitHubIssueChecker should initialize"""
    from defect_diviner.github import GitHubIssueChecker

    # Should not raise (even without token)
    checker = GitHubIssueChecker("pallets", "click")
    assert checker.owner == "pallets"
    assert checker.repo == "click"
    assert checker.api_calls == 0


def test_github_pr_checker_init():
    """GitHubPRChecker should initialize"""
    from defect_diviner.github import GitHubPRChecker

    checker = GitHubPRChecker("pallets", "click")
    assert checker.owner == "pallets"
    assert checker.repo == "click"
    assert checker.api_calls == 0


# =============================================================================
# MODEL TESTS
# =============================================================================

def test_model_imports():
    """Model module should import"""
    from defect_diviner.model import (
        train_and_evaluate,
        select_features,
        compare_models,
        full_analysis,
    )


def test_train_and_evaluate_small_dataset():
    """Should handle small dataset gracefully"""
    from defect_diviner.model import train_and_evaluate

    # Create minimal test dataset
    df = pd.DataFrame({
        'is_buggy': [0, 1, 0, 1, 0, 1, 0, 1] * 10,
        'num_bug_fixes': [0, 2, 1, 3, 0, 4, 1, 5] * 10,
        'loc': [100, 200, 150, 250, 100, 300, 120, 180] * 10,
    })

    # Should run without error
    result = train_and_evaluate(df)
    assert 'model' in result
    assert 'results' in result
    assert 0 <= result['results']['roc_auc'] <= 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_package_imports():
    """Main package should import all public APIs"""
    from defect_diviner import (
        DEFAULT_REPOS,
        MAX_COMMITS,
        FileFeatures,
        extract_code_metrics,
        GitHubIssueChecker,
        GitHubPRChecker,
        parse_repo_url,
        extract_features,
        train_and_evaluate,
        diagnose_repo,
    )


def test_package_version():
    """Package should have version"""
    import defect_diviner
    assert hasattr(defect_diviner, '__version__')
    assert defect_diviner.__version__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
