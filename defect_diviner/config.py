"""
Configuration and constants for Defect Diviner.
"""

import os
import re

# =============================================================================
# REPOSITORY SETTINGS
# =============================================================================

DEFAULT_REPOS = [
    "https://github.com/pallets/click",
    "https://github.com/psf/requests",
]

MAX_COMMITS = 300

# =============================================================================
# BUG DETECTION PATTERNS
# =============================================================================

BUG_KEYWORDS = re.compile(r'\b(fix|bug|patch|error|crash|fail)\w*\b', re.IGNORECASE)

# Expanded exclusions to reduce label noise
EXCLUDE_KEYWORDS = re.compile(
    r'\b(typo|doc|style|format|merge|revert|readme|changelog|comment|example|sample|'
    r'ci|workflow|badge|link|test|lint|typing|type.hint|annotation|deprecat)\b',
    re.IGNORECASE
)

# Regex to find issue references in commit messages
# Matches: #123, fixes #123, fix #123, closes #123, resolves #123, etc.
ISSUE_REFERENCE = re.compile(
    r'(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)?[\s:]*#(\d+)',
    re.IGNORECASE
)

# Labels that indicate a bug (case-insensitive matching)
BUG_LABELS = {'bug', 'bugfix', 'bug-fix', 'defect', 'error', 'issue', 'fix'}

# =============================================================================
# GITHUB API SETTINGS
# =============================================================================

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API_BASE = 'https://api.github.com'

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

# All available features for model training
ALL_FEATURE_COLS = [
    # Code metrics
    'loc', 'sloc', 'avg_complexity', 'max_complexity', 'maintainability_index',
    # Historical features
    'num_contributors', 'num_commits', 'num_bug_fixes', 'file_age_commits',
    'lines_added', 'lines_deleted',
    # New features (v2)
    'import_count', 'max_nesting_depth', 'commit_hour', 'author_commits',
    # Quick-win social features
    'files_in_commit', 'day_of_week', 'is_first_contribution', 'commit_msg_length',
    # PR review features
    'review_count', 'approval_count', 'changes_requested', 'pr_hours_open',
]

# Optimal feature set determined by greedy forward selection
# These features improve CV ROC AUC; others add noise
# Run `python tests/test_features.py` to re-evaluate
OPTIMAL_FEATURE_COLS = [
    'num_bug_fixes',     # files with past bugs get more bugs (strongest signal)
    'file_age_commits',  # older files are more stable
    'lines_deleted',     # code churn indicator
]
