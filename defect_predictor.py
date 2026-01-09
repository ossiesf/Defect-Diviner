"""
Defect Diviner - Bug Prediction from Code History
==================================================

A machine learning pipeline that predicts which files are likely to contain bugs
based on code metrics AND historical/social features.

Key insight: Code metrics alone (complexity, LOC) don't predict bugs well.
Adding historical features (past bugs, contributors, churn) dramatically improves results.

Usage:
    python defect_predictor.py --repo https://github.com/user/repo

Author: [Your Name]
"""

import argparse
import ast
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from pydriller import Repository
from radon.complexity import cc_visit
from radon.raw import analyze
from radon.metrics import mi_visit

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_REPOS = [
    "https://github.com/pallets/click",
    "https://github.com/psf/requests",
]

MAX_COMMITS = 300
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

# GitHub API configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API_BASE = 'https://api.github.com'

# Labels that indicate a bug (case-insensitive matching)
BUG_LABELS = {'bug', 'bugfix', 'bug-fix', 'defect', 'error', 'issue', 'fix'}


# =============================================================================
# GITHUB API INTEGRATION
# =============================================================================

class GitHubIssueChecker:
    """Fetch and cache GitHub issue labels to identify real bug fixes"""

    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.cache = {}  # issue_number -> {'is_bug': bool, 'labels': list}
        self.api_calls = 0
        self.cache_hits = 0

        self.session = requests.Session()
        if GITHUB_TOKEN:
            self.session.headers['Authorization'] = f'token {GITHUB_TOKEN}'
        self.session.headers['Accept'] = 'application/vnd.github.v3+json'
        self.session.headers['User-Agent'] = 'Defect-Diviner'

        # Check rate limit
        self._check_rate_limit()

    def _check_rate_limit(self):
        """Check and report GitHub API rate limit"""
        try:
            resp = self.session.get(f'{GITHUB_API_BASE}/rate_limit')
            if resp.status_code == 200:
                data = resp.json()
                remaining = data['resources']['core']['remaining']
                limit = data['resources']['core']['limit']
                print(f"  GitHub API: {remaining}/{limit} requests remaining", flush=True)
                if remaining < 100:
                    print(f"  WARNING: Low API quota. Set GITHUB_TOKEN env var for 5000/hr limit.")
        except Exception:
            pass

    def _fetch_issue(self, issue_num: int) -> dict | None:
        """Fetch a single issue from GitHub API"""
        if issue_num in self.cache:
            self.cache_hits += 1
            return self.cache[issue_num]

        try:
            url = f'{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/issues/{issue_num}'
            resp = self.session.get(url, timeout=10)
            self.api_calls += 1

            if resp.status_code == 200:
                data = resp.json()
                labels = [lbl['name'].lower() for lbl in data.get('labels', [])]
                is_bug = any(lbl in BUG_LABELS for lbl in labels)

                result = {'is_bug': is_bug, 'labels': labels}
                self.cache[issue_num] = result
                return result
            elif resp.status_code == 404:
                # Issue doesn't exist (might be a PR number)
                self.cache[issue_num] = {'is_bug': False, 'labels': []}
                return self.cache[issue_num]
            elif resp.status_code == 403:
                # Rate limited
                print(f"  Rate limited by GitHub API. Falling back to regex.", flush=True)
                return None

            # Rate limit: small delay between calls
            if self.api_calls % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            pass

        return None

    def is_bug_fix(self, commit_msg: str) -> tuple[bool, str]:
        """
        Determine if a commit is a bug fix using GitHub issue labels.

        Returns:
            (is_bug, method): is_bug is True/False, method is 'issue_label', 'regex', or 'none'
        """
        # First, try to find issue references
        issue_nums = ISSUE_REFERENCE.findall(commit_msg)

        if issue_nums:
            # Check each referenced issue
            for num_str in issue_nums:
                issue_num = int(num_str)
                result = self._fetch_issue(issue_num)

                if result and result['is_bug']:
                    return (True, 'issue_label')

            # Issues found but none labeled as bug
            # Still check regex as fallback (issue might not be labeled)
            if bool(BUG_KEYWORDS.search(commit_msg)) and not bool(EXCLUDE_KEYWORDS.search(commit_msg)):
                return (True, 'regex_fallback')
            return (False, 'issue_not_bug')

        # No issue references, fall back to regex
        if bool(BUG_KEYWORDS.search(commit_msg)) and not bool(EXCLUDE_KEYWORDS.search(commit_msg)):
            return (True, 'regex')

        return (False, 'none')

    def get_stats(self) -> dict:
        """Return API usage statistics"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cached_issues': len(self.cache),
            'bug_issues': sum(1 for v in self.cache.values() if v['is_bug']),
        }


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from GitHub URL"""
    # Handle: https://github.com/owner/repo or github.com/owner/repo
    parts = url.rstrip('/').split('/')
    return parts[-2], parts[-1]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FileFeatures:
    """Features extracted for a single file snapshot"""
    repo: str
    commit_hash: str
    file_path: str
    is_buggy: int

    # Code metrics (weak predictors alone)
    loc: int = 0
    sloc: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 0.0

    # Historical features (strong predictors!)
    num_contributors: int = 0
    num_commits: int = 0
    num_bug_fixes: int = 0
    file_age_commits: int = 0
    lines_added: int = 0
    lines_deleted: int = 0

    # New features
    import_count: int = 0           # Number of imports (dependency complexity)
    max_nesting_depth: int = 0      # Deepest nesting level
    commit_hour: int = 0            # Hour of commit (0-23, late night = risky)
    author_commits: int = 0         # Author's experience with this repo

    # Quick-win social features (no extra API calls)
    files_in_commit: int = 0        # Large commits = higher risk
    day_of_week: int = 0            # 0=Mon, 6=Sun (Friday deployments risky)
    is_first_contribution: int = 0  # 1 if author's first commit to repo
    commit_msg_length: int = 0      # Longer messages = better documentation


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_code_metrics(code: str) -> dict | None:
    """Extract code quality metrics using radon"""
    if not code or len(code) > 50000 or code.count('\n') < 15:
        return None
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        complexities = [b.complexity for b in cc] if cc else [0]
        return {
            'loc': raw.loc,
            'sloc': raw.sloc,
            'avg_complexity': round(sum(complexities) / len(complexities), 2),
            'max_complexity': max(complexities),
            'maintainability_index': round(mi_visit(code, True), 2),
            'import_count': count_imports(code),
            'max_nesting_depth': get_max_nesting_depth(code),
        }
    except Exception:
        return None


def count_imports(code: str) -> int:
    """Count the number of import statements in code"""
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom)))
    except Exception:
        return 0


def get_max_nesting_depth(code: str) -> int:
    """Calculate maximum nesting depth in code"""
    try:
        tree = ast.parse(code)
        return _calc_depth(tree)
    except Exception:
        return 0


def _calc_depth(node, current=0) -> int:
    """Recursively calculate nesting depth"""
    max_depth = current
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            max_depth = max(max_depth, _calc_depth(child, current + 1))
        else:
            max_depth = max(max_depth, _calc_depth(child, current))
    return max_depth


# =============================================================================
# REPO DIAGNOSTICS
# =============================================================================

def diagnose_repo(repo_url: str, sample_commits: int = 100) -> dict:
    """Analyze a repo's suitability for bug prediction training"""
    print(f"\n{'='*60}")
    print(f"REPO DIAGNOSTIC: {repo_url.split('/')[-1]}")
    print(f"{'='*60}")

    stats = {
        'total_commits': 0,
        'bug_fix_commits': 0,
        'python_files': set(),
        'other_files': set(),
        'contributors': set(),
        'commit_msg_lengths': [],
        'commit_hours': [],
        'files_per_commit': [],
    }

    for commit in Repository(repo_url, only_no_merge=True).traverse_commits():
        stats['total_commits'] += 1
        if stats['total_commits'] > sample_commits:
            break

        is_bug = bool(BUG_KEYWORDS.search(commit.msg)) and not bool(EXCLUDE_KEYWORDS.search(commit.msg))
        if is_bug:
            stats['bug_fix_commits'] += 1

        stats['contributors'].add(commit.author.email)
        stats['commit_msg_lengths'].append(len(commit.msg))
        stats['commit_hours'].append(commit.author_date.hour)
        stats['files_per_commit'].append(len(commit.modified_files))

        for mod in commit.modified_files:
            ext = Path(mod.filename).suffix
            if ext == '.py':
                stats['python_files'].add(mod.new_path or mod.old_path)
            else:
                stats['other_files'].add(ext)

    # Calculate quality scores
    bug_ratio = stats['bug_fix_commits'] / max(stats['total_commits'], 1)
    py_file_count = len(stats['python_files'])
    lang_count = len(stats['other_files']) + (1 if py_file_count > 0 else 0)
    avg_msg_len = np.mean(stats['commit_msg_lengths']) if stats['commit_msg_lengths'] else 0
    contributor_count = len(stats['contributors'])

    # Quality assessment
    quality_score = 0
    issues = []

    # Bug ratio (ideal: 10-30%)
    if 0.10 <= bug_ratio <= 0.30:
        quality_score += 25
    elif 0.05 <= bug_ratio <= 0.40:
        quality_score += 15
    else:
        issues.append(f"Bug ratio {bug_ratio:.1%} outside ideal range (10-30%)")

    # Python focus
    if lang_count <= 3:
        quality_score += 25
    else:
        issues.append(f"Multi-language repo ({lang_count} file types) - integration seams may confound results")

    # Commit message quality
    if avg_msg_len >= 30:
        quality_score += 25
    elif avg_msg_len >= 15:
        quality_score += 15
    else:
        issues.append(f"Short commit messages (avg {avg_msg_len:.0f} chars) - bug detection may be unreliable")

    # Enough Python files
    if py_file_count >= 20:
        quality_score += 25
    elif py_file_count >= 10:
        quality_score += 15
    else:
        issues.append(f"Only {py_file_count} Python files - may not provide enough samples")

    # Print report
    print(f"\nSampled: {stats['total_commits']} commits")
    print(f"\nMetrics:")
    print(f"  Bug-fix commits:     {stats['bug_fix_commits']:>4} ({bug_ratio:.1%})")
    print(f"  Python files:        {py_file_count:>4}")
    print(f"  Language types:      {lang_count:>4} ({', '.join(list(stats['other_files'])[:5]) or 'Python only'})")
    print(f"  Contributors:        {contributor_count:>4}")
    print(f"  Avg commit msg len:  {avg_msg_len:>4.0f} chars")

    print(f"\nQuality Score: {quality_score}/100")

    if quality_score >= 75:
        print("  ✓ GOOD - Suitable for training")
    elif quality_score >= 50:
        print("  ~ FAIR - Usable with caveats")
    else:
        print("  ✗ POOR - Consider alternatives")

    if issues:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  • {issue}")

    return {
        'quality_score': quality_score,
        'bug_ratio': bug_ratio,
        'python_files': py_file_count,
        'language_count': lang_count,
        'contributors': contributor_count,
        'issues': issues,
    }


def build_file_history(repo_url: str, issue_checker: GitHubIssueChecker = None) -> tuple[dict, dict, set]:
    """First pass: build historical features for each file and track author experience"""
    print(f"  Building file history...", flush=True)

    history = defaultdict(lambda: {
        'contributors': set(),
        'commits': 0,
        'bug_fixes': 0,
        'first_seen': 0,
    })

    # Track author experience (commits per author in this repo)
    author_experience = defaultdict(int)

    # Track first-time contributors (commit hashes where author made first contribution)
    first_contributions = set()

    commit_num = 0
    # Use reverse order to get recent commits (better issue labeling practices)
    for commit in Repository(repo_url, only_no_merge=True, order='reverse').traverse_commits():
        commit_num += 1
        if commit_num > MAX_COMMITS:
            break

        # Use GitHub issue labels if available, otherwise fall back to regex
        if issue_checker:
            is_bug, _ = issue_checker.is_bug_fix(commit.msg)
        else:
            is_bug = bool(BUG_KEYWORDS.search(commit.msg)) and not bool(EXCLUDE_KEYWORDS.search(commit.msg))

        # Track first-time contributors
        if author_experience[commit.author.email] == 0:
            first_contributions.add(commit.hash)

        author_experience[commit.author.email] += 1

        for mod in commit.modified_files:
            if not mod.filename.endswith('.py'):
                continue
            path = mod.new_path or mod.old_path
            h = history[path]
            h['contributors'].add(commit.author.email)
            h['commits'] += 1
            if is_bug:
                h['bug_fixes'] += 1
            if h['first_seen'] == 0:
                h['first_seen'] = commit_num

    return history, author_experience, first_contributions


def extract_features(repo_url: str, use_github_api: bool = True) -> list[dict]:
    """Extract all features from a repository"""
    repo_name = repo_url.split('/')[-1]
    print(f"\nProcessing: {repo_name}", flush=True)

    # Initialize GitHub issue checker for better bug detection
    issue_checker = None
    if use_github_api:
        try:
            owner, repo = parse_repo_url(repo_url)
            issue_checker = GitHubIssueChecker(owner, repo)
            print(f"  Using GitHub issue labels for bug detection", flush=True)
        except Exception as e:
            print(f"  GitHub API unavailable, using regex fallback: {e}", flush=True)

    history, author_experience, first_contributions = build_file_history(repo_url, issue_checker)
    results = []
    buggy, clean = 0, 0
    commit_num = 0

    # Track detection methods for stats
    detection_stats = defaultdict(int)

    print(f"  Extracting features...", flush=True)

    # Use reverse order to get recent commits (better issue labeling practices)
    for commit in Repository(repo_url, only_no_merge=True, order='reverse').traverse_commits():
        commit_num += 1
        if commit_num > MAX_COMMITS:
            break

        # Use GitHub issue labels if available, otherwise fall back to regex
        if issue_checker:
            is_bug, method = issue_checker.is_bug_fix(commit.msg)
            detection_stats[method] += 1
        else:
            is_bug = bool(BUG_KEYWORDS.search(commit.msg)) and not bool(EXCLUDE_KEYWORDS.search(commit.msg))
            detection_stats['regex' if is_bug else 'none'] += 1

        for mod in commit.modified_files:
            if not mod.filename.endswith('.py'):
                continue
            if 'test' in (mod.new_path or '').lower():
                continue
            if not mod.source_code_before or not mod.source_code:
                continue

            path = mod.new_path or mod.old_path
            h = history.get(path, {})
            code = mod.source_code_before if is_bug else mod.source_code
            metrics = extract_code_metrics(code)

            if not metrics:
                continue

            record = FileFeatures(
                repo=repo_name,
                commit_hash=commit.hash[:8],
                file_path=path,
                is_buggy=1 if is_bug else 0,
                **metrics,
                num_contributors=len(h.get('contributors', set())),
                num_commits=h.get('commits', 0),
                num_bug_fixes=h.get('bug_fixes', 0),
                file_age_commits=commit_num - h.get('first_seen', commit_num),
                lines_added=mod.added_lines,
                lines_deleted=mod.deleted_lines,
                commit_hour=commit.author_date.hour,
                author_commits=author_experience.get(commit.author.email, 0),
                # Quick-win social features
                files_in_commit=len(commit.modified_files),
                day_of_week=commit.author_date.weekday(),  # 0=Mon, 6=Sun
                is_first_contribution=1 if commit.hash in first_contributions else 0,
                commit_msg_length=len(commit.msg),
            )

            if is_bug and buggy < 150:
                results.append(asdict(record))
                buggy += 1
            elif not is_bug and clean < buggy * 2:
                results.append(asdict(record))
                clean += 1

    print(f"  Extracted: {buggy} buggy, {clean} clean samples")

    # Print detection method stats
    if issue_checker:
        stats = issue_checker.get_stats()
        print(f"  Detection methods: {dict(detection_stats)}")
        print(f"  GitHub API: {stats['api_calls']} calls, {stats['cache_hits']} cache hits, {stats['bug_issues']} bug issues found")

    return results


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train XGBoost model and evaluate"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    # All available features (new features are optional for backward compat)
    all_feature_cols = [
        # Code metrics
        'loc', 'sloc', 'avg_complexity', 'max_complexity', 'maintainability_index',
        # Historical features
        'num_contributors', 'num_commits', 'num_bug_fixes', 'file_age_commits',
        'lines_added', 'lines_deleted',
        # New features (v2)
        'import_count', 'max_nesting_depth', 'commit_hour', 'author_commits',
        # Quick-win social features
        'files_in_commit', 'day_of_week', 'is_first_contribution', 'commit_msg_length',
    ]

    # Use only features present in the dataset
    feature_cols = [c for c in all_feature_cols if c in df.columns]
    print(f"  Using {len(feature_cols)} features")

    X = df[feature_cols].fillna(0)
    y = df['is_buggy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  F1 Score: {results['f1']:.3f}")
    print(f"  ROC AUC:  {results['roc_auc']:.3f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop Predictive Features:")
    for _, row in importance.head(5).iterrows():
        bar = '#' * int(row['importance'] * 30)
        print(f"  {row['feature']:<22} {row['importance']:.3f} {bar}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"\n5-Fold CV ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    return {'model': model, 'results': results, 'importance': importance}


# =============================================================================
# FEATURE SELECTION & MODEL COMPARISON
# =============================================================================

def select_features(df: pd.DataFrame, k: int = 8) -> list[str]:
    """Select top k features using ANOVA F-test"""
    all_feature_cols = [
        'loc', 'sloc', 'avg_complexity', 'max_complexity', 'maintainability_index',
        'num_contributors', 'num_commits', 'num_bug_fixes', 'file_age_commits',
        'lines_added', 'lines_deleted',
        'import_count', 'max_nesting_depth', 'commit_hour', 'author_commits',
        'files_in_commit', 'day_of_week', 'is_first_contribution', 'commit_msg_length',
    ]
    feature_cols = [c for c in all_feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['is_buggy']

    selector = SelectKBest(f_classif, k=min(k, len(feature_cols)))
    selector.fit(X, y)

    selected = [f for f, s in zip(feature_cols, selector.get_support()) if s]
    return selected


def compare_models(df: pd.DataFrame, feature_cols: list[str] = None) -> pd.DataFrame:
    """Compare multiple models on the same data"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    all_feature_cols = [
        'loc', 'sloc', 'avg_complexity', 'max_complexity', 'maintainability_index',
        'num_contributors', 'num_commits', 'num_bug_fixes', 'file_age_commits',
        'lines_added', 'lines_deleted',
        'import_count', 'max_nesting_depth', 'commit_hour', 'author_commits',
        'files_in_commit', 'day_of_week', 'is_first_contribution', 'commit_msg_length',
    ]

    if feature_cols is None:
        feature_cols = [c for c in all_feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['is_buggy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0),
    }

    results = []

    for name, model in models.items():
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='roc_auc')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_prob),
            'CV_ROC_AUC': cv_scores.mean(),
            'CV_Std': cv_scores.std() * 2,
        })

    results_df = pd.DataFrame(results)

    print(f"\nUsing {len(feature_cols)} features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
    print(f"\n{'Model':<22} {'Acc':>8} {'F1':>8} {'ROC':>8} {'CV ROC':>8} {'±':>6}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} {row['Accuracy']:>8.3f} {row['F1']:>8.3f} {row['ROC_AUC']:>8.3f} {row['CV_ROC_AUC']:>8.3f} {row['CV_Std']:>6.3f}")

    return results_df


def full_analysis(df: pd.DataFrame):
    """Run complete analysis with feature selection and model comparison"""
    print("\n" + "="*60)
    print("FULL ANALYSIS")
    print("="*60)
    print(f"Dataset: {len(df)} samples ({df['is_buggy'].sum()} buggy, {len(df) - df['is_buggy'].sum()} clean)")

    # 1. All features comparison
    print("\n>>> COMPARISON 1: All available features")
    all_results = compare_models(df)

    # 2. Feature selection
    print("\n>>> FEATURE SELECTION")
    selected = select_features(df, k=8)
    print(f"Top 8 features by ANOVA F-test:")
    for i, f in enumerate(selected, 1):
        print(f"  {i}. {f}")

    # 3. Selected features comparison
    print("\n>>> COMPARISON 2: Selected features only")
    selected_results = compare_models(df, selected)

    # 4. Summary
    print("\n" + "="*60)
    print("SUMMARY: Best Model by CV ROC AUC")
    print("="*60)

    all_best = all_results.loc[all_results['CV_ROC_AUC'].idxmax()]
    sel_best = selected_results.loc[selected_results['CV_ROC_AUC'].idxmax()]

    print(f"  All features:      {all_best['Model']} ({all_best['CV_ROC_AUC']:.3f})")
    print(f"  Selected features: {sel_best['Model']} ({sel_best['CV_ROC_AUC']:.3f})")

    return {
        'all_features_results': all_results,
        'selected_features': selected,
        'selected_features_results': selected_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Defect Diviner - Bug Prediction')
    parser.add_argument('--repos', nargs='+', default=DEFAULT_REPOS,
                        help='GitHub repo URLs to analyze')
    parser.add_argument('--output', default='defect_dataset.csv',
                        help='Output CSV path')
    parser.add_argument('--diagnose', action='store_true',
                        help='Diagnose repo quality without full extraction')
    parser.add_argument('--no-github', action='store_true',
                        help='Disable GitHub API (use regex only for bug detection)')
    args = parser.parse_args()

    print("="*60)
    print("DEFECT DIVINER")
    print("="*60)

    use_github = not args.no_github
    if use_github:
        if GITHUB_TOKEN:
            print("Bug detection: GitHub issue labels (authenticated)")
        else:
            print("Bug detection: GitHub issue labels (unauthenticated - 60 req/hr limit)")
            print("  Tip: Set GITHUB_TOKEN env var for 5000 req/hr")
    else:
        print("Bug detection: Regex patterns only")

    # Diagnose mode: quick quality check
    if args.diagnose:
        print("\nMode: REPO DIAGNOSTIC")
        for repo_url in args.repos:
            diagnose_repo(repo_url)
        return

    print(f"\nRepos: {args.repos}")

    # Extract features from all repos
    all_results = []
    for repo_url in args.repos:
        results = extract_features(repo_url, use_github_api=use_github)
        all_results.extend(results)

    # Save dataset
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nDataset saved: {args.output} ({len(df)} samples)")

    # Train model
    if len(df) >= 50:
        train_and_evaluate(df)
    else:
        print("\nNot enough samples for training. Try adding more repos.")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
