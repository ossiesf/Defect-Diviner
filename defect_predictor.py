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
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from pydriller import Repository
from radon.complexity import cc_visit
from radon.raw import analyze
from radon.metrics import mi_visit

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
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
EXCLUDE_KEYWORDS = re.compile(r'\b(typo|doc|style|format|merge|revert)\b', re.IGNORECASE)


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
        }
    except Exception:
        return None


def build_file_history(repo_url: str) -> dict:
    """First pass: build historical features for each file"""
    print(f"  Building file history...", flush=True)

    history = defaultdict(lambda: {
        'contributors': set(),
        'commits': 0,
        'bug_fixes': 0,
        'first_seen': 0,
    })

    commit_num = 0
    for commit in Repository(repo_url, only_no_merge=True).traverse_commits():
        commit_num += 1
        if commit_num > MAX_COMMITS:
            break

        is_bug = bool(BUG_KEYWORDS.search(commit.msg)) and not bool(EXCLUDE_KEYWORDS.search(commit.msg))

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

    return history


def extract_features(repo_url: str) -> list[dict]:
    """Extract all features from a repository"""
    repo_name = repo_url.split('/')[-1]
    print(f"\nProcessing: {repo_name}", flush=True)

    history = build_file_history(repo_url)
    results = []
    buggy, clean = 0, 0
    commit_num = 0

    print(f"  Extracting features...", flush=True)

    for commit in Repository(repo_url, only_no_merge=True).traverse_commits():
        commit_num += 1
        if commit_num > MAX_COMMITS:
            break

        is_bug = bool(BUG_KEYWORDS.search(commit.msg)) and not bool(EXCLUDE_KEYWORDS.search(commit.msg))

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
            )

            if is_bug and buggy < 150:
                results.append(asdict(record))
                buggy += 1
            elif not is_bug and clean < buggy * 2:
                results.append(asdict(record))
                clean += 1

    print(f"  Extracted: {buggy} buggy, {clean} clean samples")
    return results


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train XGBoost model and evaluate"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    feature_cols = ['loc', 'sloc', 'avg_complexity', 'max_complexity',
                    'maintainability_index', 'num_contributors', 'num_commits',
                    'num_bug_fixes', 'file_age_commits', 'lines_added', 'lines_deleted']

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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Defect Diviner - Bug Prediction')
    parser.add_argument('--repos', nargs='+', default=DEFAULT_REPOS,
                        help='GitHub repo URLs to analyze')
    parser.add_argument('--output', default='defect_dataset.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print("="*60)
    print("DEFECT DIVINER")
    print("="*60)
    print(f"Repos: {args.repos}")

    # Extract features from all repos
    all_results = []
    for repo_url in args.repos:
        results = extract_features(repo_url)
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
