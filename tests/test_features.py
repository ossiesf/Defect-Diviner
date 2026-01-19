#!/usr/bin/env python3
"""
Feature Evaluation Tests

Run this after changing features or regenerating the dataset to verify
which features help vs hurt model performance.

Usage:
    python -m pytest tests/test_features.py -v
    python tests/test_features.py  # standalone
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from defect_diviner.config import ALL_FEATURE_COLS


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = Path(__file__).parent.parent / "dataset_production.csv"
CV_FOLDS = 5
SIGNIFICANCE_THRESHOLD = 0.10  # p-value threshold for "significant" correlation
IMPROVEMENT_THRESHOLD = 0.005  # minimum CV improvement to consider meaningful


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_dataset():
    """Load the production dataset"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df['is_buggy']

    return df, X, y, feature_cols


def evaluate_correlations(X, y, feature_cols):
    """
    Evaluate correlation of each feature with target.

    Returns list of dicts with: feature, correlation, p_value, mutual_info
    """
    results = []
    mi_scores = mutual_info_classif(X, y, random_state=42)

    for i, col in enumerate(feature_cols):
        corr, p_val = stats.pointbiserialr(y, X[col])
        results.append({
            'feature': col,
            'correlation': corr,
            'p_value': p_val,
            'abs_corr': abs(corr),
            'mutual_info': mi_scores[i],
            'significant': p_val < SIGNIFICANCE_THRESHOLD,
        })

    return sorted(results, key=lambda x: x['abs_corr'], reverse=True)


def evaluate_ablation(X, y, feature_cols):
    """
    Ablation study: measure CV impact when each feature is removed.

    Returns baseline score and list of dicts with: feature, cv_score, delta, helps
    """
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )

    # Baseline with all features
    baseline_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='roc_auc')
    baseline = baseline_scores.mean()
    baseline_std = baseline_scores.std()

    results = []
    for col in feature_cols:
        remaining = [c for c in feature_cols if c != col]
        X_reduced = X[remaining]
        scores = cross_val_score(model, X_reduced, y, cv=CV_FOLDS, scoring='roc_auc')
        delta = scores.mean() - baseline

        results.append({
            'feature': col,
            'cv_score': scores.mean(),
            'cv_std': scores.std(),
            'delta': delta,
            'helps': delta < -IMPROVEMENT_THRESHOLD,  # removing hurts = feature helps
            'hurts': delta > IMPROVEMENT_THRESHOLD,   # removing helps = feature hurts
        })

    return baseline, baseline_std, sorted(results, key=lambda x: x['delta'])


def find_optimal_features(df, y, feature_cols):
    """
    Greedy forward selection to find optimal feature set.

    Returns list of optimal features and their CV score.
    """
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )

    # Start with best single feature (usually num_bug_fixes)
    best_single = None
    best_score = 0
    for f in feature_cols:
        score = cross_val_score(model, df[[f]].fillna(0), y, cv=CV_FOLDS, scoring='roc_auc').mean()
        if score > best_score:
            best_score = score
            best_single = f

    selected = [best_single]
    remaining = [f for f in feature_cols if f != best_single]
    baseline = best_score

    # Greedy addition
    improved = True
    while improved and remaining:
        improved = False
        best_gain = 0
        best_feature = None

        for f in remaining:
            test_features = selected + [f]
            score = cross_val_score(
                model, df[test_features].fillna(0), y, cv=CV_FOLDS, scoring='roc_auc'
            ).mean()
            gain = score - baseline
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_new_score = score

        if best_gain > IMPROVEMENT_THRESHOLD:
            selected.append(best_feature)
            remaining.remove(best_feature)
            baseline = best_new_score
            improved = True

    # Final score with selected features
    final_scores = cross_val_score(model, df[selected].fillna(0), y, cv=CV_FOLDS, scoring='roc_auc')

    return selected, final_scores.mean(), final_scores.std()


# =============================================================================
# PYTEST TESTS
# =============================================================================

def test_dataset_exists():
    """Verify production dataset exists"""
    assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"


def test_dataset_has_samples():
    """Verify dataset has enough samples"""
    df, X, y, _ = load_dataset()
    assert len(df) >= 100, f"Dataset too small: {len(df)} samples"
    assert y.sum() >= 20, f"Not enough buggy samples: {y.sum()}"


def test_num_bug_fixes_is_predictive():
    """The num_bug_fixes feature should always be predictive"""
    df, X, y, feature_cols = load_dataset()

    if 'num_bug_fixes' not in feature_cols:
        return  # Skip if feature not present

    corr, p_val = stats.pointbiserialr(y, X['num_bug_fixes'])
    assert p_val < 0.05, f"num_bug_fixes not significant: p={p_val:.4f}"
    assert corr > 0.1, f"num_bug_fixes correlation too weak: r={corr:.4f}"


def test_optimal_beats_all_features():
    """Optimal feature set should beat using all features"""
    df, X, y, feature_cols = load_dataset()

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )

    # All features
    all_score = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='roc_auc').mean()

    # Optimal features
    optimal, opt_score, _ = find_optimal_features(df, y, feature_cols)

    # Optimal should be at least as good (within noise margin)
    assert opt_score >= all_score - 0.02, \
        f"Optimal ({opt_score:.4f}) worse than all features ({all_score:.4f})"


def test_no_severe_overfitting():
    """CV variance should not be too high"""
    df, X, y, feature_cols = load_dataset()

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )

    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='roc_auc')

    # Variance should be reasonable
    assert scores.std() < 0.15, \
        f"CV variance too high: {scores.std():.4f} (possible overfitting)"


# =============================================================================
# STANDALONE REPORT
# =============================================================================

def print_full_report():
    """Print comprehensive feature evaluation report"""
    print("=" * 70)
    print("FEATURE EVALUATION REPORT")
    print("=" * 70)

    df, X, y, feature_cols = load_dataset()
    print(f"\nDataset: {len(df)} samples ({y.sum()} buggy, {len(df) - y.sum()} clean)")
    print(f"Features: {len(feature_cols)}")

    # Correlations
    print(f"\n{'='*70}")
    print("1. FEATURE CORRELATIONS WITH TARGET")
    print(f"{'='*70}")
    print(f"\n{'Feature':<24} {'Corr':>8} {'p-value':>10} {'MI':>8} {'Signal?':<10}")
    print("-" * 70)

    corr_results = evaluate_correlations(X, y, feature_cols)
    for r in corr_results:
        sig = "YES" if r['significant'] else "NOISE?"
        print(f"{r['feature']:<24} {r['correlation']:>8.3f} {r['p_value']:>10.4f} {r['mutual_info']:>8.3f} {sig:<10}")

    # Ablation
    print(f"\n{'='*70}")
    print("2. ABLATION STUDY (CV impact when feature removed)")
    print(f"{'='*70}")

    baseline, baseline_std, ablation_results = evaluate_ablation(X, y, feature_cols)
    print(f"\nBaseline (all features): {baseline:.4f} (+/- {baseline_std*2:.4f})")
    print(f"\n{'Feature Removed':<24} {'CV ROC':>10} {'Delta':>10} {'Impact':<12}")
    print("-" * 60)

    for r in ablation_results:
        if r['helps']:
            impact = "HELPS"
        elif r['hurts']:
            impact = "HURTS!"
        else:
            impact = "minimal"
        print(f"{r['feature']:<24} {r['cv_score']:>10.4f} {r['delta']:>+10.4f} {impact:<12}")

    # Optimal set
    print(f"\n{'='*70}")
    print("3. OPTIMAL FEATURE SET (greedy selection)")
    print(f"{'='*70}")

    optimal, opt_score, opt_std = find_optimal_features(df, y, feature_cols)
    print(f"\nOptimal features: {optimal}")
    print(f"CV ROC AUC: {opt_score:.4f} (+/- {opt_std*2:.4f})")

    # Summary
    print(f"\n{'='*70}")
    print("4. SUMMARY")
    print(f"{'='*70}")

    noise_features = [r['feature'] for r in corr_results if not r['significant']]
    hurting_features = [r['feature'] for r in ablation_results if r['hurts']]
    helping_features = [r['feature'] for r in ablation_results if r['helps']]

    print(f"\nFeatures that HELP (removing hurts CV): {helping_features}")
    print(f"Features that HURT (removing improves CV): {hurting_features}")
    print(f"Features with weak correlation (p >= {SIGNIFICANCE_THRESHOLD}): {noise_features}")
    print(f"\nRecommended feature set: {optimal}")
    print(f"Expected CV ROC AUC: {opt_score:.4f}")

    # Improvement
    improvement = opt_score - baseline
    if improvement > 0:
        print(f"\nUsing optimal set improves CV by +{improvement:.4f} over all features!")

    return {
        'correlations': corr_results,
        'ablation': ablation_results,
        'optimal_features': optimal,
        'optimal_score': opt_score,
        'baseline_score': baseline,
    }


if __name__ == "__main__":
    print_full_report()
