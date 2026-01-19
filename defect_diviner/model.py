"""
Model training, evaluation, and comparison.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import ALL_FEATURE_COLS, OPTIMAL_FEATURE_COLS


def train_and_evaluate(df: pd.DataFrame, use_optimal: bool = False) -> dict:
    """
    Train XGBoost model and evaluate.

    Args:
        df: DataFrame with features and 'is_buggy' target
        use_optimal: If True, use OPTIMAL_FEATURE_COLS (3 features).
                     If False, use ALL_FEATURE_COLS (all available).
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    # Select feature set
    if use_optimal:
        base_cols = OPTIMAL_FEATURE_COLS
        print(f"  Using OPTIMAL feature set")
    else:
        base_cols = ALL_FEATURE_COLS

    # Use only features present in the dataset
    feature_cols = [c for c in base_cols if c in df.columns]
    print(f"  Using {len(feature_cols)} features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

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


def select_features(df: pd.DataFrame, k: int = 8) -> list[str]:
    """Select top k features using ANOVA F-test"""
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

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

    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

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
