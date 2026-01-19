#!/usr/bin/env python3
"""
Defect Diviner - Bug Prediction from Code History
==================================================

CLI entry point for the defect prediction pipeline.

Usage:
    python defect_predictor.py --repos https://github.com/user/repo
    python defect_predictor.py --diagnose --repos https://github.com/user/repo
    python defect_predictor.py --no-github  # regex only, no API

Author: Ossie - Senior SDET exploring ML for software quality
"""

import argparse

import pandas as pd

from defect_diviner import (
    DEFAULT_REPOS,
    GITHUB_TOKEN,
    extract_features,
    train_and_evaluate,
    diagnose_repo,
)


def main():
    parser = argparse.ArgumentParser(
        description='Defect Diviner - Bug Prediction from Code History',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python defect_predictor.py --repos https://github.com/pallets/click
  python defect_predictor.py --diagnose --repos https://github.com/user/repo
  python defect_predictor.py --no-github  # Use regex only, no GitHub API

Environment:
  GITHUB_TOKEN    Set for 5000 req/hr API limit (vs 60 unauthenticated)
        """
    )
    parser.add_argument(
        '--repos', nargs='+', default=DEFAULT_REPOS,
        help='GitHub repo URLs to analyze'
    )
    parser.add_argument(
        '--output', default='defect_dataset.csv',
        help='Output CSV path (default: defect_dataset.csv)'
    )
    parser.add_argument(
        '--diagnose', action='store_true',
        help='Diagnose repo quality without full extraction'
    )
    parser.add_argument(
        '--no-github', action='store_true',
        help='Disable GitHub API (use regex only for bug detection)'
    )
    parser.add_argument(
        '--optimal', action='store_true',
        help='Use optimal 3-feature set instead of all features'
    )
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
        train_and_evaluate(df, use_optimal=args.optimal)
    else:
        print("\nNot enough samples for training. Try adding more repos.")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
