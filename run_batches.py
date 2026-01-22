#!/usr/bin/env python3
"""
Batch runner for mining repos in manageable chunks.

Usage:
    python run_batches.py --batch 1      # Run batch 1 (repos 1-25)
    python run_batches.py --batch 2      # Run batch 2 (repos 26-50)
    python run_batches.py --batch all    # Run all batches with pauses
    python run_batches.py --merge        # Merge all batch outputs
    python run_batches.py --status       # Check progress
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Configuration
REPOS_FILE = Path(__file__).parent / "repos_to_mine.txt"
OUTPUT_DIR = Path(__file__).parent / "batch_outputs"
BATCH_SIZE = 25
PAUSE_BETWEEN_BATCHES = 300  # 5 minutes (API rate limit buffer)


def load_repos():
    """Load repo URLs from file"""
    repos = []
    with open(REPOS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("https://"):
                repos.append(line)
    return repos


def get_batch(repos, batch_num):
    """Get repos for a specific batch (1-indexed)"""
    start = (batch_num - 1) * BATCH_SIZE
    end = start + BATCH_SIZE
    return repos[start:end]


def run_batch(batch_num, repos):
    """Run extraction for a batch of repos"""
    from defect_diviner import extract_features

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"batch_{batch_num}.csv"

    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}: {len(repos)} repos")
    print(f"{'='*60}")
    print(f"Repos: {repos[0].split('/')[-1]} ... {repos[-1].split('/')[-1]}")
    print(f"Output: {output_file}")

    all_results = []
    for i, repo_url in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] ", end="")
        try:
            results = extract_features(repo_url, use_github_api=True)
            all_results.extend(results)

            # Save incrementally (in case of crash)
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"BATCH {batch_num} COMPLETE: {len(all_results)} samples")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

    return len(all_results)


def merge_batches():
    """Merge all batch outputs into final dataset"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    dfs = []
    for batch_file in sorted(OUTPUT_DIR.glob("batch_*.csv")):
        print(f"Loading {batch_file.name}...")
        df = pd.read_csv(batch_file)
        dfs.append(df)
        print(f"  {len(df)} samples")

    if not dfs:
        print("No batch files found!")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (same repo + commit + file)
    before = len(merged)
    merged = merged.drop_duplicates(subset=['repo', 'commit_hash', 'file_path'])
    after = len(merged)

    output_file = Path(__file__).parent / "dataset_large.csv"
    merged.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"MERGED: {after} samples ({before - after} duplicates removed)")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

    # Quick stats
    print(f"\nDataset stats:")
    print(f"  Repos: {merged['repo'].nunique()}")
    print(f"  Buggy: {merged['is_buggy'].sum()} ({merged['is_buggy'].mean()*100:.1f}%)")
    print(f"  Clean: {len(merged) - merged['is_buggy'].sum()}")


def show_status():
    """Show current progress"""
    repos = load_repos()
    total_batches = (len(repos) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n{'='*60}")
    print("BATCH STATUS")
    print(f"{'='*60}")
    print(f"Total repos: {len(repos)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")
    print()

    total_samples = 0
    for i in range(1, total_batches + 1):
        batch_file = OUTPUT_DIR / f"batch_{i}.csv"
        batch_repos = get_batch(repos, i)

        if batch_file.exists():
            df = pd.read_csv(batch_file)
            samples = len(df)
            total_samples += samples
            status = f"DONE ({samples} samples)"
        else:
            status = "PENDING"

        print(f"  Batch {i}: repos {(i-1)*BATCH_SIZE + 1}-{min(i*BATCH_SIZE, len(repos))} - {status}")

    print(f"\nTotal samples collected: {total_samples}")

    # Check API rate limit
    try:
        import requests
        token = os.environ.get('GITHUB_TOKEN', '')
        headers = {'Authorization': f'token {token}'} if token else {}
        resp = requests.get('https://api.github.com/rate_limit', headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            remaining = data['resources']['core']['remaining']
            limit = data['resources']['core']['limit']
            reset_time = data['resources']['core']['reset']
            reset_in = max(0, reset_time - time.time()) / 60
            print(f"\nGitHub API: {remaining}/{limit} remaining (resets in {reset_in:.0f} min)")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Batch runner for repo mining')
    parser.add_argument('--batch', type=str, help='Batch number (1-4) or "all"')
    parser.add_argument('--merge', action='store_true', help='Merge all batch outputs')
    parser.add_argument('--status', action='store_true', help='Show progress status')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.merge:
        merge_batches()
        return

    if not args.batch:
        parser.print_help()
        print("\nExamples:")
        print("  python run_batches.py --status        # Check progress")
        print("  python run_batches.py --batch 1       # Run batch 1")
        print("  python run_batches.py --batch all     # Run all batches")
        print("  python run_batches.py --merge         # Merge outputs")
        return

    repos = load_repos()
    total_batches = (len(repos) + BATCH_SIZE - 1) // BATCH_SIZE

    if args.batch == 'all':
        # Run all batches with pauses
        for batch_num in range(1, total_batches + 1):
            batch_repos = get_batch(repos, batch_num)
            if not batch_repos:
                break

            run_batch(batch_num, batch_repos)

            if batch_num < total_batches:
                print(f"\nPausing {PAUSE_BETWEEN_BATCHES}s before next batch...")
                print("(Ctrl+C to stop, progress is saved)")
                time.sleep(PAUSE_BETWEEN_BATCHES)

        # Merge at the end
        print("\nMerging all batches...")
        merge_batches()
    else:
        # Run specific batch
        batch_num = int(args.batch)
        if batch_num < 1 or batch_num > total_batches:
            print(f"Invalid batch number. Must be 1-{total_batches}")
            return

        batch_repos = get_batch(repos, batch_num)
        run_batch(batch_num, batch_repos)


if __name__ == "__main__":
    main()
