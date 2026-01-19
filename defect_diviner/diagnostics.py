"""
Repository diagnostics for assessing training data quality.
"""

from pathlib import Path

import numpy as np
from pydriller import Repository

from .config import BUG_KEYWORDS, EXCLUDE_KEYWORDS


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
        print("  GOOD - Suitable for training")
    elif quality_score >= 50:
        print("  ~ FAIR - Usable with caveats")
    else:
        print("  POOR - Consider alternatives")

    if issues:
        print(f"\nIssues:")
        for issue in issues:
            print(f"  - {issue}")

    return {
        'quality_score': quality_score,
        'bug_ratio': bug_ratio,
        'python_files': py_file_count,
        'language_count': lang_count,
        'contributors': contributor_count,
        'issues': issues,
    }
