"""
Repository mining and feature extraction pipeline.
"""

from collections import defaultdict

from pydriller import Repository

from .config import MAX_COMMITS, BUG_KEYWORDS, EXCLUDE_KEYWORDS
from .github import GitHubIssueChecker, GitHubPRChecker, parse_repo_url
from .features import FileFeatures, extract_code_metrics


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

    # Initialize GitHub API checkers for better bug detection and PR review data
    issue_checker = None
    pr_checker = None
    if use_github_api:
        try:
            owner, repo = parse_repo_url(repo_url)
            issue_checker = GitHubIssueChecker(owner, repo)
            # Share session with PR checker to avoid redundant auth
            pr_checker = GitHubPRChecker(owner, repo, session=issue_checker.session)
            print(f"  Using GitHub API for bug detection + PR reviews", flush=True)
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

        # Fetch PR review data once per commit (cached)
        pr_data = {'review_count': 0, 'approval_count': 0, 'changes_requested': 0, 'pr_hours_open': 0.0}
        if pr_checker:
            pr_data = pr_checker.get_pr_review_data(commit.hash)

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
                # PR review features
                review_count=pr_data['review_count'],
                approval_count=pr_data['approval_count'],
                changes_requested=pr_data['changes_requested'],
                pr_hours_open=pr_data['pr_hours_open'],
            )

            if is_bug and buggy < 150:
                results.append(record.to_dict())
                buggy += 1
            elif not is_bug and clean < buggy * 2:
                results.append(record.to_dict())
                clean += 1

    print(f"  Extracted: {buggy} buggy, {clean} clean samples")

    # Print detection method stats
    if issue_checker:
        stats = issue_checker.get_stats()
        print(f"  Detection methods: {dict(detection_stats)}")
        print(f"  Issue API: {stats['api_calls']} calls, {stats['cache_hits']} cache hits, {stats['bug_issues']} bug issues found")

    # Print PR review stats
    if pr_checker:
        pr_stats = pr_checker.get_stats()
        print(f"  PR API: {pr_stats['api_calls']} calls, {pr_stats['cache_hits']} cache hits, {pr_stats['prs_found']}/{pr_stats['cached_prs']} commits from PRs")

    return results
