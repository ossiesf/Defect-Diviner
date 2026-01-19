"""
GitHub API integration for issue labels and PR review data.
"""

import time
from datetime import datetime

import requests

from .config import (
    GITHUB_TOKEN,
    GITHUB_API_BASE,
    BUG_LABELS,
    BUG_KEYWORDS,
    EXCLUDE_KEYWORDS,
    ISSUE_REFERENCE,
)


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from GitHub URL"""
    # Handle: https://github.com/owner/repo or github.com/owner/repo
    parts = url.rstrip('/').split('/')
    return parts[-2], parts[-1]


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


class GitHubPRChecker:
    """Fetch and cache PR review data for commits"""

    def __init__(self, owner: str, repo: str, session: requests.Session = None):
        self.owner = owner
        self.repo = repo
        self.pr_cache = {}  # commit_sha -> PR data (or None if no PR)
        self.review_cache = {}  # pr_number -> review data
        self.api_calls = 0
        self.cache_hits = 0

        # Reuse session from GitHubIssueChecker if provided, otherwise create new
        if session:
            self.session = session
        else:
            self.session = requests.Session()
            if GITHUB_TOKEN:
                self.session.headers['Authorization'] = f'token {GITHUB_TOKEN}'
            self.session.headers['Accept'] = 'application/vnd.github.v3+json'
            self.session.headers['User-Agent'] = 'Defect-Diviner'

    def _fetch_pr_for_commit(self, commit_sha: str) -> dict | None:
        """Find the PR that a commit came from (if any)"""
        if commit_sha in self.pr_cache:
            self.cache_hits += 1
            return self.pr_cache[commit_sha]

        try:
            # GitHub API: Get PRs associated with a commit
            url = f'{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/commits/{commit_sha}/pulls'
            resp = self.session.get(url, timeout=10)
            self.api_calls += 1

            if resp.status_code == 200:
                prs = resp.json()
                if prs:
                    # Take the first (most recent) PR
                    pr = prs[0]
                    result = {
                        'number': pr['number'],
                        'created_at': pr.get('created_at'),
                        'merged_at': pr.get('merged_at'),
                    }
                    self.pr_cache[commit_sha] = result
                    return result
                else:
                    # Commit not from a PR (direct push)
                    self.pr_cache[commit_sha] = None
                    return None
            elif resp.status_code == 403:
                # Rate limited
                return None
            elif resp.status_code == 422:
                # API not available for this repo/commit
                self.pr_cache[commit_sha] = None
                return None

            # Rate limit: small delay between calls
            if self.api_calls % 10 == 0:
                time.sleep(0.5)

        except Exception:
            pass

        return None

    def _fetch_reviews(self, pr_number: int) -> dict:
        """Fetch review data for a PR"""
        if pr_number in self.review_cache:
            self.cache_hits += 1
            return self.review_cache[pr_number]

        try:
            url = f'{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews'
            resp = self.session.get(url, timeout=10)
            self.api_calls += 1

            if resp.status_code == 200:
                reviews = resp.json()
                result = {
                    'review_count': len(reviews),
                    'approval_count': sum(1 for r in reviews if r.get('state') == 'APPROVED'),
                    'changes_requested': sum(1 for r in reviews if r.get('state') == 'CHANGES_REQUESTED'),
                }
                self.review_cache[pr_number] = result
                return result
            elif resp.status_code == 403:
                # Rate limited
                return {'review_count': 0, 'approval_count': 0, 'changes_requested': 0}

            # Rate limit: small delay
            if self.api_calls % 10 == 0:
                time.sleep(0.5)

        except Exception:
            pass

        return {'review_count': 0, 'approval_count': 0, 'changes_requested': 0}

    def get_pr_review_data(self, commit_sha: str) -> dict:
        """
        Get PR review data for a commit.

        Returns dict with: review_count, approval_count, changes_requested, pr_hours_open
        """
        pr = self._fetch_pr_for_commit(commit_sha)

        if not pr:
            # No PR found for this commit
            return {
                'review_count': 0,
                'approval_count': 0,
                'changes_requested': 0,
                'pr_hours_open': 0.0,
            }

        # Get reviews
        reviews = self._fetch_reviews(pr['number'])

        # Calculate PR time open
        pr_hours_open = 0.0
        if pr.get('created_at') and pr.get('merged_at'):
            try:
                created = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                merged = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
                pr_hours_open = (merged - created).total_seconds() / 3600
            except Exception:
                pass

        return {
            'review_count': reviews['review_count'],
            'approval_count': reviews['approval_count'],
            'changes_requested': reviews['changes_requested'],
            'pr_hours_open': round(pr_hours_open, 2),
        }

    def get_stats(self) -> dict:
        """Return API usage statistics"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cached_prs': len(self.pr_cache),
            'prs_found': sum(1 for v in self.pr_cache.values() if v is not None),
            'cached_reviews': len(self.review_cache),
        }
