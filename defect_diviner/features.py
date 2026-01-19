"""
Feature definitions and code metrics extraction.
"""

import ast
from dataclasses import dataclass, asdict

from radon.complexity import cc_visit
from radon.raw import analyze
from radon.metrics import mi_visit


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

    # PR review features (high value - research shows code review is huge)
    review_count: int = 0           # Number of reviews on the PR
    approval_count: int = 0         # Number of approvals
    changes_requested: int = 0      # Number of "changes requested" reviews
    pr_hours_open: float = 0.0      # Hours PR was open before merge (rushed = risky)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


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
