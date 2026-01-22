# Defect Diviner - TODO

## Critical Finding: Feature Evaluation (Jan 2025)

**Most features add noise, not signal.** Rigorous evaluation showed:

| Feature Set | CV ROC AUC | Variance |
|-------------|------------|----------|
| All 19 features | 0.658 | +/- 0.246 |
| **Optimal 3 features** | **0.746** | **+/- 0.166** |

The optimal set is just:
1. `num_bug_fixes` - files with past bugs get more bugs (67% importance)
2. `file_age_commits` - older files are more stable (21%)
3. `lines_deleted` - code churn indicator (12%)

**Run `python tests/test_features.py` to re-evaluate after changes.**

---

## On Hold (Likely Noise)

These features were planned but evaluation suggests they'd add noise, not signal:

### Is Collaborator/Maintainer
- Would need 5-10x more data to show value
- Risk: spurious correlation with repo-specific patterns

### Author GitHub Stats
- Account age/followers are weak proxies for skill
- High risk of overfitting

**Recommendation:** Don't add more features. Focus on:
1. More training data (more repos)
2. Better labels (SZZ algorithm)

---

## Implemented Features

### PR Review Count + Time Open - DONE
- [x] `review_count` - Number of reviews on the PR
- [x] `approval_count` - Number of approvals
- [x] `changes_requested` - Number of "changes requested" reviews
- [x] `pr_hours_open` - Hours PR was open before merge
- Uses `GitHubPRChecker` class with caching
- **Status:** Implemented but evaluation shows these don't improve CV score

### Quick Wins (No API calls) - DONE
- [x] `files_in_commit` - Number of files changed
- [x] `day_of_week` - Day of week
- [x] `is_first_contribution` - Author's first commit
- [x] `commit_msg_length` - Message length
- **Status:** Implemented but evaluation shows most are noise

### GitHub Issue Labels - DONE
- [x] `GitHubIssueChecker` class for fetching issue labels
- [x] Parse issue references (#123, fixes #456)
- [x] Fallback to regex when no issue reference

### Code Refactoring - DONE
- [x] Split monolith into `defect_diviner/` package
- [x] Add test suite (`tests/`)
- [x] Add `--optimal` CLI flag for 3-feature model

---

## Future Ideas (If More Data Available)

- **SZZ Algorithm:** Better labels by tracing bug-introducing commits
- **More repos:** 5-10x data might let weaker features show value
- **Web UI:** "Scan your repo" feature (product, not model improvement)
