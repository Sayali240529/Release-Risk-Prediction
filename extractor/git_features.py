"""Git feature extraction + SZZ-style risk labelling (PyDriller).

mine_repo() walks a repo chronologically and returns one row per commit with
git features, the commit message, and a risk_label:
    1 = commit was later reverted OR identified as bug-inducing by SZZ
        (the lines it touched were later modified by a bug-fix commit)
    0 = otherwise
Commits in the last `tail_days` of history are dropped because they have not had
time to be reverted / fixed yet (right-censoring would bias labels towards 0).

author_risk_score uses only PAST commits (fraction of the author's earlier commits
whose message looks like a fix, Laplace-smoothed) so it does not leak the label.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import PurePosixPath

import pandas as pd

FIX_RE = re.compile(r"\b(fix(es|ed)?|bugs?|hotfix|defects?|crash(es)?|regression)\b", re.I)
REVERT_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.I)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|spec|__tests__)(/|$)|(_test\b|\btest_|\.test\.|\.spec\.)", re.I)
MAX_FILES = 200          # skip huge bulk commits (vendor drops, mass renames)
GLOBAL_PRIOR, K = 0.2, 5  # smoothing for author_risk_score


def repo_name(repo: str) -> str:
    name = repo.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def ensure_local_clone(repo: str, clone_dir: str) -> str:
    if os.path.isdir(repo):
        return repo
    dest = os.path.join(clone_dir, repo_name(repo))
    if not os.path.isdir(dest):
        import git  # GitPython
        os.makedirs(clone_dir, exist_ok=True)
        git.Repo.clone_from(repo, dest)
    return dest


def directory_entropy(file_lines: list[tuple[str, int]]) -> tuple[float, int]:
    """Shannon entropy (bits) of changed lines across directories; also #directories."""
    per_dir: dict[str, int] = defaultdict(int)
    for path, lines in file_lines:
        per_dir[str(PurePosixPath(path).parent)] += max(lines, 1)
    total = sum(per_dir.values())
    ent = -sum((v / total) * math.log2(v / total) for v in per_dir.values())
    return ent, len(per_dir)


def mine_repo(repo: str, clone_dir: str = "data/raw/repos", max_commits: int = 1000,
              szz_max_fixes: int = 300, tail_days: int = 30) -> pd.DataFrame:
    from pydriller import Repository
    try:
        from pydriller import Git
    except ImportError:  # pydriller < 2.0
        from pydriller import GitRepository as Git

    path = ensure_local_clone(repo, clone_dir)
    name = repo_name(repo)

    author_total, author_fix = defaultdict(int), defaultdict(int)
    first_seen: dict[str, object] = {}
    rows, fix_hashes, reverted_prefixes = [], [], set()

    for c in Repository(path, only_no_merge=True).traverse_commits():
        if max_commits and len(rows) >= max_commits:
            break
        try:
            mods = c.modified_files
        except Exception:
            continue
        if not mods or len(mods) > MAX_FILES:
            continue

        date = c.committer_date
        file_lines, ages, test_files = [], [], 0
        for m in mods:
            p = m.new_path or m.old_path
            if not p:
                continue
            file_lines.append((p, m.added_lines + m.deleted_lines))
            if p in first_seen:
                ages.append((date - first_seen[p]).days)
            else:
                first_seen[p] = date
                ages.append(0)
            test_files += bool(TEST_PATH_RE.search(p))
        if not file_lines:
            continue

        msg = c.msg or ""
        author = (c.author.email or c.author.name or "unknown").lower()
        author_risk = (author_fix[author] + GLOBAL_PRIOR * K) / (author_total[author] + K)
        ent, ndirs = directory_entropy(file_lines)
        added, deleted = c.insertions, c.deletions

        rows.append({
            "release_id": f"{name}:{c.hash[:10]}", "repo": name, "commit_hash": c.hash,
            "timestamp": date.isoformat(), "message": msg,
            "lines_added": added, "lines_deleted": deleted, "commit_churn": added + deleted,
            "files_modified": len(file_lines), "num_directories": ndirs,
            "directory_entropy": round(ent, 4),
            "test_file_ratio": round(test_files / len(file_lines), 4),
            "author_prior_commits": author_total[author],
            "author_risk_score": round(author_risk, 4),
            "avg_file_age_days": round(sum(ages) / len(ages), 2),
            "hour_of_day": date.hour, "is_weekend": int(date.weekday() >= 5),
        })

        # update running history AFTER computing this commit's features
        author_total[author] += 1
        is_fix = bool(FIX_RE.search(msg))
        author_fix[author] += is_fix
        if is_fix:
            fix_hashes.append(c.hash)
        for m in REVERT_RE.finditer(msg):
            reverted_prefixes.add(m.group(1).lower())

    if not rows:
        return pd.DataFrame()

    # ---- labelling: reverts + SZZ ----
    inducing: set[str] = set()
    git = Git(path)
    for h in fix_hashes[:szz_max_fixes]:
        try:
            for hs in git.get_commits_last_modified_lines(git.get_commit(h)).values():
                inducing |= set(hs)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    is_reverted = df["commit_hash"].apply(
        lambda h: any(h.startswith(p) or p.startswith(h[:len(p)]) for p in reverted_prefixes))
    df["risk_label"] = (df["commit_hash"].isin(inducing) | is_reverted).astype(int)

    # drop right-censored tail
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df[ts <= ts.max() - timedelta(days=tail_days)].reset_index(drop=True)
    return df
