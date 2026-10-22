"""Build data/features.csv from real repos (PyDriller) or synthetic data.

Examples (run from the project root):
    python -m extractor.dataset_builder --simulate --n-rows 600
    python -m extractor.dataset_builder --repos https://github.com/psf/requests \
        https://github.com/pallets/flask --max-commits 800 --ci-logs-dir data/raw/ci_logs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.ci_features import BaselineTracker, extract_test_duration, parse_ci_log  # noqa: E402
from extractor.text_features import TextRiskExtractor  # noqa: E402

META_COLS = ["release_id", "repo", "commit_hash", "timestamp"]
GIT_COLS = ["lines_added", "lines_deleted", "commit_churn", "files_modified", "num_directories",
            "directory_entropy", "test_file_ratio", "author_prior_commits", "author_risk_score",
            "avg_file_age_days", "hour_of_day", "is_weekend"]
CI_COLS = ["ci_warning_count", "ci_error_count", "ci_failed_test_count",
           "ci_test_duration_delta", "flaky_test_flag"]
TEXT_COLS = ["text_risk_score", "risk_keyword_count", "text_length", "text_semantic_risk"]
ALL_COLS = META_COLS + GIT_COLS + CI_COLS + TEXT_COLS + ["risk_label"]

# ------------------------------------------------------------------ simulation
RISKY_MSGS = [
    "Hotfix: patch null pointer crash in payment service",
    "Revert database schema migration after production outage",
    "Security fix for authentication vulnerability, urgent",
    "Fix regression in checkout flow introduced by breaking API change",
    "Emergency rollback of config change causing timeout",
    "Workaround for race condition in job scheduler",
    "Migrate user table schema, rewrite session handling",
    "Critical memory leak fix in cache layer",
]
SAFE_MSGS = [
    "Fix typo in README", "Update documentation for setup", "Add unit tests for parser",
    "Code formatting and lint cleanup", "Rename variable for readability",
    "Update comments in utils", "Minor style tweak in UI text",
]
NEUTRAL_MSGS = [
    "Add pagination to search results", "Implement user profile page", "Update dependency versions",
    "Improve logging output", "Add caching for product listing", "Support CSV export",
    "Handle empty input in form validator", "Optimize image loading",
]
WARN_LINES = ["WARNING: deprecated API call in module auth", "warning: unused variable 'x'",
              "WARNING: slow query detected (>500ms)", "warning: implicit conversion may lose data"]
ERR_LINES = ["ERROR: connection refused to test database", "Exception in thread worker-3",
             "FATAL: out of memory", "TIMEOUT waiting for service health check"]


def _zs(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _make_log(rng, warn, err, failed, flaky, duration, ntests):
    lines = ["[ci] checkout ok", "[ci] installing dependencies"]
    lines += [str(rng.choice(WARN_LINES)) for _ in range(warn)]
    lines += [str(rng.choice(ERR_LINES)) for _ in range(err)]
    if flaky:
        lines.append(f"[flaky] test_case_{rng.integers(1, 99)} passed on retry 2")
    lines.append(f"Ran {ntests} tests in {duration:.2f}s")
    lines.append(f"FAILED (failures={failed})" if failed else "OK")
    return "\n".join(lines)


def simulate(n: int, seed: int, pos_rate: float, extractor: TextRiskExtractor) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    added = rng.lognormal(3.8, 1.3, n).astype(int) + 1
    deleted = (added * rng.beta(2, 4, n)).astype(int)
    churn = added + deleted
    files = np.maximum(1, rng.poisson(1 + np.log1p(churn) / 1.2)).astype(int)
    ndirs = np.maximum(1, rng.binomial(files, 0.45))
    entropy = np.where(ndirs > 1, np.log2(ndirs) * rng.uniform(0.5, 1.0, n), 0.0)
    prior = rng.integers(1, 500, n)
    author_risk = rng.beta(2, 8, n)
    age = rng.exponential(300, n)
    test_ratio = rng.beta(1.5, 4, n)
    hour = rng.integers(0, 24, n)
    weekend = (rng.random(n) < 0.15).astype(int)

    z = (0.8 * _zs(np.log1p(churn)) + 0.6 * _zs(entropy) + 0.7 * _zs(author_risk)
         - 0.4 * _zs(test_ratio) - 0.3 * _zs(np.log1p(age)) + 0.3 * weekend
         + 0.3 * ((hour < 6) | (hour >= 22)) + rng.normal(0, 0.8, n))
    lo, hi = -10.0, 10.0
    for _ in range(60):  # bisection so mean probability == pos_rate
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < pos_rate else (lo, mid)
    y = (rng.random(n) < _sigmoid(z + (lo + hi) / 2)).astype(int)

    # CI + text signals depend on the label only partially -> overlap, no perfect separation
    warn = rng.poisson(np.where(y == 1, 7, 5))
    err = rng.poisson(np.where(y == 1, 0.9, 0.5))
    failed = rng.poisson(np.where(y == 1, 0.7, 0.3))
    flaky = rng.random(n) < np.where(y == 1, 0.25, 0.12)
    baseline = 120.0
    duration = baseline * (1 + rng.normal(0.10 * y, 0.25)).clip(0.3)
    ntests = rng.integers(80, 400, n)

    messages, ci_rows = [], []
    for i in range(n):
        p_risky = 0.45 if y[i] else 0.15
        p_safe = 0.10 if y[i] else 0.30
        r = rng.random()
        pool = RISKY_MSGS if r < p_risky else SAFE_MSGS if r < p_risky + p_safe else NEUTRAL_MSGS
        messages.append(str(rng.choice(pool)))
        log = _make_log(rng, warn[i], err[i], failed[i], flaky[i], duration[i], ntests[i])
        ci_rows.append(parse_ci_log(log, baseline))  # real parser on synthetic log text

    ts = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=n, freq="3h")
    repos = rng.choice(["sim-web", "sim-api", "sim-data"], n)
    df = pd.DataFrame({
        "release_id": [f"sim:{i:05d}" for i in range(n)], "repo": repos,
        "commit_hash": [f"{rng.integers(0, 16**10):010x}" for _ in range(n)],
        "timestamp": ts.astype(str),
        "lines_added": added, "lines_deleted": deleted, "commit_churn": churn,
        "files_modified": files, "num_directories": ndirs,
        "directory_entropy": entropy.round(4), "test_file_ratio": test_ratio.round(4),
        "author_prior_commits": prior, "author_risk_score": author_risk.round(4),
        "avg_file_age_days": age.round(2), "hour_of_day": hour, "is_weekend": weekend,
        "risk_label": y,
    })
    df = pd.concat([df, pd.DataFrame(ci_rows), pd.DataFrame(extractor.extract_many(messages))], axis=1)
    return df


# ------------------------------------------------------------------ real data
def _find_log(ci_dir: str, repo: str, h: str):
    for p in (Path(ci_dir) / repo / f"{h}.log", Path(ci_dir) / repo / f"{h[:7]}.log",
              Path(ci_dir) / repo / f"{h[:10]}.log"):
        if p.exists():
            return p
    return None


def build_real(repos, max_commits, ci_dir, clone_dir, extractor) -> pd.DataFrame:
    from extractor.git_features import mine_repo
    frames = []
    for repo in repos:
        print(f"[mining] {repo}")
        df = mine_repo(repo, clone_dir=clone_dir, max_commits=max_commits)
        if df.empty:
            print(f"  no usable commits in {repo}")
            continue
        tracker, ci_rows = BaselineTracker(), []
        for _, r in df.iterrows():
            p = _find_log(ci_dir, r["repo"], r["commit_hash"]) if ci_dir else None
            if p is None:
                ci_rows.append({c: np.nan for c in CI_COLS})  # CatBoost handles NaN natively
                continue
            text = p.read_text(errors="ignore")
            ci_rows.append(parse_ci_log(text, tracker.baseline()))
            tracker.update(extract_test_duration(text))
        txt = extractor.extract_many(df["message"].tolist())
        frames.append(pd.concat([df.drop(columns=["message"]), pd.DataFrame(ci_rows),
                                 pd.DataFrame(txt)], axis=1))
        print(f"  {len(df)} commits, {int(df['risk_label'].sum())} labelled risky")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Build data/features.csv")
    ap.add_argument("--simulate", action="store_true", help="generate synthetic data")
    ap.add_argument("--n-rows", type=int, default=600)
    ap.add_argument("--pos-rate", type=float, default=0.18, help="target share of risky releases (simulate)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repos", nargs="*", default=[], help="GitHub URLs or local repo paths")
    ap.add_argument("--max-commits", type=int, default=800)
    ap.add_argument("--ci-logs-dir", default="data/raw/ci_logs",
                    help="expects <dir>/<repo>/<commit_hash>.log (missing logs -> NaN CI features)")
    ap.add_argument("--clone-dir", default="data/raw/repos")
    ap.add_argument("--text-backend", choices=["tfidf", "minilm"], default="tfidf")
    ap.add_argument("--out", default="data/features.csv")
    a = ap.parse_args()

    extractor = TextRiskExtractor(a.text_backend)
    if a.simulate:
        df = simulate(a.n_rows, a.seed, a.pos_rate, extractor)
    elif a.repos:
        df = build_real(a.repos, a.max_commits, a.ci_logs_dir, a.clone_dir, extractor)
    else:
        ap.error("give --simulate or --repos ...")
    if df.empty:
        sys.exit("No data produced.")

    df = df[ALL_COLS]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"risk_label balance: {df['risk_label'].mean():.1%} positive "
          f"({int(df['risk_label'].sum())} / {len(df)})")
    if len(df) < 200:
        print("WARNING: fewer than 200 rows - project requires at least 200.")
    if df[CI_COLS].isna().all().any():
        print("NOTE: some CI columns are entirely NaN (no CI logs found). "
              "Supply --ci-logs-dir, use a dataset with CI data (TravisTorrent), or use --simulate.")


if __name__ == "__main__":
    main()
