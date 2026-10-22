# Member 1 – Data Pipeline & Feature Engineering

Produces `data/features.csv` (the CP-1 handoff to Member 2).

## Run

```bash
pip install -r requirements.txt

# Synthetic fallback (600 rows, ~18% risky) – works offline
python -m extractor.dataset_builder --simulate --n-rows 600

# Real repos (PyDriller + SZZ labelling); CI logs optional
python -m extractor.dataset_builder \
    --repos https://github.com/psf/requests https://github.com/pallets/flask \
    --max-commits 800 --ci-logs-dir data/raw/ci_logs

# Optional: MiniLM embeddings instead of TF-IDF for text_semantic_risk
python -m extractor.dataset_builder --simulate --text-backend minilm
```

CI logs are read from `data/raw/ci_logs/<repo_name>/<commit_hash>.log` (full, 7- or 10-char hash).
Commits without a log get `NaN` CI features (CatBoost handles NaN natively).

## Columns

**Identifiers – Member 2 must DROP these before training:** `release_id`, `repo`, `commit_hash`, `timestamp`
(`timestamp` is useful for a time-based train/test split).

| Group | Column | Meaning |
|---|---|---|
| Git | `lines_added`, `lines_deleted` | Lines added / removed |
| Git | `commit_churn` | added + deleted |
| Git | `files_modified` | Number of files touched |
| Git | `num_directories` | Distinct directories touched |
| Git | `directory_entropy` | Shannon entropy (bits) of changed lines across directories; high = scattered change |
| Git | `test_file_ratio` | Share of modified files that are test files |
| Git | `author_prior_commits` | Author's number of earlier commits in the mined history |
| Git | `author_risk_score` | Smoothed share of the author's *earlier* commits that were bug-fix commits (past-only, no label leakage) |
| Git | `avg_file_age_days` | Mean days since each modified file first appeared in the mined history |
| Git | `hour_of_day`, `is_weekend` | Commit time features |
| CI | `ci_warning_count` | Log lines containing "warning" |
| CI | `ci_error_count` | Hits of ERROR / EXCEPTION / FATAL / TIMEOUT |
| CI | `ci_failed_test_count` | Failed tests reported in the log |
| CI | `ci_test_duration_delta` | (test time − rolling median of previous builds) / baseline |
| CI | `flaky_test_flag` | 1 if log mentions flaky / retry / rerun / intermittent |
| Text | `text_risk_score` | 0–1 weighted risk-keyword score (hotfix, revert, migration, security, regression …) |
| Text | `risk_keyword_count` | Number of risk-keyword hits |
| Text | `text_length` | Characters in message/PR text |
| Text | `text_semantic_risk` | 0–1 similarity to risky-vs-safe anchor sentences (TF-IDF or MiniLM) |
| Target | `risk_label` | 1 = risky release, 0 = clean |

21 feature columns + `risk_label`.

## How `risk_label` is defined

- **Real mode:** 1 if the commit was later **reverted** (`This reverts commit <sha>`) or is **bug-inducing
  by SZZ** (its lines were later modified by a bug-fix commit). Commits in the last 30 days of history are
  dropped (not enough time for failures to show up). This is a *proxy* for "caused a production incident";
  state that limitation in the report.
- **Simulated mode:** label is drawn from a logistic model over git features; CI logs and messages are then
  generated with label-dependent but overlapping distributions and passed through the **real** CI/text
  extractors. Synthetic data is for pipeline testing/demo only – don't report its metrics as real results.

## Interfaces (don't change after CP-1 without telling Member 2)

- Column names above.
- `extractor.ci_features.parse_ci_log(text, baseline_duration=None)` and
  `extractor.text_features.extract_text_features(text)` return dicts keyed by the feature names –
  Member 3 can reuse them to turn raw log / PR text from the dashboard form into model inputs.
