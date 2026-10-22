"""Temporary synthetic data so you can build/test before Member 1 delivers features.csv.
Writes data/features_dummy.csv. Run: python model/make_dummy_data.py
Then: python model/train.py --data data/features_dummy.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
df = pd.DataFrame({
    "commit_churn": rng.lognormal(5, 1, n).astype(int),
    "files_modified": rng.poisson(6, n) + 1,
    "directory_entropy": rng.beta(2, 5, n) * 3,
    "author_risk_score": rng.beta(2, 8, n),
    "ci_warning_count": rng.poisson(5, n),
    "ci_test_duration_delta": rng.normal(0, 0.2, n),
    "flaky_test_flag": rng.binomial(1, 0.15, n),
    "text_risk_score": rng.beta(2, 6, n),
    "commit_message_risk_keywords": rng.poisson(0.5, n),
})
z = (0.0008 * df.commit_churn + 0.9 * df.directory_entropy + 2.0 * df.author_risk_score
     + 0.12 * df.ci_warning_count + 1.0 * df.flaky_test_flag + 2.0 * df.text_risk_score
     + 0.4 * df.commit_message_risk_keywords + rng.normal(0, 0.7, n))
df["risk_label"] = (z > np.quantile(z, 0.82)).astype(int)  # ~18% risky -> imbalanced
out = Path(__file__).resolve().parent.parent / "data" / "features_dummy.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out}  (positive rate {df.risk_label.mean():.2f})")
