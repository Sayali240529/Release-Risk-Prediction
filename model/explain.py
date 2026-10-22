"""SHAP explainability: builds/saves the TreeExplainer and a global importance chart.

Usage: python model/explain.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ART = Path(__file__).resolve().parent / "artifacts"
REPORT = ART / "report"

# Plain-English labels for the top-factor display. Add any new feature from Member 1 here.
FEATURE_LABELS = {
    "commit_churn": "Large code change (lines added + deleted)",
    "lines_added": "Many lines added",
    "lines_deleted": "Many lines deleted",
    "files_modified": "Many files modified",
    "directory_entropy": "Changes scattered across many modules",
    "author_risk_score": "Author has a history of buggy commits",
    "file_age": "Changes touch very new or very old files",
    "ci_warning_count": "High CI warning count",
    "ci_error_count": "CI logs contain errors/exceptions",
    "ci_test_duration_delta": "Test run time changed sharply vs. baseline",
    "flaky_test_flag": "Flaky tests detected in CI",
    "text_risk_score": "Risky wording in PR/bug description",
    "commit_message_risk_keywords": "Risk keywords in commit message (hotfix, revert, migration...)",
}


def readable(feature: str) -> str:
    """Human-readable label; falls back to a cleaned-up column name."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def build_explainer():
    model = joblib.load(ART / "model.pkl")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, ART / "shap_explainer.pkl")
    return model, explainer


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    model, explainer = build_explainer()
    cols = json.loads((ART / "feature_meta.json").read_text())["feature_columns"]
    X = pd.read_csv(ART / "test_split.csv")[cols]

    sv = explainer.shap_values(X)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values()
    importance.index = [readable(c) for c in importance.index]

    plt.figure(figsize=(8, max(3, 0.4 * len(importance))))
    importance.plot.barh(color="#d9534f")
    plt.xlabel("Mean |SHAP value| (impact on risk, log-odds)")
    plt.title("Global Feature Importance")
    plt.tight_layout(); plt.savefig(REPORT / "shap_global_importance.png", dpi=150); plt.close()
    print(f"Saved explainer -> {ART / 'shap_explainer.pkl'}")
    print(f"Saved chart     -> {REPORT / 'shap_global_importance.png'}")


if __name__ == "__main__":
    main()
