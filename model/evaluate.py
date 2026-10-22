"""Evaluate the trained model on the held-out test split.

Reports PR-AUC (main metric), ROC-AUC, F1 and the confusion matrix. No plain accuracy.
Usage: python model/evaluate.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)

ART = Path(__file__).resolve().parent / "artifacts"
REPORT = ART / "report"
LABEL = "risk_label"


def main(threshold: float = 0.5):
    REPORT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(ART / "model.pkl")
    cols = json.loads((ART / "feature_meta.json").read_text())["feature_columns"]
    test = pd.read_csv(ART / "test_split.csv")
    X, y = test[cols], test[LABEL]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "baseline_pr_auc (positive rate)": float(y.mean()),
        "threshold": threshold,
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y, pred).tolist(),
    }
    print(json.dumps(metrics, indent=2))
    (REPORT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # PR curve (with no-skill baseline)
    p, r, _ = precision_recall_curve(y, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, label=f"CatBoost (PR-AUC={metrics['pr_auc']:.3f})")
    plt.axhline(y.mean(), ls="--", c="grey", label="No-skill baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORT / "pr_curve.png", dpi=150); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"CatBoost (ROC-AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORT / "roc_curve.png", dpi=150); plt.close()

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Clean", "Risky"])
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout(); plt.savefig(REPORT / "confusion_matrix.png", dpi=150); plt.close()
    print(f"Charts saved in {REPORT}")


if __name__ == "__main__":
    main()
