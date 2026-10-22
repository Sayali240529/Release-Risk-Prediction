"""Train the CatBoost release-risk classifier.

Usage:  python model/train.py [--data data/features.csv]
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).resolve().parent / "artifacts"
LABEL = "risk_label"
SEED = 42


def load_data(path: Path):
    df = pd.read_csv(path)
    if LABEL not in df.columns:
        raise ValueError(f"'{LABEL}' column missing from {path}")
    # Keep numeric feature columns only (drops ids like commit hashes / repo names).
    X = df.drop(columns=[LABEL]).select_dtypes(include="number")
    y = df[LABEL].astype(int)
    return X, y


def main(data_path: Path):
    ART.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    print(f"Rows: {len(X)}  Features: {X.shape[1]}  Positive rate: {y.mean():.3f}")

    # 80/20 split, stratified so both sets keep the failure rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    # Small validation slice from TRAIN only (for early stopping) -> test set stays untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    # Class imbalance: weight failures by (#clean / #failures).
    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    print(f"scale_pos_weight = {spw:.2f}")

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        eval_metric="PRAUC",
        scale_pos_weight=spw,
        random_seed=SEED,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

    joblib.dump(model, ART / "model.pkl")
    X_train.assign(**{LABEL: y_train}).to_csv(ART / "train_split.csv", index=False)
    X_test.assign(**{LABEL: y_test}).to_csv(ART / "test_split.csv", index=False)
    meta = {
        "feature_columns": list(X.columns),
        "defaults": X_train.median().to_dict(),  # used by predict() for missing inputs
        "scale_pos_weight": spw,
    }
    (ART / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {ART / 'model.pkl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "features.csv"))
    main(Path(ap.parse_args().data))
