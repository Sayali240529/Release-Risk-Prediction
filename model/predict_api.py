"""Interface for Member 3. Only call predict(); never touch the model directly.

    from model.predict_api import predict
    predict({"commit_churn": 900, "ci_warning_count": 14, ...})
    -> {"score": 0.73, "action": "MANUAL_REVIEW",
        "top_3_factors": [("High CI warning count", 0.21), ...],   # human-readable label, SHAP value
        "top_3_raw": [("ci_warning_count", 0.21), ...],            # raw column names
        "shap_all": {"ci_warning_count": 0.21, ...}}               # every feature (for charts)

SHAP values are in log-odds units: positive = pushes risk UP, negative = pushes it DOWN.
Missing features are filled with the training-set median.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

try:
    from model.explain import readable
    from model.policy import decide
except ImportError:  # when run directly from inside model/
    from explain import readable
    from policy import decide

ART = Path(__file__).resolve().parent / "artifacts"


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(ART / "model.pkl")
    meta = json.loads((ART / "feature_meta.json").read_text())
    try:
        explainer = joblib.load(ART / "shap_explainer.pkl")
    except Exception:  # missing/incompatible pickle -> rebuild on the fly
        import shap
        explainer = shap.TreeExplainer(model)
    return model, explainer, meta


def predict(features_dict: dict) -> dict:
    model, explainer, meta = _load()
    cols, defaults = meta["feature_columns"], meta["defaults"]

    unknown = set(features_dict) - set(cols)
    if unknown:
        raise ValueError(f"Unknown feature(s): {sorted(unknown)}. Expected: {cols}")

    row = {c: features_dict.get(c, defaults[c]) for c in cols}
    X = pd.DataFrame([row], columns=cols)

    score = float(model.predict_proba(X)[0, 1])
    shap_row = explainer.shap_values(X)[0]
    contrib = sorted(zip(cols, (float(v) for v in shap_row)), key=lambda t: t[1], reverse=True)

    top = [(c, round(v, 3)) for c, v in contrib[:3] if v > 0]  # only factors that RAISE risk
    return {
        "score": round(score, 4),
        "action": decide(score),
        "top_3_factors": [(readable(c), v) for c, v in top],
        "top_3_raw": top,
        "shap_all": {c: round(v, 4) for c, v in contrib},
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(predict({}))
