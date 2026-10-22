"""Release Risk Engine dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.predict_api import predict  # noqa: E402  (the ONLY model interface we use)
from dashboard.components.action_card import render_action_card  # noqa: E402
from dashboard.components.gauge import risk_gauge  # noqa: E402
from dashboard.components.shap_chart import shap_bar_chart  # noqa: E402

HISTORY_FILE = Path(__file__).resolve().parent / "history.csv"
OUTCOMES = ["Unknown", "Clean", "Incident", "Reverted"]

# (column, label, min, max, default, step, type). Keep names/units identical to data/features.csv.
FEATURES = [
    ("commit_churn", "Lines changed (added + deleted)", 0, 5000, 300, 10, int),
    ("files_modified", "Files modified", 1, 100, 5, 1, int),
    ("directory_entropy", "Directory entropy (0 = one module, higher = scattered)", 0.0, 3.0, 0.8, 0.05, float),
    ("author_risk_score", "Author's historical bug rate (0-1)", 0.0, 1.0, 0.10, 0.01, float),
    ("ci_warning_count", "CI warning count", 0, 100, 5, 1, int),
    ("ci_test_duration_delta", "Test duration change vs baseline (0.2 = +20%)", -1.0, 2.0, 0.0, 0.05, float),
]

# TODO: replace with Member 1's extractor/text_features.py so inference matches training.
RISK_WORDS = {"hotfix", "revert", "rollback", "migration", "security", "regression",
              "critical", "urgent", "schema", "database", "breaking", "patch"}


def text_to_features(text: str) -> dict:
    words = [w.strip(".,:;!?()[]\"'").lower() for w in text.split()]
    hits = sum(w in RISK_WORDS for w in words)
    return {"commit_message_risk_keywords": hits, "text_risk_score": min(1.0, hits / 4)}


def feature_inputs(init: dict, prefix: str) -> dict:
    """Sliders for the numeric features + flaky checkbox. Returns a features dict."""
    out = {}
    cols = st.columns(2)
    for i, (name, label, lo, hi, default, step, typ) in enumerate(FEATURES):
        v = min(max(typ(init.get(name, default)), typ(lo)), typ(hi))
        out[name] = cols[i % 2].slider(label, typ(lo), typ(hi), v, typ(step), key=f"{prefix}_{name}")
    out["flaky_test_flag"] = int(st.checkbox(
        "Flaky tests detected in CI", value=bool(init.get("flaky_test_flag", 0)), key=f"{prefix}_flaky"))
    return out


def load_history() -> pd.DataFrame:
    cols = ["timestamp", "release", "score", "action", "top_factor", "actual_outcome"]
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=cols)


def add_history(name, result):
    top = result["top_3_factors"][0][0] if result["top_3_factors"] else "-"
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "release": name,
           "score": result["score"], "action": result["action"],
           "top_factor": top, "actual_outcome": "Unknown"}
    pd.concat([load_history(), pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)


# ---------------------------------------------------------------- pages
def page_input():
    st.header("Release Input")
    with st.form("release_form"):
        name = st.text_input("Release / PR name", "release-v1.0")
        pr_text = st.text_area("PR title, commit message or bug report text",
                               "Hotfix: database migration for user table")
        feats = feature_inputs({}, "form")
        submitted = st.form_submit_button("Analyze Risk", type="primary")
    if submitted:
        features = {**feats, **text_to_features(pr_text)}
        try:
            result = predict(features)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        st.session_state["current"] = {"id": datetime.now().timestamp(), "name": name,
                                       "features": features, "result": result}
        add_history(name, result)
        st.session_state["goto"] = "Risk Analysis"
        st.rerun()


def page_analysis():
    st.header("Risk Analysis")
    cur = st.session_state.get("current")
    if not cur:
        st.info("No release analysed yet. Go to **Release Input** first.")
        return
    res = cur["result"]
    st.subheader(cur["name"])
    left, right = st.columns([1, 1])
    left.plotly_chart(risk_gauge(res["score"]), use_container_width=True)
    with right:
        render_action_card(res["action"], res["score"])
        st.markdown("#### Top risk drivers")
        if res["top_3_factors"]:
            for label, val in res["top_3_factors"]:
                st.markdown(f"- **{label}** (impact {val:+.2f})")
        else:
            st.markdown("No factor is pushing the risk up noticeably.")
    st.plotly_chart(shap_bar_chart(res), use_container_width=True)

    with st.expander("What-if simulator"):
        st.caption("Change the inputs to see how the score and decision would move.")
        wi = feature_inputs(cur["features"], f"wi{cur['id']}")
        wi_features = {**cur["features"], **wi}
        new = predict(wi_features)
        c1, c2 = st.columns(2)
        c1.metric("What-if risk score", f"{new['score']:.0%}",
                  delta=f"{(new['score'] - res['score']) * 100:+.1f} pts", delta_color="inverse")
        c2.metric("What-if action", new["action"])


def page_history():
    st.header("History")
    df = load_history()
    if df.empty:
        st.info("No releases analysed yet.")
        return
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True,
        disabled=[c for c in df.columns if c != "actual_outcome"],
        column_config={"actual_outcome": st.column_config.SelectboxColumn("actual_outcome", options=OUTCOMES),
                       "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    if not edited.equals(df):
        edited.to_csv(HISTORY_FILE, index=False)
        st.toast("Outcome saved")


# ---------------------------------------------------------------- main
st.set_page_config(page_title="Release Risk Engine", page_icon="🚦", layout="wide")
if "goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("goto")
st.title("🚦 Release Risk Engine")
page = st.sidebar.radio("Navigate", ["Release Input", "Risk Analysis", "History"], key="nav")
{"Release Input": page_input, "Risk Analysis": page_analysis, "History": page_history}[page]()
