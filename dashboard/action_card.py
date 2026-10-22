"""Big coloured card showing the deployment decision + one explanatory sentence."""
import streamlit as st

CARDS = {
    "AUTO_PROMOTE": ("PROMOTE", "#2ecc71", "#ffffff",
                     "Risk is low, so this release is safe to deploy normally."),
    "HOLD_CANARY": ("HOLD - CANARY", "#f1c40f", "#222222",
                    "Risk is moderate, so roll out to about 5% of traffic and monitor closely before promoting."),
    "MANUAL_REVIEW": ("ROLLBACK / MANUAL REVIEW", "#e74c3c", "#ffffff",
                      "Risk is high, so block the deployment and have the team review before release."),
}


def render_action_card(action: str, score: float) -> None:
    title, bg, fg, sentence = CARDS.get(action, (action, "#888888", "#ffffff", "Unrecognised action."))
    html = (
        f'<div style="background:{bg};color:{fg};padding:24px;border-radius:14px;text-align:center;">'
        f'<div style="font-size:2.4rem;font-weight:800;letter-spacing:1px;">{title}</div>'
        f'<div style="font-size:1.1rem;margin-top:8px;">Risk score {score:.0%}. {sentence}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
