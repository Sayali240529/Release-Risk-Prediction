"""Policy engine: converts a risk probability into a deployment action."""

AUTO_PROMOTE = "AUTO_PROMOTE"
HOLD_CANARY = "HOLD_CANARY"
MANUAL_REVIEW = "MANUAL_REVIEW"

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.70


def decide(score: float) -> str:
    """Risk < 0.35 -> AUTO_PROMOTE; 0.35 <= Risk < 0.70 -> HOLD_CANARY; Risk >= 0.70 -> MANUAL_REVIEW."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    if score < LOW_THRESHOLD:
        return AUTO_PROMOTE
    if score < HIGH_THRESHOLD:
        return HOLD_CANARY
    return MANUAL_REVIEW


if __name__ == "__main__":
    for s in (0.0, 0.34, 0.35, 0.69, 0.70, 1.0):
        print(s, decide(s))
