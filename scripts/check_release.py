#!/usr/bin/env python
"""CI deployment gate.

    python ci_gate/check_release.py --input release_features.json

Exit codes: 0 = AUTO_PROMOTE (deploy), 1 = HOLD_CANARY or MANUAL_REVIEW (block),
            2 = the gate itself failed (bad file, unknown feature...). Also non-zero, so CI fails safe.
Input JSON: flat {"commit_churn": 900, ...} or {"features": {...}}. Missing features use training medians.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Release risk gate")
    ap.add_argument("--input", required=True, help="JSON file with release features")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = ap.parse_args(argv)

    try:
        data = json.loads(Path(args.input).read_text())
        from model.predict_api import predict
        result = predict(data.get("features", data))
    except Exception as e:  # fail closed
        print(f"ERROR: release gate could not run: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Risk score : {result['score']:.0%}")
        print(f"Decision   : {result['action']}")
        for label, val in result["top_3_factors"]:
            print(f"  - {label} ({val:+.2f})")

    return 0 if result["action"] == "AUTO_PROMOTE" else 1


if __name__ == "__main__":
    sys.exit(main())
