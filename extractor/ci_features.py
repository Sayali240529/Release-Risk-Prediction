"""CI build-log feature extraction.

Parses a plain-text CI log and returns:
    ci_warning_count        lines containing "warning"
    ci_error_count          hits of ERROR / EXCEPTION / FATAL / TIMEOUT
    ci_failed_test_count    failed tests reported by the log
    ci_test_duration_delta  (test_time - baseline) / baseline  (0.0 if unknown)
    flaky_test_flag         1 if the log shows retries / flaky / intermittent tests
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

_WARN = re.compile(r"\bwarn(?:ing)?s?\b", re.I)
_ERR = re.compile(r"\b(ERROR|EXCEPTION|FATAL|TIMEOUT|TIMED OUT)\b", re.I)
_FLAKY = re.compile(r"flaky|\bretry\b|\bretried\b|\brerun\b|passed on retry|intermittent", re.I)

_DURATION = [
    re.compile(r"Ran\s+\d+\s+tests?\s+in\s+([\d.]+)\s*s", re.I),            # unittest
    re.compile(r"Finished in\s+([\d.]+)\s*(?:seconds|secs|s)\b", re.I),     # rspec/mocha
    re.compile(r"Time elapsed:\s*([\d.]+)\s*(?:sec|s)\b", re.I),            # maven
    re.compile(r"Total (?:test )?time:\s*([\d.]+)\s*s", re.I),
    re.compile(r"={3,}.*\bin\s+([\d.]+)s.*={3,}"),                          # pytest
]
_FAILED = [
    re.compile(r"failures=(\d+)", re.I),
    re.compile(r"(\d+)\s+failed", re.I),
    re.compile(r"Failures:\s*(\d+)", re.I),
]
_FAILED_LINE = re.compile(r"^\s*(?:FAILED\b|FAIL:)", re.M)


def extract_test_duration(text: str) -> Optional[float]:
    total, found = 0.0, False
    for rx in _DURATION:
        for m in rx.finditer(text):
            total += float(m.group(1))
            found = True
    return total if found else None


def _failed_tests(text: str) -> int:
    nums = [int(m.group(1)) for rx in _FAILED for m in rx.finditer(text)]
    return sum(nums) if nums else len(_FAILED_LINE.findall(text))


def parse_ci_log(text: str, baseline_duration: Optional[float] = None) -> Dict[str, float]:
    dur = extract_test_duration(text)
    if dur is not None and baseline_duration and baseline_duration > 0:
        delta = (dur - baseline_duration) / baseline_duration
    else:
        delta = 0.0
    return {
        "ci_warning_count": sum(1 for ln in text.splitlines() if _WARN.search(ln)),
        "ci_error_count": len(_ERR.findall(text)),
        "ci_failed_test_count": _failed_tests(text),
        "ci_test_duration_delta": round(delta, 4),
        "flaky_test_flag": int(bool(_FLAKY.search(text))),
    }


def parse_ci_file(path: str | Path, baseline_duration: Optional[float] = None) -> Dict[str, float]:
    return parse_ci_log(Path(path).read_text(errors="ignore"), baseline_duration)


class BaselineTracker:
    """Rolling median of previous test durations = 'baseline' for the next build."""

    def __init__(self, window: int = 20):
        self.window, self.hist = window, []  # type: int, List[float]

    def baseline(self) -> Optional[float]:
        return statistics.median(self.hist[-self.window:]) if self.hist else None

    def update(self, duration: Optional[float]) -> None:
        if duration is not None:
            self.hist.append(duration)


if __name__ == "__main__":
    demo = ("WARNING: deprecated API\nwarning: unused var\nERROR: db down\n"
            "[flaky] test_x passed on retry 2\nRan 120 tests in 150.00s\nFAILED (failures=2)\n")
    print(parse_ci_log(demo, baseline_duration=120.0))
