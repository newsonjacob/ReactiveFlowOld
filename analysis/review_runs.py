#!/usr/bin/env python3
"""Review and visualize UAV run logs."""

from __future__ import annotations

import glob
import os
import subprocess
import logging
from typing import List, Tuple

# Use absolute import so the script works when executed directly
import sys

# Ensure the repository root is on sys.path when executed directly
if __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )

from analysis.summarize_runs import summarize_log
from analysis.utils import retain_recent_views

logger = logging.getLogger(__name__)


def ensure_visualization(log_path: str, html_path: str) -> None:
    """Generate a flight visualization if ``html_path`` does not exist."""
    if os.path.exists(html_path):
        return

    cmd = [
        "python",
        os.path.join("analysis", "visualize_flight.py"),
        "--log",
        log_path,
        "--obstacles",
        os.path.join("analysis", "obstacles.json"),
        "--output",
        html_path,
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    pattern = os.path.join("flow_logs", "full_log_*.csv")
    logs = sorted(glob.glob(pattern))
    if not logs:
        logger.warning("No log files found matching %s", pattern)
        return

    results: List[Tuple[str, int, int, float]] = []

    for path in logs:
        try:
            frames, collisions, distance = summarize_log(path)
            results.append((path, frames, collisions, distance))
        except Exception as exc:
            logger.error("Error processing %s: %s", path, exc)
            continue

        timestamp = os.path.basename(path)[len("full_log_"):-len(".csv")]
        html_name = f"flight_view_{timestamp}.html"
        html_path = os.path.join("analysis", html_name)
        ensure_visualization(path, html_path)
        retain_recent_views("analysis")

    report_lines = ["log,frames,collisions,distance"]
    for path, frames, collisions, distance in results:
        name = os.path.basename(path)
        report_lines.append(f"{name},{frames},{collisions},{distance:.2f}")

    report_path = os.path.join("analysis", "summary_report.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))
    logger.info("Summary written to %s", report_path)


if __name__ == "__main__":
    main()
