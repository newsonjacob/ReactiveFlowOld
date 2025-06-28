#!/usr/bin/env python3
"""Summarize flow log CSV files.

This script scans ``flow_logs`` for files named ``full_log_*.csv`` and
prints basic statistics for each run: total frame count, the number of
frames with a collision, and the straight‑line distance traveled from
the first to the last recorded position.
"""

from __future__ import annotations

import argparse
import glob
import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def summarize_log(path: str) -> Tuple[int, int, float]:
    """Return summary statistics for a log file.

    Args:
        path: Path to a ``full_log_*.csv`` file.

    Returns:
        ``(frames, collisions, distance)`` where ``frames`` is the
        number of rows in the CSV, ``collisions`` counts how many rows
        have ``collided`` > 0, and ``distance`` is the Euclidean
        distance between the first and last ``(pos_x, pos_y, pos_z)``
        entries.
    """
    df = pd.read_csv(path)

    frames = len(df)
    if "collided" in df.columns:
        collisions = int((df["collided"] > 0).sum())
    else:
        collisions = 0

    start = df.loc[0, ["pos_x", "pos_y", "pos_z"]].to_numpy()
    end = df.loc[df.index[-1], ["pos_x", "pos_y", "pos_z"]].to_numpy()
    distance = float(np.linalg.norm(end - start))

    return frames, collisions, distance


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Summarize UAV run logs")
    parser.add_argument(
        "--log-dir",
        default="flow_logs",
        help="Directory containing full_log_*.csv files",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.log_dir, "full_log_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning("No log files found matching %s", pattern)
        return

    for path in files:
        try:
            frames, collisions, distance = summarize_log(path)
            name = os.path.basename(path)
            logger.info(
                "%s: frames=%d, collisions=%d, distance=%.2f",
                name,
                frames,
                collisions,
                distance,
            )
        except Exception as exc:
            logger.error("Error processing %s: %s", path, exc)


if __name__ == "__main__":
    main()
