# SPDX-License-Identifier: GPL-3.0-or-later
"""APWR-backed causal two-level SKAB changepoint detector.

Leaderboard channel: Level 1 warning. Level 2 adds safety/maintenance severity,
not a suppression of warnings. No access to test labels during fitting or scoring.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from math import ceil
from pathlib import Path

from .prime_supply import load_pool, select_moduli, exact_feature_numerators

SENSOR_COLUMNS = (
    "Accelerometer2RMS", "Current", "Temperature", "Volume Flow RateRMS"
)
SCALE = 10 ** 6
FAST = 15
SLOW = 60
TRAIN_ROWS = 400
CALIBRATION_BEGIN = FAST + SLOW + 2
QUANTILE = 0.99
PERSISTENCE = 3
COOLDOWN_SECONDS = 60
CONFIRMATION_SECONDS = 60


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        rows = list(r)
    if len(rows) <= TRAIN_ROWS:
        raise ValueError(f"Not enough rows: {path}")
    if not all(c in rows[0] for c in SENSOR_COLUMNS):
        raise ValueError("Missing expected SKAB sensor channels")
    return rows


def quantize(s: str) -> int:
    return int((Decimal(s) * SCALE).to_integral_value(rounding=ROUND_HALF_UP))


def _threshold(numbers: list[int | None]) -> int:
    values = sorted(v for v in numbers[CALIBRATION_BEGIN:TRAIN_ROWS] if v is not None)
    if not values:
        raise ValueError("Insufficient normal calibration measurements")
    return values[ceil(QUANTILE * len(values)) - 1]


def predict(rows: list[dict[str, str]], prime_pool: tuple[int, ...],
            corrupt_lane: int | None = None) -> dict:
    """Return binary changepoint warnings on rows 400: and optional confirmations."""
    parsed = {key: [quantize(r[key]) for r in rows] for key in SENSOR_COLUMNS}
    maximum = max(abs(v) for values in parsed.values() for v in values[:TRAIN_ROWS])
    primes = select_moduli(prime_pool, maximum, fast=FAST, slow=SLOW)

    active = []
    for key in SENSOR_COLUMNS:
        signal = exact_feature_numerators(parsed[key], primes, FAST, SLOW,
                                         corrupt_lane=corrupt_lane)
        threshold = _threshold(signal)
        active.append([i >= TRAIN_ROWS and x is not None and x > threshold
                       for i, x in enumerate(signal)])

    timestamps = [datetime.fromisoformat(row["datetime"]) for row in rows]
    warnings: list[int] = []
    confirmations: list[int] = []
    run = 0
    next_allowed = datetime.min
    for i in range(TRAIN_ROWS, len(rows)):
        run = run + 1 if any(mask[i] for mask in active) else 0
        if run < PERSISTENCE or timestamps[i] < next_allowed:
            continue
        warnings.append(i)
        run = 0
        next_allowed = timestamps[i] + timedelta(seconds=COOLDOWN_SECONDS)
        corroborated = set()
        for j in range(i, len(rows)):
            if timestamps[j] > timestamps[i] + timedelta(seconds=CONFIRMATION_SECONDS):
                break
            corroborated.update(k for k, mask in enumerate(active) if mask[j])
            if len(corroborated) >= 2:
                confirmations.append(j)
                break
    warning_set = set(warnings)
    return {
        "primes": primes,
        "warnings": warnings,
        "confirmations": confirmations,
        "test_predictions": [int(i in warning_set) for i in range(TRAIN_ROWS, len(rows))],
        "test_datetimes": timestamps[TRAIN_ROWS:],
    }


def predict_file(csv_path: Path, apwr_report: Path | None = None,
                 corrupt_lane: int | None = None) -> dict:
    pool = load_pool(apwr_report)
    return predict(load_csv(csv_path), pool, corrupt_lane)