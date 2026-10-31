# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the official SKAB change-point evaluator on APWR-backed predictions.

Usage (from SKAB checkout root, with SKAB dependencies installed):
  python /path/to/benchmark.py --skab-root . --apwr-report /path/to/reports/prime-window-primes.csv

Predicts only rows 400 onward. Uses SKAB core.metrics.chp_score rather than
claiming a custom score as an official benchmark measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .detector import load_csv, predict_file, TRAIN_ROWS


def run(skab_root: Path, report: Path | None = None) -> dict:
    import pandas as pd

    skab_root = skab_root.resolve()
    metric_path = skab_root / "core" / "metrics.py"
    if not metric_path.exists():
        raise FileNotFoundError("Missing official SKAB core/metrics.py")
    spec = importlib.util.spec_from_file_location("skab_official_metrics", metric_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = sorted(
        (p for p in (skab_root / "data").rglob("*.csv")
         if "anomaly-free" not in p.parts),
        key=lambda x: str(x)
    )
    if len(paths) != 34:
        raise ValueError(f"Expected 34 labelled SKAB files, found {len(paths)}")

    cp_true, cp_pred = [], []
    files = []
    for path in paths:
        rows = load_csv(path)
        answer = predict_file(path, report)
        ix = pd.to_datetime([r["datetime"] for r in rows[TRAIN_ROWS:]])
        truth = pd.Series([int(float(r["changepoint"]))
                           for r in rows[TRAIN_ROWS:]], index=ix, dtype=int)
        pred = pd.Series(answer["test_predictions"], index=ix, dtype=int)
        cp_true.append(truth)
        cp_pred.append(pred)
        files.append({
            "file": path.relative_to(skab_root).as_posix(),
            "training_rows": TRAIN_ROWS,
            "scored_rows": len(pred),
            "labelled_change_points": int(truth.sum()),
            "predicted_warning_points": int(pred.sum()),
            "apwr_prime_moduli": list(answer["primes"]),
            "has_anomalous_training_rows": any(
                float(r["anomaly"]) == 1.0 for r in rows[:TRAIN_ROWS]
            ),
        })

    scores = module.chp_score(
        cp_true, cp_pred, metric="nab", window_width="60s",
        anomaly_window_destination="righter", verbose=False
    )
    delay, missed, false_alarms, true_cp = module.chp_score(
        cp_true, cp_pred, metric="average_time", window_width="60s",
        anomaly_window_destination="righter", verbose=False
    )
    return {
        "benchmark": "waico/SKAB",
        "scoring_function": "core.metrics.chp_score",
        "training_rows_per_file": TRAIN_ROWS,
        "detector": "causal APWR RNS 15-sample-vs-60-prior-window; 99th pct; 3-consecutive; 60s cooldown",
        "scoring_profile": "NAB, 60s righter",
        "scores": scores,
        "change_points": int(true_cp),
        "missed_change_points": int(missed),
        "false_alarms": int(false_alarms),
        "mean_detection_delay_seconds": float(pd.Timedelta(delay).total_seconds()) if pd.notna(delay) else None,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skab-root", required=True, type=Path)
    parser.add_argument("--apwr-report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.skab_root, args.apwr_report)
    txt = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()