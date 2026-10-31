# APWR-RNS: causal industrial change-point detection on SKAB

**Submission candidate — reproducible with the upstream SKAB evaluator.**
This public implementation is self-contained; it **does not contain, import or modify**
the private APWR prime-generation engine or any private project files.

## Evaluation protocol

- Corpus: the 34 annotated SKAB v0.9 experiments (excluding the anomaly-free data).
- Split: first **400 rows of each recording** for calibration; predict subsequent rows only.
- Sensor channels: `Accelerometer2RMS`, `Current`, `Temperature`,
  `Volume Flow RateRMS` (flow-rate channel).
- Signal: exact integer numerator of mean(last 15) minus mean(preceding 60),
  with values quantized using `Decimal` to six decimal places.
- Calibration: 99th percentile of eligible training-window signals, rows 77–399.
- Level 1: three consecutive above-threshold observations on any channel,
  then a 60-second alert cooldown. **Level 1 is the benchmark output.**
- Level 2: corroboration by two distinct channels within 60 seconds
  of a Level-1 alert. Level 2 escalates operational severity but does not
  suppress Level-1 warnings.
- Benchmark: upstream `core.metrics.chp_score` with
  `metric="nab"`, `window_width="60s"`,
  `anomaly_window_destination="righter"`. Uses upstream's
  `average_time` scoring for delay, missed CPs and false positives.

## APWR as computational infrastructure

The five small prime moduli `474343, 486103, 508037, 514103, 484621`
are recorded outputs from the CptnTech APWR APW01/APW02 prime-generation
research. The public integration verifies their primality by exact trial
division, selects suitable moduli based on the numerical workload bound,
and computes each feature by exact modular prefix sums.

Each group of three of the five prime channels has sufficient combined
range for the signed result. CRT consensus can identify and correct the
single-lane residue corruption tested by the regression suite. This protects
**arithmetic computation**, not physical sensor measurements. No 200-digit
Titan prime is required for this streaming calculation. The engine is
deliberately standalone; no APWR prime-generation internals are exposed.

The public repository does not independently verify the historical source
provenance of the listed prime outputs; the exact primality and arithmetic
claims are independently testable here.

## Reproduce

From the root of this SKAB fork, using Python 3.10+:

```bash
python -m pip install numpy pandas matplotlib
python -m unittest discover -s tests -p 'test_apwr_rns.py' -v
python -m core.apwr_rns.benchmark --skab-root . --out results/results-APWR_RNS.json
```

A directly executable notebook is at `notebooks/APWR_RNS.ipynb`.
The report from the reference replay is checked in as
`results/results-APWR_RNS.json`. All computation is causal at the
observation level; test labels are read by the evaluation runner **only
after** predictions are generated.

## Scores on SKAB v0.9

The full 34-recording replay was verified on the upstream SKAB evaluator
at commit `b2c0d46c2971dcbfe71e26087b6d231998bb91c2`:

| Measure | APWR-RNS (proposed) |
| --- | ---: |
| NAB Standard | **58.07** |
| NAB LowFP | **49.35** |
| NAB LowFN | **66.27** |
| Missed change points | 22 of 127 |
| False-positive alerts | 161 |
| Mean detection delay | 23.94 seconds |

The 34 recordings are publicly accessible; the five `valve1/0.csv` through
`valve1/4.csv` files were considered during prototype selection.
The other 29 were not used for choosing the fast-warning rule; their
recording-level holdout Standard score was about **55.35** under the
same 400-row split. This is *not* a blind private benchmark.

The `other/2.csv` recording contains anomaly-marked rows within its
first 400 training observations, an issue shared with the baseline
loader. No labels were used to tune thresholds during the scored portion.

**Publication status:** these are independently reproducible local
scores obtained with SKAB's own metric code; no ranking or official
first place is claimed until SKAB maintainers review and accept a PR.

## License and attribution

The standalone public code is provided under SPDX
`GPL-3.0-or-later`, consistent with the SKAB repository license.
Original SKAB dataset, metric implementation and baseline methods are
credited to their upstream authors.
