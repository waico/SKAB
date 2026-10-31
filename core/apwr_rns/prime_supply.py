# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only APWR prime catalog and exact, single-lane-correcting RNS computations.

The source prime factory (Apwr11.py) is NOT modified, imported or rerun.
Prime provenance: original APWR research report at reports/prime-instrument-reference.json,
and optionally reports/prime-window-primes.csv (recorded APW01/APW02 outputs).
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from math import isqrt
from pathlib import Path

# Small proven APWR-polynomial-output primes, with provenance documented in README.
# These are fallback catalog entries: they are NOT newly synthesized primes.
DOCUMENTED_APWR_PRIMES = (474343, 486103, 508037, 514103, 484621)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for divisor in range(2, isqrt(n) + 1):
        if n % divisor == 0:
            return False
    return True


def load_pool(report: Path | None = None) -> tuple[int, ...]:
    """Import recorded APWR polynomial outputs, selecting only proven primes."""
    candidate = set(DOCUMENTED_APWR_PRIMES)
    if report is not None and report.exists():
        with report.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for p in json.loads(row.get("returned_proven_primes", "[]")):
                    candidate.add(int(p))
    return tuple(sorted(p for p in candidate if is_prime(p)))


def select_moduli(pool: tuple[int, ...], max_abs_signal: int,
                  fast: int = 15, slow: int = 60,
                  count: int = 5, data_lanes: int = 3) -> tuple[int, ...]:
    """Choose minimum-size primes satisfying exact CRT bounds and 2-lane redundancy.

    Each signal numerator = slow * sum(last fast) - fast * sum(previous slow).
    Thus abs(numerator) <= 2 * fast * slow * max_abs_sample.
    """
    if max_abs_signal < 0 or fast <= 0 or slow <= 0:
        raise ValueError("Invalid workload bounds")
    if count < data_lanes + 2:
        raise ValueError("Need two redundant channels")
    bound = 2 * fast * slow * max_abs_signal
    candidates = sorted(set(pool), key=lambda p: (p.bit_length(), p))
    # Choose smallest usable size class; no requirement that p-1 has a large 2-adic
    # factor, because this application computes streaming sums, NOT an NTT.
    for end in range(count, len(candidates) + 1):
        chosen = tuple(candidates[end-count:end])
        smallest_product = 1
        for p in chosen[:data_lanes]:
            smallest_product *= p
        if smallest_product > 2 * bound:
            return chosen
    raise ValueError("APWR pool capacity insufficient; request larger primes from APWR menu 2/21")


def crt_signed(residues: tuple[int, ...], primes: tuple[int, ...]) -> int:
    x, modulus = 0, 1
    for residue, p in zip(residues, primes):
        x += modulus * ((residue - x) * pow(modulus, -1, p) % p)
        modulus *= p
    return x if 2 * x <= modulus else x - modulus


def reconstruct(residues: tuple[int, ...], primes: tuple[int, ...],
                bound: int, data_lanes: int = 3) -> tuple[int, tuple[int, ...]]:
    """Accept a UNIQUE candidate matching >= count-1 lanes, reject ambiguity.

    Guarantees tested one-lane residue corruption is corrected, provided
    the CRT bound holds for every relevant data-lane combination.
    """
    if len(residues) != len(primes):
        raise ValueError("Mismatched residues and moduli")
    # Fast path: normal operation requires only three CRT data lanes and two
    # consistency checks. Recover from any single corrupted lane on demand.
    first = crt_signed(residues[:data_lanes], primes[:data_lanes])
    if abs(first) <= bound and sum(first % primes[j] == residues[j]
                                   for j in range(len(primes))) >= len(primes) - 1:
        value = first
    else:
        candidates = set()
        for selection in combinations(range(len(primes)), data_lanes):
            p = tuple(primes[i] for i in selection)
            r = tuple(residues[i] for i in selection)
            x = crt_signed(r, p)
            if abs(x) <= bound and sum(x % primes[j] == residues[j]
                                       for j in range(len(primes))) >= len(primes) - 1:
                candidates.add(x)
        if len(candidates) != 1:
            raise ArithmeticError("Residue consensus missing or ambiguous")
        value = candidates.pop()
    bad = tuple(i for i, p in enumerate(primes) if value % p != residues[i])
    return value, bad


def exact_feature_numerators(samples: list[int], primes: tuple[int, ...],
                             fast: int = 15, slow: int = 60,
                             corrupt_lane: int | None = None) -> list[int | None]:
    """Compute window-difference numerators using APWR-selected residue lanes."""
    if not samples:
        return []
    if corrupt_lane is not None and not 0 <= corrupt_lane < len(primes):
        raise ValueError("Invalid residue lane")
    bound = 2 * fast * slow * max(map(abs, samples))
    # Proof-capacity check applies to every data_lanes subset, not just the first.
    for chosen in combinations(primes, 3):
        if chosen[0] * chosen[1] * chosen[2] <= 2 * bound:
            raise ArithmeticError("Insufficient certified CRT range")

    prefix = [[0] for _ in primes]
    for v in samples:
        for j, p in enumerate(primes):
            prefix[j].append((prefix[j][-1] + v) % p)

    result: list[int | None] = [None] * len(samples)
    for i in range(fast + slow - 1, len(samples)):
        rs = []
        for j, p in enumerate(primes):
            seq = prefix[j]
            recent = (seq[i + 1] - seq[i + 1 - fast]) % p
            older = (seq[i + 1 - fast] - seq[i + 1 - fast - slow]) % p
            residue = (slow * recent - fast * older) % p
            if corrupt_lane == j:
                residue = (residue + 1) % p
            rs.append(residue)
        number, _ = reconstruct(tuple(rs), primes, bound)
        result[i] = abs(number)
    return result