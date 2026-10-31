# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from datetime import datetime, timedelta

from core.apwr_rns.prime_supply import (
    DOCUMENTED_APWR_PRIMES, load_pool, select_moduli, reconstruct,
    exact_feature_numerators,
)
from core.apwr_rns.detector import predict, quantize


class APWRRNSSubmissionTests(unittest.TestCase):
    def test_prime_provenance_and_capacity(self):
        pool = load_pool()
        self.assertEqual(set(DOCUMENTED_APWR_PRIMES), set(pool))
        chosen = select_moduli(pool, max_abs_signal=150_000_000)
        self.assertEqual(len(chosen), 5)
        self.assertGreater(chosen[0]*chosen[1]*chosen[2],
                           2 * 2 * 15 * 60 * 150_000_000)

    def test_fault_channel_reconstruction(self):
        p = select_moduli(load_pool(), max_abs_signal=130_000_000)
        value = 38_003_097_932
        residues = [value % m for m in p]
        residues[1] = (residues[1] + 13) % p[1]
        recovered, bad = reconstruct(tuple(residues), p, bound=250_000_000_000)
        self.assertEqual(recovered, value)
        self.assertEqual(bad, (1,))

    def test_exact_window_feature_and_fault(self):
        samples = [quantize(f"{(i//40) * 0.125 + 67.0:.6f}") for i in range(450)]
        primes = select_moduli(load_pool(), max(map(abs, samples)))
        healthy = exact_feature_numerators(samples, primes)
        degraded = exact_feature_numerators(samples, primes, corrupt_lane=2)
        self.assertEqual(healthy, degraded)
        self.assertEqual(len(healthy), 450)
        self.assertEqual(healthy[:74], [None] * 74)
        i=190
        expected = abs(60*sum(samples[i-14:i+1])-15*sum(samples[i-74:i-14]))
        self.assertEqual(healthy[i], expected)

    def test_prediction_causal_and_reproducible(self):
        t0 = datetime(2020, 3, 9, 0, 0)
        rows = []
        for i in range(630):
            fault = i >= 480
            base = float(i % 9)*.001
            rows.append({
                "datetime": str(t0 + timedelta(seconds=i)),
                "Accelerometer2RMS": str(.04 + base + (0.09 if fault else 0)),
                "Current": str(1.3 + base),
                "Temperature": str(70 + base + (3 if fault else 0)),
                "Volume Flow RateRMS": str(32 + base),
            })
        a = predict(rows, load_pool())
        b = predict(rows, load_pool(), corrupt_lane=2)
        self.assertEqual(a["test_predictions"], b["test_predictions"])
        self.assertTrue(any(a["test_predictions"]))
        self.assertTrue(all(a["test_datetimes"][i] >= t0 for i in range(230)))


if __name__ == "__main__":
    unittest.main()