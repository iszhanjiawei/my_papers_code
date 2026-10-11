"""Small offline tests of sampling balance, leakage guards and duration checks."""
from collections import Counter, defaultdict
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from infer import finalize
from protocol import COUNT, DEFAULT_RUN, rows


class ProtocolTests(unittest.TestCase):
    def test_balanced_assignment(self):
        assignments = [(s, (2 * s + 7 * t) % 66) for s in range(33) for t in range(20)]
        self.assertEqual(Counter(r for _, r in assignments), Counter({r: 10 for r in range(66)}))
        for r in range(66):
            self.assertEqual(len({s for s, ref in assignments if ref == r}), 10)
        for s in range(33):
            self.assertEqual(len({r for sp, r in assignments if sp == s}), 20)

    def test_duration_adjustment_is_bounded_tail_only(self):
        wave = np.full(48000, .1, dtype=np.float32)
        fixed, delta = finalize(wave, 16000, 48160)
        self.assertEqual(delta, 160)
        np.testing.assert_array_equal(fixed[:48000], wave)
        self.assertTrue(np.all(fixed[48000:] == 0))
        with self.assertRaises(ValueError):
            finalize(wave, 16000, 50000)

    def test_nonfinite_and_silence_rejected(self):
        for wave in (np.zeros(48000), np.full(48000, np.nan), np.empty(0)):
            with self.assertRaises(ValueError):
                finalize(wave, 16000, 48000)

    def test_resampling_preserves_duration(self):
        wave = np.full(66150, .1, dtype=np.float32)
        fixed, delta = finalize(wave, 22050, 48000)
        self.assertEqual(len(fixed), 48000)
        self.assertEqual(delta, 0)

    @unittest.skipUnless((DEFAULT_RUN / 'selection.json').exists(), 'No local frozen manifest')
    def test_actual_manifest_balance_and_no_gt_input(self):
        items = rows(DEFAULT_RUN)
        self.assertEqual(len(items), COUNT)
        self.assertEqual(set(Counter(r['target_speaker_id'] for r in items).values()), {20})
        self.assertEqual(len({r['ref_id'] for r in items}), 66)
        self.assertEqual(set(Counter(r['ref_id'] for r in items).values()), {10})
        self.assertEqual(len({r['target_speaker_id'] for r in items[:33]}), 33)
        self.assertTrue(all('target_gt_audio' not in r and len(r['target_text'].split()) == 6 for r in items))
        self.assertTrue(all('target_gt_audio' in r for r in rows(DEFAULT_RUN, evaluation=True)))


if __name__ == '__main__':
    unittest.main()
