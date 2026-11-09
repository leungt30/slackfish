"""T5: the training data order and the validation split."""
import types
import unittest

import torch

import helpers  # noqa: F401 (import paths)
from NoFeat import score
from NoFeat.train import ShardStream, val_split


class ShardStreamTest(unittest.TestCase):
    def stream(self, n=10_007):
        return ShardStream(n, seed=7, shard_size=1000, window_shards=3, augment_prob=0.5)

    def test_every_position_once_per_epoch(self):
        n = 10_007
        stream = self.stream(n)
        for epoch in (1, 2):
            idx, swap = stream.take(epoch, 0, n)
            self.assertEqual(sorted(idx.tolist()), list(range(n)))
            self.assertEqual(len(swap), n)
        self.assertFalse(torch.equal(stream.take(1, 0, n)[0], stream.take(2, 0, n)[0]))   # a new order each epoch

    def test_any_slice_matches_the_full_order(self):
        """Resuming mid-epoch (take from an arbitrary start) rebuilds the same batches."""
        n = 10_007
        idx, swap = self.stream(n).take(3, 0, n)
        for start, count in ((0, 1), (999, 2), (2500, 4000), (n - 5, 5)):
            fresh = self.stream(n)   # a restarted process, with nothing cached
            got_idx, got_swap = fresh.take(3, start, count)
            self.assertTrue(torch.equal(got_idx, idx[start:start + count]), (start, count))
            self.assertTrue(torch.equal(got_swap, swap[start:start + count]), (start, count))


class ValidationSplitTest(unittest.TestCase):
    def test_split_is_a_partition(self):
        val, train = val_split(10_000, seed=42, frac=0.02)
        self.assertEqual(len(val), 200)
        self.assertEqual(sorted(torch.cat([val, train]).tolist()), list(range(10_000)))

    def test_score_uses_a_subset_of_trainings_validation(self):
        """NoFeat.score with --val-frac at most training's scores only held-out positions."""
        n = 10_000
        held_out = set(val_split(n, seed=42, frac=0.02)[0].tolist())
        data = {"eval": torch.zeros(n)}
        for frac in (0.005, 0.02):
            scored = score.validation_set(data, types.SimpleNamespace(seed=42, val_frac=frac))
            self.assertLessEqual(set(scored.tolist()), held_out, frac)
        bigger = score.validation_set(data, types.SimpleNamespace(seed=42, val_frac=0.05))
        self.assertFalse(set(bigger.tolist()) <= held_out)   # why --val-frac must not exceed training's

    def test_shuffled_folder_uses_its_last_positions(self):
        data = {"eval": torch.zeros(100), "val": 7}
        self.assertEqual(score.validation_set(data, None).tolist(), list(range(93, 100)))


if __name__ == "__main__":
    unittest.main()
