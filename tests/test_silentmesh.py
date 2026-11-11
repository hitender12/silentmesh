"""Tests for silentmesh: morphing must hide shape, unmorphed must not."""

import math
import random
import unittest

from silentmesh import (
    CoverModel,
    accuracy,
    audit,
    default_video_model,
    demorph,
    epsilon_estimate,
    generate_cover_traffic,
    morph_message,
    train_classifier,
    unmorphed_stream,
)


class TestMorphRoundTrip(unittest.TestCase):
    def test_roundtrip_preserves_message(self):
        rng = random.Random(1)
        model = default_video_model()
        for msg in (b"", b"hi", b"hello world" * 100, bytes(range(256)) * 40):
            packets = morph_message(model, msg, rng)
            self.assertEqual(demorph(packets), msg)

    def test_packet_sizes_stay_in_cover_range(self):
        rng = random.Random(2)
        model = default_video_model()
        packets = morph_message(model, b"x" * 20_000, rng)
        for p in packets:
            self.assertGreaterEqual(p.size, 64)
            self.assertLessEqual(p.size, 9000)
            self.assertLessEqual(p.payload_len, p.size)


class TestCoverModel(unittest.TestCase):
    def test_from_observations_recovers_parameters(self):
        rng = random.Random(3)
        true = CoverModel(size_mu=math.log(1200), size_sigma=0.6, gap_mean_ms=12.0)
        sizes = [true.sample_size(rng) for _ in range(5000)]
        gaps = [true.sample_gap(rng) for _ in range(5000)]
        fitted = CoverModel.from_observations(sizes, gaps)
        self.assertAlmostEqual(fitted.size_mu, math.log(1200), delta=0.1)
        self.assertAlmostEqual(fitted.size_sigma, 0.6, delta=0.1)
        self.assertAlmostEqual(fitted.gap_mean_ms, 12.0, delta=1.0)


class TestLeakageAudit(unittest.TestCase):
    def test_morphed_traffic_near_chance(self):
        result = audit(n_streams=60, seed=0)
        self.assertLess(result["morphed_accuracy"], 0.62, result)

    def test_unmorphed_traffic_easily_classified(self):
        result = audit(n_streams=60, seed=0)
        self.assertGreater(result["unmorphed_accuracy"], 0.85, result)

    def test_epsilon_estimates_are_numeric_and_ordered(self):
        result = audit(n_streams=60, seed=0)
        for key in ("morphed_epsilon_hat", "unmorphed_epsilon_hat"):
            self.assertTrue(math.isfinite(result[key]))
            self.assertGreaterEqual(result[key], 0.0)
        self.assertLess(result["morphed_epsilon_hat"], 1.0, result)
        self.assertGreater(result["unmorphed_epsilon_hat"], 2.0, result)
        self.assertLess(
            result["morphed_epsilon_hat"], result["unmorphed_epsilon_hat"], result
        )

    def test_classifier_beats_chance_on_unmorphed_only(self):
        rng = random.Random(4)
        model = default_video_model()
        cover = [generate_cover_traffic(model, rng.randrange(8, 16), rng) for _ in range(40)]
        morphed = [morph_message(model, bytes(14_000), rng) for _ in range(40)]
        plain = [unmorphed_stream(bytes(14_000)) for _ in range(40)]
        clf_m = train_classifier(cover[:20], morphed[:20])
        clf_p = train_classifier(cover[:20], plain[:20])
        self.assertLess(accuracy(clf_m, cover[20:], morphed[20:]), 0.62)
        self.assertGreater(accuracy(clf_p, cover[20:], plain[20:]), 0.85)


if __name__ == "__main__":
    unittest.main()
