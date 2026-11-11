"""Leakage auditor: how well can an observer tell morphed traffic apart
from real cover traffic?

Method (honest, empirical, not a proof):

* Each stream is reduced to shape features an observer can see: mean
  and std of packet sizes, mean and std of inter-packet gaps.
* A nearest-centroid classifier is trained on labelled cover vs. other
  streams and evaluated on held-out streams.
* Accuracy near 0.5 (chance) means the morph works; accuracy near 1.0
  means the shape leaks.
* The empirical epsilon is a *lower-bound estimate* derived from the
  best threshold test on the classifier score:
  eps_hat = max over thresholds of ln((TPR + d) / (FPR + d)), d = 0.01.
  It says "an observer *at least this good* exists", not "no observer
  can do better". A small eps_hat with a weak classifier is evidence,
  not a guarantee; a real deployment needs the DP analysis in
  ARCHITECTURE.md.
"""

from __future__ import annotations

import math
import random
import statistics

from .cover import CoverModel, Packet, default_video_model, generate_cover_traffic
from .morph import morph_message, unmorphed_stream

SMOOTHING = 0.01


def stream_features(packets: list[Packet]) -> list[float]:
    sizes = [p.size for p in packets]
    gaps = [p.gap_ms for p in packets[1:]] or [0.0]
    return [
        statistics.fmean(sizes),
        statistics.pstdev(sizes) if len(sizes) > 1 else 0.0,
        statistics.fmean(gaps),
        statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
    ]


def _standardize(features: list[list[float]]):
    mu = [statistics.fmean(col) for col in zip(*features)]
    sigma = []
    for col in zip(*features):
        sd = statistics.pstdev(col)
        sigma.append(sd if sd > 1e-9 else 1.0)
    return mu, sigma


def _apply_standardize(vec, mu, sigma):
    return [(v - m) / s for v, m, s in zip(vec, mu, sigma)]


class CentroidClassifier:
    """Nearest-centroid classifier on standardized shape features."""

    def __init__(self, mu, sigma, centroid_cover, centroid_other):
        self.mu = mu
        self.sigma = sigma
        self.centroid_cover = centroid_cover
        self.centroid_other = centroid_other

    def score(self, packets: list[Packet]) -> float:
        """Positive => looks like 'other', negative => looks like cover."""
        z = _apply_standardize(stream_features(packets), self.mu, self.sigma)
        d_cover = sum((a - b) ** 2 for a, b in zip(z, self.centroid_cover))
        d_other = sum((a - b) ** 2 for a, b in zip(z, self.centroid_other))
        return d_cover - d_other

    def predict(self, packets: list[Packet]) -> int:
        return 1 if self.score(packets) > 0 else 0


def train_classifier(cover_streams, other_streams) -> CentroidClassifier:
    all_feats = [stream_features(s) for s in cover_streams + other_streams]
    mu, sigma = _standardize(all_feats)

    def centroid(streams):
        zs = [_apply_standardize(stream_features(s), mu, sigma) for s in streams]
        return [statistics.fmean(col) for col in zip(*zs)]

    return CentroidClassifier(mu, sigma, centroid(cover_streams), centroid(other_streams))


def accuracy(classifier, cover_streams, other_streams) -> float:
    correct = sum(1 for s in cover_streams if classifier.predict(s) == 0)
    correct += sum(1 for s in other_streams if classifier.predict(s) == 1)
    return correct / max(1, len(cover_streams) + len(other_streams))


def epsilon_estimate(classifier, cover_streams, other_streams) -> float:
    """Empirical epsilon lower-bound estimate from the classifier score.

    Sweeps thresholds over the score; at each threshold computes
    TPR (fraction of 'other' streams above threshold) and FPR
    (fraction of cover streams above threshold), and returns the
    largest ln((TPR + d) / (FPR + d)). Returns 0.0 when the classifier
    is at chance.
    """
    cover_scores = sorted(classifier.score(s) for s in cover_streams)
    other_scores = sorted(classifier.score(s) for s in other_streams)
    if not cover_scores or not other_scores:
        return 0.0
    thresholds = sorted(set(cover_scores + other_scores))
    best = 0.0
    for t in thresholds:
        tpr = sum(1 for s in other_scores if s > t) / len(other_scores)
        fpr = sum(1 for s in cover_scores if s > t) / len(cover_scores)
        eps = math.log((tpr + SMOOTHING) / (fpr + SMOOTHING))
        best = max(best, eps)
    return best


def audit(
    model: CoverModel | None = None,
    n_streams: int = 60,
    message_len: int = 14_000,
    seed: int = 0,
) -> dict:
    """Full audit: train on half the streams, evaluate on the other half.

    Returns accuracies and epsilon estimates for morphed and unmorphed
    traffic against cover traffic, plus stream counts.
    """
    model = model or default_video_model()
    rng = random.Random(seed)

    def make_cover():
        return generate_cover_traffic(model, rng.randrange(8, 16), rng)

    def make_morphed():
        msg = bytes(rng.randrange(256) for _ in range(message_len))
        return morph_message(model, msg, rng)

    def make_unmorphed():
        msg = bytes(rng.randrange(256) for _ in range(message_len))
        return unmorphed_stream(msg)

    cover = [make_cover() for _ in range(n_streams)]
    morphed = [make_morphed() for _ in range(n_streams)]
    unmorphed = [make_unmorphed() for _ in range(n_streams)]

    half = n_streams // 2
    clf_morph = train_classifier(cover[:half], morphed[:half])
    clf_plain = train_classifier(cover[:half], unmorphed[:half])
    acc_morph = accuracy(clf_morph, cover[half:], morphed[half:])
    acc_plain = accuracy(clf_plain, cover[half:], unmorphed[half:])
    return {
        "n_streams_per_class": n_streams,
        "morphed_accuracy": acc_morph,
        "morphed_advantage": 2 * abs(acc_morph - 0.5),
        "morphed_epsilon_hat": epsilon_estimate(clf_morph, cover[half:], morphed[half:]),
        "unmorphed_accuracy": acc_plain,
        "unmorphed_advantage": 2 * abs(acc_plain - 0.5),
        "unmorphed_epsilon_hat": epsilon_estimate(clf_plain, cover[half:], unmorphed[half:]),
    }
