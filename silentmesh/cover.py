"""Cover-traffic model: fitted size/timing distributions.

The model describes what an observer sees on the wire for a chosen
cover application (default: video streaming over QUIC).  Packet sizes
are lognormal, inter-packet gaps are exponential.  Parameters are fitted
from observations with ``from_observations``; ``default_video_model``
uses illustrative parameters for synthetic cover data, documented as
such in the README.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MIN_SIZE = 64
MAX_SIZE = 9000
HEADER_BYTES = 16  # reserved per packet for framing / encrypted header


@dataclass
class Packet:
    """One packet as seen on the wire.

    ``size`` is the wire size. ``payload`` is the real data carried
    (empty for chaff); in production it would ride encrypted inside the
    padded packet, so an observer sees only ``size`` and ``gap_ms``.
    ``gap_ms`` is the delay before this packet was sent.
    """

    size: int
    gap_ms: float
    payload: bytes = b""

    @property
    def payload_len(self) -> int:
        return len(self.payload)


@dataclass
class CoverModel:
    """Lognormal packet sizes + exponential inter-packet gaps."""

    size_mu: float  # mean of log(size)
    size_sigma: float  # std of log(size)
    gap_mean_ms: float
    label: str = "cover"

    @classmethod
    def from_observations(cls, sizes, gaps, label: str = "cover") -> "CoverModel":
        """Fit lognormal size params and exponential gap mean from samples."""
        if not sizes or not gaps:
            raise ValueError("need non-empty size and gap observations")
        logs = [math.log(max(1, s)) for s in sizes]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / len(logs)
        return cls(
            size_mu=mu,
            size_sigma=math.sqrt(var) or 0.1,
            gap_mean_ms=sum(gaps) / len(gaps),
            label=label,
        )

    def sample_size(self, rng: random.Random) -> int:
        size = int(round(rng.lognormvariate(self.size_mu, self.size_sigma)))
        return max(MIN_SIZE, min(MAX_SIZE, size))

    def sample_gap(self, rng: random.Random) -> float:
        return rng.expovariate(1.0 / self.gap_mean_ms)

    def sample_stream(self, n_packets: int, rng: random.Random) -> list[Packet]:
        """A cover stream: no payload, sizes/gaps drawn from the model."""
        packets: list[Packet] = []
        for i in range(n_packets):
            packets.append(
                Packet(
                    size=self.sample_size(rng),
                    gap_ms=0.0 if i == 0 else self.sample_gap(rng),
                )
            )
        return packets


def default_video_model() -> CoverModel:
    """Illustrative video-streaming cover model (synthetic parameters).

    Median packet size ~1200 bytes (lognormal, sigma 0.6), mean gap 12 ms.
    These numbers are plausible for chunked video over QUIC but are NOT
    fitted from real traffic captures; fit your own with
    ``CoverModel.from_observations`` for production claims.
    """
    return CoverModel(size_mu=math.log(1200), size_sigma=0.6, gap_mean_ms=12.0, label="video")


def generate_cover_traffic(model: CoverModel, n_packets: int, rng: random.Random) -> list[Packet]:
    return model.sample_stream(n_packets, rng)
