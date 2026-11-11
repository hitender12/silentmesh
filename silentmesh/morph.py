"""Morpher: shape a message to match the cover distribution.

A message is split into chunks, each chunk is padded up to a
cover-sampled packet size, and packets are spaced with cover-sampled
gaps. Optional chaff packets (no payload) are interleaved so the
stream's packet count and burstiness also look like cover traffic.

The receiver runs :func:`demorph`, which concatenates the payloads in
order. Padding and chaff are discarded.
"""

from __future__ import annotations

import random

from .cover import HEADER_BYTES, CoverModel, Packet


def morph_message(
    model: CoverModel,
    message: bytes,
    rng: random.Random,
    chaff_prob: float = 0.15,
) -> list[Packet]:
    """Morph ``message`` into a cover-shaped packet stream."""
    packets: list[Packet] = []
    offset = 0
    first = True
    while offset < len(message):
        size = model.sample_size(rng)
        capacity = max(1, size - HEADER_BYTES)
        chunk = message[offset : offset + capacity]
        offset += len(chunk)
        packets.append(
            Packet(
                size=size,
                gap_ms=0.0 if first else model.sample_gap(rng),
                payload=chunk,
            )
        )
        first = False
        if rng.random() < chaff_prob:
            packets.append(
                Packet(
                    size=model.sample_size(rng),
                    gap_ms=model.sample_gap(rng),
                    payload=b"",
                )
            )
    return packets


def demorph(packets: list[Packet]) -> bytes:
    """Reassemble the original message from a morphed stream."""
    return b"".join(p.payload for p in packets)


def unmorphed_stream(message: bytes, chunk: int = 256, gap_ms: float = 1.0) -> list[Packet]:
    """A naive stream: fixed-size chunks, fixed gaps. Trivially classifiable.

    This is the baseline the auditor compares against: traffic that makes
    no attempt to hide its shape.
    """
    packets: list[Packet] = []
    offset = 0
    while offset < len(message):
        piece = message[offset : offset + chunk]
        offset += len(piece)
        packets.append(
            Packet(size=len(piece), gap_ms=0.0 if not packets else gap_ms, payload=piece)
        )
    return packets
