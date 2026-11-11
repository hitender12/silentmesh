# ARCHITECTURE.md — silentmesh

## Design

```
message bytes
     │
     ▼
┌─────────────┐   cover model    ┌──────────────┐
│   morpher   │ ◀── (lognormal   │  cover model │
│ pad/chunk/  │    sizes, exp.   │  fitted from │
│ delay/chaff │    gaps)         │  observations│
└─────────────┘                  └──────────────┘
     │  packet stream (size, gap, payload)
     ▼
┌─────────────┐
│   demorph   │ ──▶ original message (exact round-trip)
└─────────────┘

     │  same packet stream, plus real cover streams
     ▼
┌─────────────┐
│   auditor   │ ──▶ accuracy, advantage, empirical epsilon
│ (centroid   │
│  classifier)│
└─────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `cover.py` | `CoverModel` (lognormal sizes, exponential gaps), `Packet` (wire size, gap, payload), `from_observations` fitting, `default_video_model`. |
| `morph.py` | `morph_message`: split into cover-sized chunks, pad to sampled sizes, space with sampled gaps, interleave chaff. `demorph`: exact reassembly. `unmorphed_stream`: fixed-size/fixed-gap baseline. |
| `audit.py` | `stream_features` (size/gap mean+std — only what an observer sees), `CentroidClassifier`, `accuracy`, `epsilon_estimate`, `audit` (train/test split). |

## Key decisions

- **Features are observer-visible only.** The classifier sees packet
  sizes and gaps, never payloads or payload lengths. That is the
  whole point of the audit.
- **Nearest-centroid, not a strong model.** The auditor is
  deliberately weak: if even a weak classifier cannot beat chance,
  the morph is doing real work. Stronger adversaries belong in the
  evaluation plan, not hidden.
- **Empirical epsilon is a lower bound, labelled as such.** It is
  derived from the best threshold test on the classifier score:
  `eps = max_t ln((TPR + d) / (FPR + d))`. It proves an observer
  *at least this good* exists. It does not prove no better observer
  exists. The README says so plainly.
- **Chaff is interleaved, not appended.** Appended chaff would be
  trivially separable (tail of the stream); interleaving keeps the
  whole stream cover-shaped.
- **Deterministic seeds.** Every audit is reproducible from its seed.

## Threat model (honest)

- **Adversary:** passive observer of packet sizes and timings on one
  link. Not an active attacker, not a global observer, not a
  timing-correlation attacker across multiple links.
- **Cover assumption:** the observer cannot distinguish morphed
  packets from cover packets *except* by shape. Content is assumed
  encrypted; payload lengths are hidden by padding (in production,
  payload length lives in an encrypted header — here the simulator
  keeps it alongside the packet for reassembly testing).
- **Out of scope:** real network transport, cover-model drift,
  active probing, and multi-session linkage.

## Roadmap

1. Stronger adversaries: gradient boosting, sequence models, and
   cross-session correlation in the auditor.
2. Formal DP analysis: per-packet privacy accounting for the morph
   (composition over a session), replacing the empirical epsilon.
3. Real transport: QUIC-based sender/receiver with the morpher inline.
4. Cover models fitted from real captures (video, VoIP, web browsing)
   and per-cover audit reports.
