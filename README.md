# silentmesh

**Traffic-shape morphing against traffic analysis.**

Tor hides content, not patterns. Timing and size analysis can still
deanonymize users. silentmesh morphs messages to match the statistical
distribution of chosen cover traffic — and measures, honestly, how
much an observer can still learn from shape alone.

## The novel idea

Most traffic-obfuscation tools pad to a fixed size or add random
delays. silentmesh instead *fits* a cover distribution (packet sizes
and inter-packet gaps) and morphs every message into that distribution,
then audits the result with a classifier and an empirical
differential-privacy-style epsilon. The morph and the audit are the
same package: you do not just claim indistinguishability, you measure
it.

## What works today (honest status)

- A cover-traffic model: lognormal packet sizes + exponential gaps,
  fitted from observations (`CoverModel.from_observations`), with an
  illustrative video-streaming default.
- A morpher: chunks, pads, and delays messages to cover shape, with
  interleaved chaff packets; `demorph` reassembles the message
  exactly.
- A leakage auditor: shape features (size/gap mean and std), a
  nearest-centroid classifier, held-out accuracy, and an empirical
  epsilon estimate.
- **Measured result (seed 0, 60 streams per class): morphed traffic
  is classified at near-chance accuracy, while unmorphed traffic is
  classified at ~100%.** Run it yourself: `python -m silentmesh`.

What this is *not*: a proof of anonymity. The epsilon is an
*empirical lower-bound estimate* against one weak classifier, not a
formal DP guarantee. Real cover parameters must be fitted from real
captures (the default is synthetic), and a production system needs
the full analysis in `ARCHITECTURE.md` — stronger adversaries,
multi-feature leakage, and real network transport.

## Quickstart

Requires Python 3.10+ and nothing else (standard library only).

```bash
# Runnable demo (morph + round-trip + audit)
python demo.py

# Leakage audit
python -m silentmesh
python -m silentmesh --streams 100 --seed 7

# Tests
python -m unittest discover -s tests -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Core modules:

- `silentmesh/cover.py` — cover-traffic model and packet model.
- `silentmesh/morph.py` — morpher (pad/chunk/delay/chaff) and demorph.
- `silentmesh/audit.py` — shape features, centroid classifier,
  accuracy, and empirical epsilon estimation.

## Evaluation plan

1. **Classifier accuracy**: morphed vs. cover should stay near 0.5
   (chance) across seeds; unmorphed vs. cover should stay near 1.0.
   Both are asserted in the test suite.
2. **Epsilon**: morphed empirical epsilon should stay small (< 1.0)
   and well below the unmorphed baseline (> 2.0).
3. **Stronger adversaries** (future): gradient-boosted trees,
   sequence models over raw packet traces, and multi-session
   correlation — the audit must survive all of them, not just the
   centroid classifier.
4. **Real cover data**: fit the model from real captures and re-run
   the audit; report any accuracy change.
