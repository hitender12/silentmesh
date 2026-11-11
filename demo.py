"""Demo: morph a message, reassemble it, and audit the leakage."""

import random

from silentmesh import (
    audit,
    default_video_model,
    demorph,
    morph_message,
    unmorphed_stream,
)


def main() -> None:
    rng = random.Random(42)
    model = default_video_model()
    message = b"Meet at the usual place. Bring the documents." * 200  # ~9.4 KB

    morphed = morph_message(model, message, rng)
    print("silentmesh demo")
    print("=" * 60)
    print(f"message: {len(message)} bytes -> {len(morphed)} morphed packets")
    print(f"first packets (size, gap_ms, payload_len):")
    for p in morphed[:6]:
        print(f"  size={p.size:5d}  gap={p.gap_ms:6.2f} ms  payload={p.payload_len}")
    print(f"round-trip ok: {demorph(morphed) == message}")
    print(f"unmorphed would be {len(unmorphed_stream(message))} fixed 256-byte packets")
    print()
    result = audit(n_streams=60, seed=0)
    print("audit (60 streams per class, half train / half test):")
    print(f"  morphed   accuracy={result['morphed_accuracy']:.3f}  eps_hat={result['morphed_epsilon_hat']:.3f}")
    print(f"  unmorphed accuracy={result['unmorphed_accuracy']:.3f}  eps_hat={result['unmorphed_epsilon_hat']:.3f}")
    print("  (accuracy 0.5 = chance: the observer cannot tell morphed from cover)")


if __name__ == "__main__":
    main()
