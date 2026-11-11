"""CLI: run the leakage audit and print the result.

Usage:
    python -m silentmesh
    python -m silentmesh --streams 100 --seed 7
"""

from __future__ import annotations

import argparse

from .audit import audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="silentmesh")
    parser.add_argument("--streams", type=int, default=60)
    parser.add_argument("--message-len", type=int, default=14_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = audit(n_streams=args.streams, message_len=args.message_len, seed=args.seed)
    print("silentmesh leakage audit")
    print("=" * 60)
    print(f"streams per class: {result['n_streams_per_class']} (half train, half test)")
    print()
    print("morphed traffic vs cover:")
    print(f"  classifier accuracy : {result['morphed_accuracy']:.3f}  (0.5 = chance)")
    print(f"  distinguishing adv.: {result['morphed_advantage']:.3f}")
    print(f"  empirical epsilon   : {result['morphed_epsilon_hat']:.3f}")
    print()
    print("unmorphed traffic vs cover (baseline):")
    print(f"  classifier accuracy : {result['unmorphed_accuracy']:.3f}")
    print(f"  distinguishing adv.: {result['unmorphed_advantage']:.3f}")
    print(f"  empirical epsilon   : {result['unmorphed_epsilon_hat']:.3f}")
    print()
    if result["morphed_accuracy"] < 0.62:
        print("PASS: morphed traffic is near chance; the shape morph works.")
    else:
        print("FAIL: morphed traffic is distinguishable; the morph needs work.")
    return 0 if result["morphed_accuracy"] < 0.62 else 1


if __name__ == "__main__":
    raise SystemExit(main())
