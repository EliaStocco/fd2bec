"""Check the committed MACE-POLAR Born-charge regression."""

import argparse
from pathlib import Path

import numpy as np
from fd2bec.io import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bec", required=True)
    parser.add_argument(
        "--reference",
        default=Path(__file__).with_name("bec-reference.txt"),
        type=Path,
    )
    args = parser.parse_args()

    structures = read(args.dataset, index=":")
    if len(structures) != 19:
        raise AssertionError(f"Expected 19 finite-displacement structures, got {len(structures)}")
    dipoles = np.asarray([atoms.info["REF_dipole"] for atoms in structures])
    if not np.all(np.isfinite(dipoles)):
        raise AssertionError("MACE-POLAR produced non-finite dipoles")

    actual = np.loadtxt(args.bec).reshape(-1, 3, 3)
    expected = np.loadtxt(args.reference).reshape(-1, 3, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    print(f"MACE-POLAR regression passed; maximum error: {np.max(np.abs(actual - expected)):.3e}")


if __name__ == "__main__":
    main()
