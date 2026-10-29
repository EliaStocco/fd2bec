"""Generate parent-symmetry-inequivalent pairs of periodic structures."""

import argparse
from pathlib import Path

from fd2bec.cli import cli, read_input_structures
from fd2bec.cli.parser import add_shared_argument
from fd2bec.domain_variants import canonicalize_lattice_orientation, inequivalent_structure_pairs
from fd2bec.io import write

description = (
    "Find pairs of parent-symmetry-related domain variants that are inequivalent under "
    "simultaneous parent operations. The output extxyz stores each representative pair "
    "as two consecutive, metadata-labelled frames."
)


def prepare_args(descr: str) -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=descr)
    add_shared_argument(parser, "input_structure")
    add_shared_argument(parser, "symprec")
    parser.add_argument(
        "-j",
        "--second",
        metavar="\b",
        required=True,
        help="path to the second multi-frame input structure file",
    )
    parser.add_argument(
        "-p",
        "--parent",
        metavar="\b",
        required=True,
        help="parent-phase reference structure defining the symmetry and coordinate axes",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="\b",
        required=True,
        help="output extxyz containing two consecutive frames for every representative pair",
    )
    parser.add_argument(
        "--include-identical",
        action="store_true",
        help="include trivial pairs with identical endpoint frames when both inputs are the same file",
    )
    return parser


def _pair_metadata(pair_index: int, endpoint: str, source_index: int, orbit_size: int) -> dict:
    """Return extxyz metadata identifying one endpoint of an output pair."""
    return {
        "fd2bec_pair_index": pair_index,
        "fd2bec_pair_endpoint": endpoint,
        "fd2bec_source_frame": source_index + 1,
        "fd2bec_equivalent_pair_count": orbit_size,
    }


@cli(prepare_args, description)
def main(args: argparse.Namespace) -> None:
    """Read domain sets, classify their pair orbits, and write representatives."""
    first = read_input_structures(args.input, index=":", label="first domain variants")
    second = read_input_structures(args.second, index=":", label="second domain variants")
    parent = read_input_structures(args.parent, label="parent reference structure")
    same_input = Path(args.input).resolve() == Path(args.second).resolve()
    pairs = inequivalent_structure_pairs(
        first,
        second,
        parent,
        symprec=args.symprec,
        treat_reversed_as_equivalent=same_input,
        include_identical=args.include_identical,
    )

    raw_pair_count = len(first) * len(second)
    if same_input:
        raw_pair_count = len(first) * (len(first) - 1) // 2
        if args.include_identical:
            raw_pair_count += len(first)
    print(
        f"Found {len(pairs)} inequivalent pairs from {raw_pair_count} candidate "
        f"pairs ({len(first)} x {len(second)} input variants)."
    )
    print("Representative pairs (one-based input-frame indices):")
    for pair_index, pair in enumerate(pairs, start=1):
        members = ", ".join(
            f"({first_index + 1}, {second_index + 1})"
            for first_index, second_index in pair.equivalent_pairs
        )
        print(
            f"  {pair_index}: ({pair.first_index + 1}, {pair.second_index + 1}) "
            f"[{len(pair.equivalent_pairs)} equivalent: {members}]"
        )

    output = []
    canonical_first = [canonicalize_lattice_orientation(atoms, parent) for atoms in first]
    canonical_second = [canonicalize_lattice_orientation(atoms, parent) for atoms in second]
    for pair_index, pair in enumerate(pairs, start=1):
        first_endpoint = canonical_first[pair.first_index].copy()
        second_endpoint = canonical_second[pair.second_index].copy()
        first_endpoint.info.update(
            _pair_metadata(pair_index, "first", pair.first_index, len(pair.equivalent_pairs))
        )
        second_endpoint.info.update(
            _pair_metadata(pair_index, "second", pair.second_index, len(pair.equivalent_pairs))
        )
        output.extend((first_endpoint, second_endpoint))

    print(f"Writing {len(pairs)} representative pairs to {args.output} ... ", end="")
    write(args.output, output)
    print("done")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
