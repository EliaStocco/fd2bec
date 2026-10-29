"""Check finite-displacement preparation folders and ZIP archives."""

import argparse
from pathlib import Path

from fd2bec.preparation_audit import audit_path, find_preparation_paths

description = "Check files and BEC-rank sufficiency in prepared finite-displacement calculations."


def prepare_args(descr):
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("paths", nargs="+", metavar="PATH", help="folder or ZIP archive to inspect")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also scan extxyz frames and displacement-table rows (slower for large archives)",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="audit cached symmetry-reduced BEC fit rank when cache data is available",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip this many discovered bundles (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="check at most this many discovered bundles",
    )
    return parser


def main(arguments=None):
    """Run the read-only preparation audit and return a shell-compatible status."""
    parser = prepare_args(description)
    args = parser.parse_args(arguments)
    requested = [Path(path) for path in args.paths]
    if args.offset < 0 or args.limit is not None and args.limit <= 0:
        parser.error("--offset must be non-negative and --limit must be positive")
    missing = [path for path in requested if not path.exists()]
    for path in missing:
        print("ERROR {}: path does not exist".format(path))
    candidates = find_preparation_paths([path for path in requested if path.exists()])
    if not candidates:
        print("ERROR: no preparation folders or structure-*.zip archives found")
        return 2
    candidates = candidates[args.offset :]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("ERROR: no preparation bundles remain after --offset/--limit")
        return 2

    failures = 0
    for path in candidates:
        audit = audit_path(path, deep=args.deep, check_rank=args.rank)
        status = "PASS" if audit.passed else "FAIL"
        print("{} {}".format(status, path))
        for detail in audit.details:
            print("  {}".format(detail))
        for warning in audit.warnings:
            print("  WARNING: {}".format(warning))
        for error in audit.errors:
            print("  ERROR: {}".format(error))
        failures += not audit.passed
    print("Checked {} bundle(s): {} failed.".format(len(candidates), failures + len(missing)))
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
