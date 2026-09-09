#!/usr/bin/env python3
"""List the command-line scripts provided by fd2bec."""

# Tested by pytest: tests/test_fd2bec_help.py

import argparse
import ast
import sys
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fd2bec.show import print_scripts

DESCRIPTION = "Search for scripts and their descriptions in 'fd2bec'."
CLI_ROOT = Path(__file__).resolve().parent


def parse_script(path: Path) -> Optional[ast.Module]:
    """Parse *path* without importing or executing it."""
    try:
        with tokenize.open(str(path)) as source:
            return ast.parse(source.read(), filename=str(path))
    except (OSError, SyntaxError) as error:
        print(f"Warning: could not inspect {path}: {error}", file=sys.stderr)
        return None


def has_main_function(tree: ast.Module) -> bool:
    """Return whether the syntax tree defines a function named ``main``."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        for node in ast.walk(tree)
    )


def read_description(tree: ast.Module) -> Optional[str]:
    """Read the first line of a module-level ``description`` assignment."""
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "description" for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "description"
        ):
            value = node.value

        if value is not None:
            try:
                description = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                if isinstance(value, ast.JoinedStr):
                    description = "".join(
                        part.value
                        for part in value.values
                        if isinstance(part, ast.Constant) and isinstance(part.value, str)
                    )
                else:
                    return None

            if isinstance(description, str):
                lines = [line.strip() for line in description.splitlines() if line.strip()]
                return lines[0] if lines else None
            return None
    return None


def available_folders(cli_root: Path = CLI_ROOT) -> List[str]:
    """Return the CLI groups that contain at least one Python file."""
    folders = {
        path.relative_to(cli_root).parts[0]
        for path in cli_root.rglob("*.py")
        if len(path.relative_to(cli_root).parts) > 1
        and path.name != "__init__.py"
        and "__pycache__" not in path.parts
    }
    return sorted(folders, key=str.casefold)


def find_scripts(
    folders: Optional[Iterable[str]] = None, cli_root: Path = CLI_ROOT
) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Find CLI modules with a ``main`` function, grouped by folder."""
    selected = set(folders) if folders else None
    scripts: Dict[str, List[Tuple[str, Optional[str]]]] = {}

    for path in sorted(cli_root.rglob("*.py")):
        relative = path.relative_to(cli_root)
        if path.name == "__init__.py" or "__pycache__" in relative.parts:
            continue

        folder = relative.parts[0] if len(relative.parts) > 1 else "root"
        if selected is not None and folder not in selected:
            continue

        tree = parse_script(path)
        if tree is None or not has_main_function(tree):
            continue

        scripts.setdefault(folder, []).append((path.name, read_description(tree)))

    return {
        folder: sorted(entries, key=lambda entry: entry[0].casefold())
        for folder, entries in sorted(scripts.items(), key=lambda item: item[0].casefold())
    }


def prepare_parser() -> argparse.ArgumentParser:
    folders = available_folders()
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-f",
        "--folders",
        nargs="+",
        choices=folders,
        metavar="FOLDER",
        help="folders to search (available: %(choices)s)",
    )
    parser.add_argument(
        "-s",
        "--show-folders",
        action="store_true",
        help="show only folder names",
    )
    parser.add_argument(
        "-d",
        "--descriptions",
        action="store_true",
        help="show each script's description",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable colored output",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = prepare_parser()
    args = parser.parse_args(argv)

    scripts = find_scripts(args.folders)
    print(f"\n\tLooking for scripts in '{CLI_ROOT}'\n")
    print_scripts(
        scripts,
        show_folders=args.show_folders,
        descriptions=args.descriptions,
        color=not args.no_color and sys.stdout.isatty(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
