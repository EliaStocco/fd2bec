import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "generate_code_map.py"
SPEC = importlib.util.spec_from_file_location("generate_code_map", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
code_map = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(code_map)


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_package_tree_lists_python_sources_only(tmp_path):
    package = tmp_path / "fd2bec"
    write(package / "__init__.py")
    write(package / "core.py")
    write(package / "notes.txt")
    write(package / "nested" / "module.py")
    write(package / "empty" / "notes.txt")

    tree = code_map.package_tree(package)

    assert tree.splitlines() == [
        "fd2bec/",
        "|-- nested/",
        "|   `-- module.py",
        "|-- __init__.py",
        "`-- core.py",
    ]
    assert "notes.txt" not in tree
    assert "empty" not in tree


def test_new_cli_script_is_automatically_marked_undocumented(tmp_path):
    package = tmp_path / "fd2bec"
    write(package / "__init__.py")
    write(package / "cli" / "__init__.py")
    write(package / "cli" / "new_script.py", "def main():\n    pass\n")
    write(
        tmp_path / "pyproject.toml",
        '[project.scripts]\nnew-command = "fd2bec.cli.new_script:main"\n',
    )
    generated = tmp_path / "docs" / "source" / "generated"
    generated.mkdir(parents=True)
    modules, _ = code_map.source_index(package)

    rst = code_map.package_structure_rst(tmp_path, package, modules, {}, generated)

    assert "new-command" in rst
    assert "fd2bec.cli.new_script" in rst
    assert "**No explicit documentation found**" in rst

    write(
        tmp_path / "docs" / "source" / "new_script.rst",
        "Run ``new-command`` to create the new data.\n",
    )
    documented_rst = code_map.package_structure_rst(tmp_path, package, modules, {}, generated)

    assert "Yes:" in documented_rst
    assert "docs/source/new_script.rst" in documented_rst


def test_pytest_map_links_direct_function_class_and_method_uses(tmp_path):
    package = tmp_path / "fd2bec"
    write(package / "__init__.py")
    write(
        package / "alpha.py",
        "def used():\n    pass\n\n"
        "def unused():\n    pass\n\n"
        "class Thing:\n"
        "    @classmethod\n"
        "    def method(cls):\n"
        "        pass\n",
    )
    tests = tmp_path / "tests"
    write(
        tests / "test_alpha.py",
        "from fd2bec.alpha import Thing, used\n\n"
        "def test_used():\n    used()\n\n"
        "def test_constructor():\n    Thing()\n\n"
        "def test_method():\n    Thing.method()\n",
    )
    _, symbols = code_map.source_index(package)

    related = code_map.pytest_symbol_map(tests, symbols)

    assert related["fd2bec.alpha.used"] == {"tests/test_alpha.py::test_used"}
    assert related["fd2bec.alpha.Thing"] == {"tests/test_alpha.py::test_constructor"}
    assert related["fd2bec.alpha.Thing.method"] == {"tests/test_alpha.py::test_method"}
    assert "fd2bec.alpha.unused" not in related


def test_dependencies_are_collapsed_to_readable_subsystems(tmp_path):
    package = tmp_path / "fd2bec"
    write(package / "__init__.py")
    write(package / "tensor.py")
    write(package / "core.py", "import fd2bec.tensor\n")
    write(package / "cli" / "__init__.py")
    write(package / "cli" / "demo" / "__init__.py")
    write(
        package / "cli" / "demo" / "runner.py",
        "from fd2bec.core import helper\n",
    )
    modules, _ = code_map.source_index(package)

    direct = code_map.module_dependencies(modules)
    collapsed = code_map.subsystem_dependencies(direct, {"demo"})

    assert collapsed["core"] == {"tensor"}
    assert collapsed["cli.demo"] == {"core"}
    dot = code_map.dependency_dot(collapsed)
    assert '"cli.demo" -> "core";' in dot
    assert '"core" -> "tensor";' in dot
