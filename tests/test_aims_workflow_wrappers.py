from argparse import Namespace
from pathlib import Path

from fd2bec.cli.aims.post_process_aims import postprocess_commands
from fd2bec.cli.aims.prepare_aims import preparation_commands


def test_preparation_commands_use_unified_displacement_workflow():
    args = Namespace(
        input="reference.extxyz",
        what="piezo",
        amplitude=0.002,
        no_symmetry=False,
        number=4,
        seed=17,
        displacements_output="displacements.txt",
        structures_output="displaced.extxyz",
        output="geometries",
    )

    generate, export = preparation_commands(args)

    assert "fd2bec.cli.displacements.generate_displacements" in generate
    assert generate[generate.index("-w") + 1] == "piezo"
    assert generate[-4:] == ["--number", "4", "--seed", "17"]
    assert "fd2bec.cli.displacements.extxyz2folder" in export
    assert export[-4:] == ["-f", "aims", "-o", "geometries"]


def test_postprocess_commands_share_configured_paths():
    args = Namespace(
        input="reference.extxyz",
        results="aims-results",
        format="aims_polarization",
        dataset="work/dataset.extxyz",
        output="work/bec",
    )

    build, fit, charges = postprocess_commands(args)

    assert "fd2bec.cli.dPdR.build_dataset4dPdR" in build
    assert build[-2:] == ["-o", "work/dataset.extxyz"]
    assert "fd2bec.cli.dPdR.dPdR2bec" in fit
    assert fit[-2:] == ["-o", "work/bec"]
    assert "fd2bec.cli.general.bec2charges" in charges
    assert charges[-4:] == [
        "-i",
        str(Path("work/bec/bec.txt")),
        "-o",
        str(Path("work/bec/charges.txt")),
    ]
