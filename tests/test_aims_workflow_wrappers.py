from argparse import Namespace
from pathlib import Path

from fd2bec import SYMPREC
from fd2bec.cli.aims import post_process_aims, prepare_aims
from fd2bec.cli.aims.post_process_aims import postprocess_commands, postprocess_paths
from fd2bec.cli.aims.prepare_aims import preparation_commands, preparation_paths


def test_aims_workflows_accept_the_combined_response_option():
    preparation = prepare_aims.prepare_args(prepare_aims.description).parse_args(
        ["-i", "reference.extxyz", "--what", "both"]
    )
    postprocessing = post_process_aims.prepare_args(post_process_aims.description).parse_args(
        ["-i", "reference.extxyz", "--what", "both"]
    )

    assert preparation.what == "both"
    assert postprocessing.what == "both"


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
        symprec=SYMPREC,
    )

    generate, export = preparation_commands(args)

    assert "fd2bec.cli.displacements.generate_displacements" in generate
    assert generate[generate.index("-w") + 1] == "piezo"
    assert generate[generate.index("-sp") + 1] == str(SYMPREC)
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
        symprec=SYMPREC,
    )

    build, fit, charges = postprocess_commands(args)

    assert "fd2bec.cli.dPdR.build_dataset4dPdR" in build
    assert build[-2:] == ["-o", "work/dataset.extxyz"]
    assert "fd2bec.cli.dPdR.dPdR2bec" in fit
    assert fit[fit.index("-sp") + 1] == str(SYMPREC)
    assert fit[-2:] == ["-o", "work/bec"]
    assert "fd2bec.cli.general.bec2charges" in charges
    assert charges[-4:] == [
        "-i",
        str(Path("work/bec/bec.txt")),
        "-o",
        str(Path("work/bec/charges.txt")),
    ]


def test_postprocess_commands_support_piezoelectric_workflow():
    args = Namespace(
        input="reference.extxyz",
        what="piezo",
        results="aims-results",
        pattern="aims.n=*.out",
        format="aims_polarization",
        dataset="work/piezo.extxyz",
        output="work/piezoelectric",
        symprec=SYMPREC,
    )

    build, fit = postprocess_commands(args)

    assert "fd2bec.cli.dPdS.build_dataset4dPdS_aims" in build
    assert build[-6:] == [
        "-i",
        "aims-results",
        "--pattern",
        "aims.n=*.out",
        "-o",
        "work/piezo.extxyz",
    ]
    assert "fd2bec.cli.dPdS.dPdS2piezo" in fit
    assert fit[-8:] == [
        "-i",
        "work/piezo.extxyz",
        "-r",
        "reference.extxyz",
        "-sp",
        str(SYMPREC),
        "-o",
        "work/piezoelectric",
    ]


def test_preparation_commands_keep_both_response_workflows_separate():
    args = Namespace(
        input="reference.extxyz",
        what="both",
        amplitude=0.002,
        no_symmetry=False,
        number=None,
        seed=None,
        displacements_output="displacements.txt",
        structures_output="displaced-structures.extxyz",
        output="geometries",
        log="fd2bec-log.txt",
        symprec=SYMPREC,
    )

    bec_paths = preparation_paths(args, "bec")
    piezo_paths = preparation_paths(args, "piezo")
    bec_generate, bec_export = preparation_commands(args, "bec")
    piezo_generate, piezo_export = preparation_commands(args, "piezo")

    assert bec_paths == (
        Path("displaced-structures.bec.extxyz"),
        Path("displacements.bec.txt"),
        Path("geometries/bec"),
        Path("fd2bec-log.bec.txt"),
    )
    assert piezo_paths == (
        Path("displaced-structures.piezo.extxyz"),
        Path("displacements.piezo.txt"),
        Path("geometries/piezo"),
        Path("fd2bec-log.piezo.txt"),
    )
    assert bec_generate[bec_generate.index("-w") + 1] == "bec"
    assert piezo_generate[piezo_generate.index("-w") + 1] == "piezo"
    assert bec_export[-1] == "geometries/bec"
    assert piezo_export[-1] == "geometries/piezo"


def test_postprocess_commands_keep_both_response_workflows_separate():
    args = Namespace(
        input="reference.extxyz",
        what="both",
        results="results",
        pattern="aims.n=*.out",
        format="aims_polarization",
        dataset="dataset.extxyz",
        output=".",
        log="fd2bec-log.pp.txt",
        symprec=SYMPREC,
    )

    bec_paths = postprocess_paths(args, "bec")
    piezo_paths = postprocess_paths(args, "piezo")
    bec_build, _, _ = postprocess_commands(args, "bec")
    piezo_build, _ = postprocess_commands(args, "piezo")

    assert bec_paths == (
        Path("results/bec"),
        Path("dataset.bec.extxyz"),
        Path("bec"),
        Path("fd2bec-log.pp.bec.txt"),
    )
    assert piezo_paths == (
        Path("results/piezo"),
        Path("dataset.piezo.extxyz"),
        Path("piezo"),
        Path("fd2bec-log.pp.piezo.txt"),
    )
    assert bec_build[bec_build.index("-i") + 1] == "results/bec"
    assert piezo_build[piezo_build.index("-i") + 1] == "results/piezo"
