import argparse

import pytest

from fd2bec import SYMPREC
from fd2bec.cli.aims import post_process_aims, prepare_aims
from fd2bec.cli.dataset import add_data
from fd2bec.cli.dFdE import dFdE2bec
from fd2bec.cli.displacements import generate_displacements
from fd2bec.cli.dPdR import dPdR2bec
from fd2bec.cli.dPdS import dPdS2piezo
from fd2bec.cli.general import (
    add_bec,
    add_piezo,
    bec_extxyz2txt,
    info_extxyz2txt,
    optimize,
    show_piezo,
)
from fd2bec.cli.ml import mace_polar_dPdR, mace_polar_dPdS
from fd2bec.cli.parser import add_shared_argument
from fd2bec.cli.periodic import generate_supercell
from fd2bec.cli.qe import prepare_qe
from fd2bec.cli.structures import (
    convert_format,
    shift_origin,
    sort_structure,
    space_group,
    space_group_dataset,
    tensor_symmetries,
)

SYMMETRY_CLI_CASES = (
    (post_process_aims, ["-i", "input.extxyz"]),
    (prepare_aims, ["-i", "input.extxyz"]),
    (prepare_qe, ["-i", "input.extxyz", "-t", "scf.in"]),
    (dFdE2bec, ["-i", "dataset.extxyz"]),
    (dPdR2bec, ["-i", "dataset.extxyz"]),
    (dPdS2piezo, ["-i", "dataset.extxyz"]),
    (generate_displacements, ["-i", "input.extxyz", "-o", "output.extxyz"]),
    (optimize, ["-i", "input.extxyz", "-p", "1234"]),
    (show_piezo, ["-i", "input.extxyz", "-o", "piezo.txt"]),
    (mace_polar_dPdR, ["-i", "input.extxyz"]),
    (convert_format, ["-i", "input.extxyz", "-o", "output.cif"]),
    (space_group, ["-i", "input.extxyz"]),
    (space_group_dataset, ["-i", "input.extxyz", "-o", "output.csv"]),
    (tensor_symmetries, ["-i", "input.extxyz", "-n", "bec"]),
)

INPUT_STRUCTURE_CLI_CASES = (
    (post_process_aims, ["-i", "input.extxyz"]),
    (prepare_aims, ["-i", "input.extxyz"]),
    (prepare_qe, ["-i", "input.extxyz", "-t", "scf.in"]),
    (generate_displacements, ["-i", "input.extxyz", "-o", "output.extxyz"]),
    (optimize, ["-i", "input.extxyz", "-p", "1234"]),
    (show_piezo, ["-i", "input.extxyz", "-o", "piezo.txt"]),
    (add_bec, ["-i", "input.extxyz", "-d", "bec.txt"]),
    (add_piezo, ["-i", "input.extxyz", "-d", "piezo.txt"]),
    (mace_polar_dPdR, ["-i", "input.extxyz"]),
    (mace_polar_dPdS, ["-i", "input.extxyz"]),
    (generate_supercell, ["-i", "input.extxyz", "-s", "2 2 2", "-o", "output.extxyz"]),
    (convert_format, ["-i", "input.extxyz", "-o", "output.cif"]),
    (shift_origin, ["-i", "input.extxyz", "-o", "output.extxyz"]),
    (
        sort_structure,
        ["-r", "reference.extxyz", "-i", "input.extxyz", "-o", "output.extxyz"],
    ),
    (space_group, ["-i", "input.extxyz"]),
    (tensor_symmetries, ["-i", "input.extxyz", "-n", "bec"]),
)

OUTPUT_STRUCTURE_CLI_CASES = (
    (generate_supercell, ["-i", "input.extxyz", "-s", "2 2 2", "-o", "output.extxyz"]),
    (convert_format, ["-i", "input.extxyz", "-o", "output.cif"]),
    (shift_origin, ["-i", "input.extxyz", "-o", "output.extxyz"]),
    (
        sort_structure,
        ["-r", "reference.extxyz", "-i", "input.extxyz", "-o", "output.extxyz"],
    ),
)

STRUCTURE_INDEX_CLI_CASES = (
    (convert_format, ["-i", "input.extxyz", "-o", "output.cif"]),
    (shift_origin, ["-i", "input.extxyz", "-o", "output.extxyz"]),
)

DATA_FILE_CLI_CASES = (
    (add_bec, ["-i", "input.extxyz", "-d", "data.txt"]),
    (add_piezo, ["-i", "input.extxyz", "-d", "data.txt"]),
    (
        add_data,
        [
            "-i",
            "input.extxyz",
            "-n",
            "data_name",
            "-d",
            "data.txt",
            "-w",
            "info",
            "-o",
            "output.extxyz",
        ],
    ),
)

DATA_NAME_CLI_CASES = (
    (
        add_data,
        [
            "-i",
            "input.extxyz",
            "-n",
            "data_name",
            "-d",
            "data.txt",
            "-w",
            "info",
            "-o",
            "output.extxyz",
        ],
    ),
    (
        bec_extxyz2txt,
        ["-i", "input.extxyz", "-n", "data_name", "-o", "output.txt"],
    ),
    (
        info_extxyz2txt,
        ["-i", "input.extxyz", "-n", "data_name", "-o", "output.txt"],
    ),
)

RESPONSE_QUANTITY_CLI_CASES = (
    (post_process_aims, ["-i", "input.extxyz"]),
    (prepare_aims, ["-i", "input.extxyz"]),
    (prepare_qe, ["-i", "input.extxyz", "-t", "scf.in"]),
)

CARTESIAN_AMPLITUDE_CLI_CASES = (
    (prepare_aims, ["-i", "input.extxyz"]),
    (prepare_qe, ["-i", "input.extxyz", "-t", "scf.in"]),
    (generate_displacements, ["-i", "input.extxyz", "-o", "output.extxyz"]),
    (mace_polar_dPdR, ["-i", "input.extxyz"]),
)


@pytest.mark.parametrize(("module", "required_args"), SYMMETRY_CLI_CASES)
def test_symmetry_clis_share_the_symprec_argument(module, required_args):
    parser = module.prepare_args(module.description)

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "-sp", "2e-4"])

    assert defaults.symprec == SYMPREC
    assert custom.symprec == 2e-4


def test_shared_argument_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown shared argument"):
        add_shared_argument(argparse.ArgumentParser(), "missing")


def test_shared_symprec_requires_a_positive_value():
    parser = argparse.ArgumentParser()
    add_shared_argument(parser, "symprec")

    with pytest.raises(SystemExit):
        parser.parse_args(["--symprec", "0"])


@pytest.mark.parametrize(("module", "required_args"), INPUT_STRUCTURE_CLI_CASES)
def test_structure_clis_share_the_input_structure_argument(module, required_args):
    parser = module.prepare_args(module.description)

    args = parser.parse_args(required_args)

    assert args.input == "input.extxyz"
    assert parser._option_string_actions["--input"].help == "path to the input atomic structure"


@pytest.mark.parametrize(("module", "required_args"), OUTPUT_STRUCTURE_CLI_CASES)
def test_structure_clis_share_the_output_structure_argument(module, required_args):
    parser = module.prepare_args(module.description)

    args = parser.parse_args(required_args)

    assert args.output.startswith("output.")
    assert parser._option_string_actions["--output"].help == "path to the output atomic structure"


@pytest.mark.parametrize(("module", "required_args"), STRUCTURE_INDEX_CLI_CASES)
def test_structure_clis_share_the_structure_index_argument(module, required_args):
    parser = module.prepare_args(module.description)

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "--index", "3"])

    assert defaults.index == 0
    assert custom.index == 3


@pytest.mark.parametrize(("module", "required_args"), DATA_FILE_CLI_CASES)
def test_data_clis_share_the_data_file_argument(module, required_args):
    parser = module.prepare_args(module.description)

    args = parser.parse_args(required_args)

    assert args.data == "data.txt"
    assert parser._option_string_actions["--data"].help == "path to the input numeric data file"


@pytest.mark.parametrize(("module", "required_args"), DATA_NAME_CLI_CASES)
def test_data_clis_share_the_data_name_argument(module, required_args):
    parser = module.prepare_args(module.description)

    args = parser.parse_args(required_args)

    assert args.name == "data_name"
    assert (
        parser._option_string_actions["--name"].help
        == "name of the structure info or per-atom array"
    )


@pytest.mark.parametrize(("module", "required_args"), RESPONSE_QUANTITY_CLI_CASES)
def test_workflow_clis_share_the_response_quantity_argument(module, required_args):
    parser = module.prepare_args(module.description)

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "--what", "piezo"])

    assert defaults.what == "bec"
    assert custom.what == "piezo"
    assert parser._option_string_actions["--what"].choices == ("bec", "piezo")


def test_displacement_cli_uses_the_broader_displacement_target_argument():
    parser = generate_displacements.prepare_args(generate_displacements.description)
    required_args = ["-i", "input.extxyz", "-o", "output.extxyz"]

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "--what", "force_constants"])

    assert defaults.what == "bec"
    assert custom.what == "force_constants"
    assert parser._option_string_actions["--what"].choices == (
        "bec",
        "piezo",
        "forces",
        "stress",
        "elastic",
        "force_constants",
    )


def test_add_data_uses_the_data_destination_argument():
    parser = add_data.prepare_args(add_data.description)
    required_args = [
        "-i",
        "input.extxyz",
        "-n",
        "data_name",
        "-d",
        "data.txt",
        "-o",
        "output.extxyz",
    ]

    info = parser.parse_args([*required_args, "--what", "info"])
    arrays = parser.parse_args([*required_args, "--what", "arrays"])

    assert info.what == "info"
    assert arrays.what == "arrays"
    assert parser._option_string_actions["--what"].choices == (
        "i",
        "info",
        "a",
        "array",
        "arrays",
    )


def test_shared_what_arguments_reject_values_from_other_semantics():
    response_parser = prepare_aims.prepare_args(prepare_aims.description)
    data_parser = add_data.prepare_args(add_data.description)

    with pytest.raises(SystemExit):
        response_parser.parse_args(["-i", "input.extxyz", "--what", "forces"])
    with pytest.raises(SystemExit):
        data_parser.parse_args(
            [
                "-i",
                "input.extxyz",
                "-n",
                "data_name",
                "-d",
                "data.txt",
                "-o",
                "output.extxyz",
                "--what",
                "bec",
            ]
        )


@pytest.mark.parametrize(("module", "required_args"), CARTESIAN_AMPLITUDE_CLI_CASES)
def test_displacement_clis_share_the_cartesian_amplitude_argument(module, required_args):
    parser = module.prepare_args(module.description)

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "--amplitude", "0.002"])

    assert defaults.amplitude == 1e-3
    assert custom.amplitude == 0.002
    assert (
        parser._option_string_actions["--amplitude"].help
        == "Cartesian atomic or cell displacement amplitude in Angstrom (default: %(default)s)"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--amplitude", "0"])


def test_mace_piezo_cli_uses_the_dimensionless_strain_amplitude_argument():
    parser = mace_polar_dPdS.prepare_args(mace_polar_dPdS.description)
    required_args = ["-i", "input.extxyz"]

    defaults = parser.parse_args(required_args)
    custom = parser.parse_args([*required_args, "--amplitude", "0.002"])

    assert defaults.amplitude == 1e-3
    assert custom.amplitude == 0.002
    assert (
        parser._option_string_actions["--amplitude"].help
        == "dimensionless strain amplitude (default: %(default)s)"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*required_args, "--amplitude", "-0.001"])
