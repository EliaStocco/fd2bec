from fd2bec.cli.structures import standardize_centrosymmetric


def test_standardize_centrosymmetric_parser_requires_an_output_path():
    parser = standardize_centrosymmetric.prepare_args(standardize_centrosymmetric.description)

    args = parser.parse_args(["-i", "input.extxyz", "-o", "output.extxyz"])

    assert args.input == "input.extxyz"
    assert args.output == "output.extxyz"
