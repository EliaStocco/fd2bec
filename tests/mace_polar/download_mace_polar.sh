#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_path="${script_dir}/MACE-POLAR-1-M.model"
partial_path="${model_path}.part"
model_url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model"
expected_sha256="fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"

if [[ ! -s "${model_path}" ]]; then
    curl --fail --location --show-error --output "${partial_path}" "${model_url}"
    mv -- "${partial_path}" "${model_path}"
fi

echo "${expected_sha256}  ${model_path}" | sha256sum --check --status
echo "MACE-POLAR checkpoint ready: ${model_path}"
