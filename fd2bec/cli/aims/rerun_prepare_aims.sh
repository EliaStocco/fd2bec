#!/usr/bin/env bash
# Recreate one structure folder's prepare_aims outputs from start.extxyz.
#
# Run this from a structure-* directory:
#   AIMS_CONTROL_TEMPLATE=/path/to/aims.in AIMS_FOLDER=/path/to/FHIaims \
#       /path/to/rerun_prepare_aims.sh
#
# The script removes only files made by the preparation workflow.  It keeps
# start.extxyz, then rebuilds the geometry files, control files, and summary.
set -Eeuo pipefail

control_template="${AIMS_CONTROL_TEMPLATE:?Set AIMS_CONTROL_TEMPLATE to your aims.in template.}"
aims_folder="${AIMS_FOLDER:?Set AIMS_FOLDER to the FHI-aims installation directory.}"

[[ -f start.extxyz ]] || {
    echo "Run this script from a structure folder containing start.extxyz." >&2
    exit 1
}
[[ -f "$control_template" ]] || {
    echo "Control template not found: $control_template" >&2
    exit 1
}

rm -rf -- geometries
rm -f -- \
    symmetries.txt \
    control.in control.first.in control.general.in control.other.in \
    species.tight.in \
    displaced-structures.extxyz displacements.txt \
    fd2bec-log.txt sourceme.sh summary.txt

: > summary.txt
space_group -i start.extxyz > symmetries.txt 2>> summary.txt
tensor_symmetries -i start.extxyz -n bec --basis fractional >> symmetries.txt 2>> summary.txt
cp -- "$control_template" control.in 2>> summary.txt
prepare_aims \
    -i start.extxyz \
    --basis tight \
    --aims-folder "$aims_folder" \
    2>&1 | tee -a summary.txt

rm -f -- fd2bec-log.txt
