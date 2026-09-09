#!/usr/bin/env bash
# Re-run prepare_aims for every structure-* folder below a failed dataset.
#
# Usage:
#   AIMS_CONTROL_TEMPLATE=/path/to/aims.in AIMS_FOLDER=/path/to/FHIaims \
#       /path/to/rerun_failed_prepare_aims.sh /path/to/failed
set -Eeuo pipefail

failed_dir="${1:?Usage: $0 FAILED_DIRECTORY}"
single_rerun="$(dirname "$0")/rerun_prepare_aims.sh"

[[ -d "$failed_dir" ]] || {
    echo "Failed directory not found: $failed_dir" >&2
    exit 1
}

status=0
while IFS= read -r -d '' folder; do
    echo "Re-running ${folder#"$failed_dir"/}"
    if ! (cd "$folder" && "$single_rerun"); then
        echo "Failed: $folder" >&2
        status=1
    fi
done < <(find "$failed_dir" -mindepth 2 -maxdepth 2 -type d -name 'structure-*' -print0 | sort -z)

exit "$status"
