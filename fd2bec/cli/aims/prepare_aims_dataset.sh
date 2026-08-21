#!/usr/bin/env bash
# Prepare every structure in an extxyz dataset for FHI-aims, then archive it.
#
# Usage: source prepare_aims_dataset.sh [dataset.extxyz] [calculations-dir]
#
# The script first splits a new dataset into one folder per structure.  It then
# prepares each folder in parallel, writes a log, creates rerun.sh, and stores
# completed folders as ZIP files.  Calling it again resumes unfinished work;
# archived folders are not recreated.
set -Eeuo pipefail

input_file="${1:-dataset.extxyz}"
output_dir="${2:-calculations}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
control_template="${AIMS_CONTROL_TEMPLATE:-$script_dir/aims.in}"
aims_folder="${AIMS_FOLDER:-/home/stoccoel/codes/FHIaims-polarization/FHIaims-scalapack}"
rerun_template="$script_dir/prepare_aims_dataset_rerun.sh.template"

if [[ ! -f "$input_file" ]]; then
    echo "Dataset file not found: $input_file" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$control_template" ]]; then
    echo "Control template not found: $control_template" >&2
    echo "Set AIMS_CONTROL_TEMPLATE to the control.in template to use." >&2
    return 1 2>/dev/null || exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "zip is required to archive completed structure folders" >&2
    return 1 2>/dev/null || exit 1
fi

if ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is required to pin workers to the requested CPUs" >&2
    return 1 2>/dev/null || exit 1
fi

# Slurm constrains this list to the CPUs allocated to the job.  On a regular
# machine it contains every CPU available to this process.
available_cpu_ids() {
    local allowed range start end cpu
    local -a ranges

    allowed=$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status)
    [[ -n "$allowed" ]] || return 1
    IFS=',' read -r -a ranges <<< "$allowed"
    for range in "${ranges[@]}"; do
        if [[ "$range" == *-* ]]; then
            start=${range%-*}
            end=${range#*-}
            for ((cpu = start; cpu <= end; cpu++)); do
                printf '%s\n' "$cpu"
            done
        else
            printf '%s\n' "$range"
        fi
    done
}

mapfile -t cpu_slots < <(available_cpu_ids)
if ((${#cpu_slots[@]} == 0)); then
    echo "Could not determine the CPUs available to this process" >&2
    return 1 2>/dev/null || exit 1
fi

# Only split a dataset when this is a new calculation directory.  Running the
# splitter after folders have been archived would recreate those folders.
shopt -s nullglob
existing_results=("$output_dir"/structure-* "$output_dir"/structure-*.zip)
shopt -u nullglob
if ((${#existing_results[@]} == 0)); then
    echo "Splitting $input_file into $output_dir"
    split_extxyz -i "$input_file" -o "$output_dir"
else
    echo "Using existing structure folders and archives in $output_dir"
fi

create_rerun_script() {
    local folder="${1%/}"
    local log_file="${folder}/summary.txt"
    local rerun_tmp

    if [[ ! -f "$rerun_template" ]]; then
        echo "Missing rerun-script template: $rerun_template" | tee -a "$log_file" >&2
        return 1
    fi

    if ! rerun_tmp=$(mktemp "${folder}/.rerun.sh.XXXXXX" 2>> "$log_file"); then
        echo "Failed to create rerun.sh for $folder" | tee -a "$log_file" >&2
        return 1
    fi
    if ! sed \
        -e "s|__CONTROL_TEMPLATE__|$control_template|g" \
        -e "s|__AIMS_FOLDER__|$aims_folder|g" \
        "$rerun_template" > "$rerun_tmp" 2>> "$log_file" \
        || ! chmod 755 "$rerun_tmp" 2>> "$log_file" \
        || ! mv -f -- "$rerun_tmp" "${folder}/rerun.sh" 2>> "$log_file"; then
        echo "Failed to write rerun.sh for $folder" | tee -a "$log_file" >&2
        rm -f -- "$rerun_tmp" 2>> "$log_file"
        return 1
    fi
}

archive_folder() {
    local folder="${1%/}"
    local archive="${folder}.zip"
    local archive_tmp
    local log_file="${folder}/summary.txt"

    if ! : >> "$log_file"; then
        echo "Cannot write archive log for $folder" >&2
        return 1
    fi
    create_rerun_script "$folder" || return 1
    if ! archive_tmp=$(mktemp "${output_dir}/.${folder##*/}.zip.XXXXXX" 2>> "$log_file"); then
        echo "Failed to create a temporary archive for $folder" | tee -a "$log_file" >&2
        return 1
    fi
    if ! rm -f -- "$archive_tmp" 2>> "$log_file"; then
        echo "Failed to prepare the temporary archive for $folder" | tee -a "$log_file" >&2
        return 1
    fi

    echo "Archiving $folder" | tee -a "$log_file"
    if (
        cd "$output_dir"
        zip -qr "${archive_tmp##*/}" "${folder##*/}"
    ) >> "$log_file" 2>&1 && mv -f -- "$archive_tmp" "$archive" 2>> "$log_file"; then
        if ! rm -rf -- "$folder" 2>> "$log_file"; then
            echo "Archive created for $folder, but its folder could not be removed" \
                | tee -a "$log_file" >&2
            return 1
        fi
    else
        echo "Failed to archive $folder; leaving its folder intact" \
            | tee -a "$log_file" >&2
        rm -f -- "$archive_tmp" 2>> "$log_file"
        return 1
    fi
}

folders=()
completed_folders=()
for folder in "$output_dir"/structure-*/; do
    [[ -d "$folder" ]] || continue

    # A successful prepare_aims run always writes the reference geometry.
    # Treat its presence as the completion marker so reruns only pick up
    # failed or not-yet-started structure folders.
    if [[ -f "${folder}geometries/geometry.n=0.in" ]]; then
        completed_folders+=("$folder")
        continue
    fi

    folders+=("$folder")
done

process_folder() {
    local folder="$1"

    echo "Processing ${folder%/}"
    (
        cd "$folder"
        : > summary.txt

        space_group -i start.extxyz > symmetries.txt 2>> summary.txt || exit 1
        tensor_symmetries -i start.extxyz -n bec --basis fractional \
            >> symmetries.txt 2>> summary.txt || exit 1
        cp "$control_template" control.in 2>> summary.txt || exit 1
        prepare_aims \
            -i start.extxyz \
            --basis tight \
            --aims-folder "$aims_folder" \
            2>&1 | tee -a summary.txt || exit 1

        rm fd2bec-log.txt 2>> summary.txt || exit 1
    ) || {
        echo "Preparation failed for ${folder%/}; see summary.txt" \
            | tee -a "${folder%/}/summary.txt" >&2
        return 1
    }

    archive_folder "$folder"
}

status=0
archived=0
for folder in "${completed_folders[@]}"; do
    if archive_folder "$folder"; then
        archived=$((archived + 1))
    else
        status=1
    fi
done

if ((${#folders[@]})); then
    echo "Archived $archived completed structure folder(s); processing ${#folders[@]} remaining."

    pids=()
    for slot_index in "${!cpu_slots[@]}"; do
        (
            cpu_slot="${cpu_slots[$slot_index]}"

            # This affinity is inherited by every command launched by the worker.
            taskset --cpu-list --pid "$cpu_slot" "$BASHPID" >/dev/null

            for ((folder_index = slot_index;
                  folder_index < ${#folders[@]};
                  folder_index += ${#cpu_slots[@]})); do
                process_folder "${folders[$folder_index]}"
            done
        ) &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
else
    echo "Archived $archived completed structure folder(s); no processing remains."
fi

if ((status != 0)); then
    echo "One or more workers failed" >&2
fi

return "$status" 2>/dev/null || exit "$status"
