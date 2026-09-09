#!/usr/bin/env bash
ulimit -s unlimited

#-----------------------------------#
# Functions
get_current_date_time() {
    date +"%Y-%m-%d %H:%M:%S"
}

calculate_elapsed_time() {
    start_time="$1"
    end_time="$2"
    start_seconds=$(date -d "$start_time" +%s)
    end_seconds=$(date -d "$end_time" +%s)
    elapsed_seconds=$((end_seconds - start_seconds))
    echo "$elapsed_seconds seconds"
}

run_aims(){
    echo "# Job ID: $SLURM_JOB_ID" >> "$LOG_FILE"
    echo "# Date and Time: $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
    echo "Running ${AIMS_OUTPUT_FILE}" >> "$LOG_FILE"
    start_time=$(get_current_date_time)
    cmd="srun ${AIMS} &> aims.out"
    echo "$cmd"
    eval "$cmd"
    end_time=$(get_current_date_time)
    echo "# End Time: $end_time" >> "$LOG_FILE"
    echo "# Elapsed Time: $(calculate_elapsed_time "$start_time" "$end_time")" >> "$LOG_FILE"
    echo ""
    cp aims.out ${AIMS_OUTPUT_FILE}
}

#-----------------------------------#
# Logging
LOG_FILE="log.out"
rm -f "$LOG_FILE"
use_csc="${use_csc:-USE_CSC_DEFAULT}"
delete_csc="${delete_csc:-true}"

mkdir -p results
mapfile -t geometry_files < <(
    find geometries -maxdepth 1 -type f -name 'geometry.n=*.in' -printf '%f\n' | sort -V
)
first_geometry=true
for geometry_file in "${geometry_files[@]}"; do
    gfile="geometries/${geometry_file}"
    n="${geometry_file#geometry.n=}"
    n="${n%.in}"

    if [[ "${first_geometry}" == "true" ]]; then
        csc_control="control.first.in"
        first_geometry=false
    else
        csc_control="control.other.in"
    fi

    export AIMS_OUTPUT_FILE="results/aims.n=${n}.out"
    if [[ ! -e "${AIMS_OUTPUT_FILE}" ]]; then
        cp "${gfile}" geometry.in
        if [[ "${use_csc}" == "true" ]] ; then
            cp "${csc_control}" control.in
        else
            cp control.general.in control.in
        fi
        run_aims
    fi
done
if [[ "${delete_csc}" == "true" ]]; then
    rm -f -- *.csc
fi
touch DONE
