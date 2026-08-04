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


mkdir -p results
for n in {0..NNN}; do
    gfile="geometries/geometry.n=${n}.in"
    if [ ! -e "${gfile}" ]; then
        break
    fi
    export AIMS_OUTPUT_FILE="results/aims.n=${n}.out"
    if [ ! -e "${AIMS_OUTPUT_FILE}" ]; then
        cp ${gfile} geometry.in
        run_aims
    fi
done
