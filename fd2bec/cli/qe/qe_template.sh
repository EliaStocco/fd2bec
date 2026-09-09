# This file is meant to be sourced by the user's submission script.
# Set QE to the Quantum ESPRESSO command before sourcing this file, for example:
# export QE="srun /path/to/pw.x"

if [ -z "${QE:-}" ]; then
    echo "FD2BEC: set QE to the Quantum ESPRESSO pw.x command first." >&2
    return 1 2>/dev/null || exit 1
fi

# All paths are relative to the folder containing this script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Join a QE input template with one generated geometry.
make_input() {
    template="$1"
    geometry="$2"
    input="$3"

    cat "$template" > "$input"
    echo >> "$input"
    cat "$geometry" >> "$input"
}

mkdir -p "$ROOT_DIR/work" "$ROOT_DIR/results"

# Find all structures prepared by prepare_qe.py.
geometry_files=("$ROOT_DIR"/geometries/geometry.n=*.in)
if [ ! -f "${geometry_files[0]}" ]; then
    echo "FD2BEC: no geometry files found in $ROOT_DIR/geometries." >&2
    return 1 2>/dev/null || exit 1
fi

for geometry in "${geometry_files[@]}"; do
    # Use the geometry filename for its work and results directories.
    name="${geometry##*/}"
    name="${name%.in}"
    work_dir="$ROOT_DIR/work/$name"
    result_dir="$ROOT_DIR/results/$name"
    mkdir -p "$work_dir" "$result_dir"

    # Run the SCF calculation unless it already finished successfully.
    make_input "$ROOT_DIR/templates/scf.in" "$geometry" "$work_dir/scf.in"
    if [ ! -f "$result_dir/scf.out" ] || ! grep -q "JOB DONE" "$result_dir/scf.out"; then
        (cd "$work_dir" && $QE -in scf.in > "$result_dir/scf.out") || return
    fi

    # Run one Berry-phase NSCF calculation for each lattice direction.
    for direction in 1 2 3; do
        calculation="nscf.g=$direction"
        make_input "$ROOT_DIR/templates/$calculation.in" "$geometry" \
            "$work_dir/$calculation.in"
        if [ ! -f "$result_dir/$calculation.out" ] || \
            ! grep -q "JOB DONE" "$result_dir/$calculation.out"; then
            (cd "$work_dir" && $QE -in "$calculation.in" \
                > "$result_dir/$calculation.out") || return
        fi
    done
done

# This empty file means that every SCF and NSCF calculation finished.
: > "$ROOT_DIR/DONE"

unset ROOT_DIR geometry_files geometry name work_dir result_dir direction calculation
unset -f make_input
