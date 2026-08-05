# This file is meant to be sourced by the user's submission script.
# Set QE to the pw.x launch command first, for example:
# export QE="srun /path/to/pw.x"

if [ -z "${QE:-}" ]; then
    echo "FD2BEC: set QE to the Quantum ESPRESSO pw.x command before sourcing this file." >&2
    return 1 2>/dev/null || exit 1
fi

FD2BEC_QE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FD2BEC_QE_ROOT="${FD2BEC_QE_SCRIPT_DIR}/__FD2BEC_RELATIVE_ROOT__"
FD2BEC_QE_LAST_INDEX=__FD2BEC_LAST_INDEX__
mkdir -p "${FD2BEC_QE_ROOT}/results"

fd2bec_qe_input() {
    fd2bec_template="$1"
    fd2bec_geometry="$2"
    fd2bec_destination="$3"
    {
        cat "${fd2bec_template}"
        printf '\n'
        cat "${fd2bec_geometry}"
    } > "${fd2bec_destination}"
}

for fd2bec_n in $(seq 0 "${FD2BEC_QE_LAST_INDEX}"); do
    fd2bec_geometry="${FD2BEC_QE_ROOT}/geometries/geometry.n=${fd2bec_n}.in"
    fd2bec_run_dir="${FD2BEC_QE_ROOT}/work/geometry.n=${fd2bec_n}"
    fd2bec_result_dir="${FD2BEC_QE_ROOT}/results/geometry.n=${fd2bec_n}"
    mkdir -p "${fd2bec_run_dir}" "${fd2bec_result_dir}"

    fd2bec_qe_input "${FD2BEC_QE_ROOT}/templates/scf.in" \
        "${fd2bec_geometry}" "${fd2bec_run_dir}/scf.in"
    if [ ! -s "${fd2bec_result_dir}/scf.out" ]; then
        (cd "${fd2bec_run_dir}" && ${QE} -in scf.in > "${fd2bec_result_dir}/scf.out") || return
    fi

    for fd2bec_gdir in 1 2 3; do
        fd2bec_name="nscf.g=${fd2bec_gdir}"
        fd2bec_qe_input "${FD2BEC_QE_ROOT}/templates/${fd2bec_name}.in" \
            "${fd2bec_geometry}" "${fd2bec_run_dir}/${fd2bec_name}.in"
        if [ ! -s "${fd2bec_result_dir}/${fd2bec_name}.out" ]; then
            (cd "${fd2bec_run_dir}" && ${QE} -in "${fd2bec_name}.in" \
                > "${fd2bec_result_dir}/${fd2bec_name}.out") || return
        fi
    done
done

unset fd2bec_n fd2bec_gdir fd2bec_geometry fd2bec_run_dir fd2bec_result_dir fd2bec_name
unset FD2BEC_QE_SCRIPT_DIR FD2BEC_QE_ROOT FD2BEC_QE_LAST_INDEX
unset -f fd2bec_qe_input
