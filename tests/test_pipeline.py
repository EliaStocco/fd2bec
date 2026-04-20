import subprocess
import shutil
from pathlib import Path
import pytest
import numpy as np

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path(__file__).parent / "data"
DATASETS = [p for p in DATA_DIR.iterdir() if p.is_dir()]
DEBUG_DIR = Path(__file__).parent / "test_outputs"

# -------------------------
# Helper
# -------------------------    
def run_pipeline(workdir: Path, method: str):
    
    subprocess.run(
        [
            "prepare_linear_system",
            "-uc", "start.extxyz",
            # "-sc", "supercell.extxyz",
            "-b", "dipole.txt",
            "-A", "displacement.txt",
            "-o", "to_solve.json",
            # "-asr", str(asr),
            # "-spg", spg
        ],
        cwd=workdir,
        check=True,
    )

    subprocess.run(
        [
            "solve_linear_system",
            "-i", "to_solve.json",
            "-o", "solution.json",
            "-m", method,
        ],
        cwd=workdir,
        check=True,
    )

    subprocess.run(
        [
            "solution2bec_txt",
            "-i", "solution.json",
            "-o", "bec.txt",
        ],
        cwd=workdir,
        check=True,
    )


# -------------------------
# The test
# -------------------------
COMBOS = [
    pytest.param(folder, method,
                 id=f"{folder.name}_{method}")
    for folder in DATASETS
    # for asr in [-1, 0, 1, 10, 100, 1e3, 1e4]
    for method in ["lstsq", "pseudo-inverse"] # 
    # for spg in ["true", "false"]
    # if asr == -1
]

@pytest.mark.parametrize("folder, method", COMBOS)
def test_pipeline(tmp_path: Path, folder: Path, method: str,pytestconfig):
    
    # Copy dataset into tmp working directory
    for file in folder.iterdir():
        shutil.copy(file, tmp_path / file.name)

    test_id = f"{folder.name}_{method}"
    debug_out = DEBUG_DIR / test_id

    try:
        # Run pipeline
        run_pipeline(tmp_path, method)

        output = tmp_path / "bec.txt"

        assert output.exists(), f"Missing output for {test_id}"
        assert output.stat().st_size > 0, f"Empty output for {test_id}"

        # Save debug snapshot
        shutil.copytree(tmp_path, debug_out, dirs_exist_ok=True)

        # -------------------------
        # Compute error
        # -------------------------
        ref = np.loadtxt(folder / "bec-ref.txt")
        bec = np.loadtxt(output)

        norm = np.linalg.norm(ref - bec)

        # -------------------------
        # Append row correctly
        # -------------------------
        pytestconfig.results.append({
            "folder": folder.name,
            "method": method,
            "norm": float(norm),
        })


    except Exception:
        debug_out.parent.mkdir(exist_ok=True)
        shutil.copytree(tmp_path, debug_out, dirs_exist_ok=True)

        print(f"\n[DEBUG] Output saved to: {debug_out}")
        raise


# -------------------------
# Run manually + save results
# -------------------------
if __name__ == "__main__":
    pytest.main([__file__])