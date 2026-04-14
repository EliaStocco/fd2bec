import subprocess
import shutil
from pathlib import Path
import pytest

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path(__file__).parent / "data"
DATASETS = [p for p in DATA_DIR.iterdir() if p.is_dir()]

DEBUG_DIR = Path(__file__).parent / "test_outputs"


# -------------------------
# Helper
# -------------------------
def run_pipeline(workdir: Path, asr: float, method: str):
    subprocess.run(
        [
            "prepare_linear_system",
            "-uc", "start.extxyz",
            "-sc", "supercell.extxyz",
            "-b", "dipole.txt",
            "-A", "displacement.txt",
            "-o", "to_solve.json",
            "-asr", str(asr),
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
            "-o", f"bec.txt",
        ],
        cwd=workdir,
        check=True,
    )


# -------------------------
# The test
# -------------------------
@pytest.mark.parametrize("method", ['pseudo-inverse', 'lstsq'], ids=str)
@pytest.mark.parametrize("asr", [-1, 0, 1, 10, 100, 1000], ids=str)
@pytest.mark.parametrize("folder", DATASETS, ids=lambda p: p.name)
def test_pipeline(tmp_path: Path, folder: Path, asr: float, method: str, request):

    # Copy dataset into tmp working directory
    for file in folder.iterdir():
        shutil.copy(file, tmp_path / file.name)

    test_id = f"{folder.name}_asr{asr}_{method}"
    debug_out = DEBUG_DIR / test_id

    try:
        # Run pipeline
        run_pipeline(tmp_path, asr, method)

        # Assertions
        output = tmp_path / "bec.txt"
        assert output.exists(), f"Missing output for {test_id}"
        assert output.stat().st_size > 0, f"Empty output for {test_id}"
        shutil.copytree(tmp_path, debug_out, dirs_exist_ok=True)
        
        # ref = np.loadtxt(tmp_path / "bec-ref.txt")
        # bec = np.loadtxt(output)
        # assert np.allclose(ref,bec,atol=BEC_TOL*len(bec))
        
        pass

    except Exception:
        # On failure: always save outputs
        debug_out.parent.mkdir(exist_ok=True)
        shutil.copytree(tmp_path, debug_out, dirs_exist_ok=True)
        print(f"\n[DEBUG] Output saved to: {debug_out}")
        raise
    
if __name__ == "__main__":
    pytest.main([__file__])