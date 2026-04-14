import pytest
import os
from pathlib import Path
import pandas as pd

test_dir = Path(__file__).parent.parent/"tests"
repo_root = Path(os.path.dirname(__file__))
structures_dir = repo_root / "structures"

@pytest.fixture(params=[1, 2, 3, 4])
def structure(request):
    n = request.param
    file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
    return n, file_path

_RESULTS = []

def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        print("\nNo results collected.")
        return

    df = pd.DataFrame(_RESULTS)
    df = df.sort_values("norm")

    out = Path(test_dir/"results.csv")
    df.to_csv(out, index=False)

    print("\nSaved results.csv")
    print(df)
    
    df_to_pdf(df,test_dir/"table.pdf")


def df_to_pdf(df, filename):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.3 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()