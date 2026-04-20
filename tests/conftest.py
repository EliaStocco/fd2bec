import pytest
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
# structures_dir = repo_root / "fd2bec" / "structures"

@pytest.fixture(scope="session")
def structures_dir():
    return Path(__file__).resolve().parents[1] / "fd2bec" / "structures"


@pytest.fixture(params=[1, 2, 3, 4])
def structure(request, structures_dir):
    n = request.param
    file_path = structures_dir / f"BaTiO3.{n}x{n}x{n}.extxyz"
    return n, file_path


def pytest_configure(config):
    config.results = []   # shared storage


def pytest_sessionfinish(session, exitstatus):
    results = getattr(session.config, "results", [])

    if not results:
        print("\nNo results collected.")
        return

    df = pd.DataFrame(results).sort_values("norm")

    out = Path(session.config.rootpath) / "results.csv"
    df.to_csv(out, index=False)

    print("\nSaved results.csv")
    print(df)
    
    df_to_pdf(df,Path(session.config.rootpath) / "table.pdf")
    
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
    


