"""Diagnostics for Born effective-charge acoustic-sum-rule residuals."""

from pathlib import Path
from typing import Sequence

import numpy as np


def write_acoustic_sum_rule_diagnostics(
    path: Path,
    bec: np.ndarray,
    symbols: Sequence[str],
    positions: np.ndarray,
    *,
    tolerance: float,
) -> None:
    """Write atom- and species-resolved information for an ASR residual."""
    bec = np.asarray(bec, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError(f"BEC data must have shape (N, 3, 3), got {bec.shape}.")
    if len(symbols) != len(bec) or positions.shape != (len(bec), 3):
        raise ValueError("Symbols and positions must provide one entry per Born-charge tensor.")

    asr_sum = np.sum(bec, axis=0)
    asr_mean = np.mean(bec, axis=0)
    mean_norm = np.linalg.norm(asr_mean, "fro") / np.sqrt(asr_mean.shape[0])
    sum_norm = np.linalg.norm(asr_sum, "fro") / np.sqrt(asr_sum.shape[0])
    species = {}
    for symbol, tensor in zip(symbols, bec):
        species[symbol] = species.get(symbol, np.zeros((3, 3))) + tensor

    with Path(path).open("w", encoding="utf-8") as output:
        output.write("# Acoustic sum-rule diagnostic\n")
        output.write(f"# mean residual norm: {mean_norm:.12e}\n")
        output.write(f"# total residual norm: {sum_norm:.12e}\n")
        output.write(f"# tolerance: {tolerance:.12e}\n")
        output.write("# mean_I Z*_I (the quantity compared with the tolerance)\n")
        np.savetxt(output, asr_mean, fmt="% .12e")
        output.write("# sum_I Z*_I (rows: dipole; columns: displacement)\n")
        np.savetxt(output, asr_sum, fmt="% .12e")

        output.write("#\n# Species-resolved sums\n")
        output.write("# species count Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz\n")
        for symbol, tensor in species.items():
            count = sum(item == symbol for item in symbols)
            values = " ".join(f"{value: .12e}" for value in tensor.flat)
            output.write(f"{symbol} {count} {values}\n")

        output.write("#\n# Atom-resolved Born effective charges\n")
        output.write("# index species x y z Zxx Zxy Zxz Zyx Zyy Zyz Zzx Zzy Zzz\n")
        for index, (symbol, position, tensor) in enumerate(zip(symbols, positions, bec)):
            coordinates = " ".join(f"{value: .12e}" for value in position)
            values = " ".join(f"{value: .12e}" for value in tensor.flat)
            output.write(f"{index} {symbol} {coordinates} {values}\n")
