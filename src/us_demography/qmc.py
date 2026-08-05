from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .model import (
    MODEL_YEARS,
    CohortComponentModel,
    linear_convergence_path,
    make_group_scaled_asfr,
    make_tfr_scaled_asfr,
    summarize_projection,
)


def run_qmc_sensitivity(
    data_dir: str | Path,
    draws: int = 2048,
    seed: int = 20260805,
) -> pd.DataFrame:
    """Run a design-based Sobol sensitivity exercise.

    The ranges below are analyst-specified stress-test ranges. Results are not
    posterior draws and must not be presented as probabilistic confidence or
    prediction intervals.
    """
    if draws <= 0 or draws & (draws - 1):
        raise ValueError("draws must be a positive power of two")

    data_dir = Path(data_dir)
    model = CohortComponentModel(data_dir)

    cbo_frame = pd.read_csv(data_dir / "cbo_2026_assumptions.csv")
    cbo = cbo_frame.set_index("year")
    cbo_nim = {
        year: float(cbo.loc[year, "net_international_migration"])
        for year in MODEL_YEARS
    }
    cbo_tfr = {
        year: float(cbo.loc[year, "tfr_total"])
        for year in MODEL_YEARS
    }
    asfr_cbo = make_group_scaled_asfr(model.asfr_base, cbo_frame)

    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    sample = sampler.random_base2(m=int(np.log2(draws)))

    # Explicit stress-test ranges
    long_run_tfr = 0.9 + sample[:, 0] * (1.9 - 0.9)
    long_run_nim = -500_000 + sample[:, 1] * 2_500_000
    convergence_year = np.rint(2035 + sample[:, 2] * 35).astype(int)

    rows: list[dict[str, float]] = []
    for draw in range(draws):
        tfr_path = linear_convergence_path(
            cbo_tfr,
            float(long_run_tfr[draw]),
            int(convergence_year[draw]),
        )
        nim_path = linear_convergence_path(
            cbo_nim,
            float(long_run_nim[draw]),
            int(convergence_year[draw]),
        )
        asfr_path = make_tfr_scaled_asfr(asfr_cbo, tfr_path)
        summary = summarize_projection(
            model.project(nim_path, asfr_path), f"draw_{draw}"
        ).set_index("year")

        rows.append(
            {
                "draw": draw,
                "long_run_TFR": float(long_run_tfr[draw]),
                "long_run_NIM": float(long_run_nim[draw]),
                "convergence_year": int(convergence_year[draw]),
                "population_2100": float(
                    summary.loc[2100, "population"]
                ),
                "births_2100": float(summary.loc[2100, "births"]),
                "old_age_dependency_2100": float(
                    summary.loc[2100, "old_age_dependency"]
                ),
                "total_dependency_2100": float(
                    summary.loc[2100, "total_dependency"]
                ),
            }
        )

    return pd.DataFrame(rows)
