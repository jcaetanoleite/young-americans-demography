from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, root

from .model import (
    MODEL_YEARS,
    CohortComponentModel,
    linear_convergence_path,
    make_group_scaled_asfr,
    make_tfr_scaled_asfr,
    summarize_projection,
)


@dataclass
class ScenarioBundle:
    timeseries: pd.DataFrame
    frontier: pd.DataFrame
    parameters: dict[str, float]


def run_revised_scenarios(data_dir: str | Path) -> ScenarioBundle:
    data_dir = Path(data_dir)
    model = CohortComponentModel(data_dir)

    official_paths = {
        scenario: {
            year: float(model.nim_scenarios[scenario].loc[year])
            for year in MODEL_YEARS
        }
        for scenario in ["zero", "low", "main", "high"]
    }

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

    baseline = summarize_projection(
        model.project(official_paths["main"], model.asfr_base),
        "Census principal rebased para 2025",
    )

    nim_574 = {year: 574_000.0 for year in MODEL_YEARS}
    restriction_census = summarize_projection(
        model.project(nim_574, model.asfr_base),
        "NIM 574 mil; fecundidade Census",
    )
    restriction_cbo = summarize_projection(
        model.project(nim_574, asfr_cbo),
        "NIM 574 mil; fecundidade CBO",
    )

    zero_census = summarize_projection(
        model.project(official_paths["zero"], model.asfr_base),
        "Entrada estrangeira zero; fecundidade Census",
    )
    zero_cbo = summarize_projection(
        model.project(official_paths["zero"], asfr_cbo),
        "Entrada estrangeira zero; fecundidade CBO",
    )

    china = pd.read_csv(
        data_dir / "china_un_wpp_2024_targets.csv"
    ).set_index("Year")
    target_oadr = float(china.loc[2100, "oadr"])
    target_tdr = float(china.loc[2100, "tdr"])

    def conditional_projection(
        long_run_tfr: float,
        long_run_nim: float,
        convergence_year: int = 2035,
    ) -> pd.DataFrame:
        tfr_path = linear_convergence_path(
            cbo_tfr, long_run_tfr, convergence_year
        )
        nim_path = linear_convergence_path(
            cbo_nim, long_run_nim, convergence_year
        )
        asfr_path = make_tfr_scaled_asfr(asfr_cbo, tfr_path)
        return summarize_projection(
            model.project(nim_path, asfr_path), "conditional"
        )

    tfr_match_oadr = brentq(
        lambda tfr: conditional_projection(tfr, 0.0)
        .set_index("year")
        .loc[2100, "old_age_dependency"]
        - target_oadr,
        0.6,
        1.8,
    )
    tfr_match_tdr = brentq(
        lambda tfr: conditional_projection(tfr, 0.0)
        .set_index("year")
        .loc[2100, "total_dependency"]
        - target_tdr,
        0.6,
        1.8,
    )

    china_like_oadr = conditional_projection(tfr_match_oadr, 0.0)
    china_like_oadr["scenario"] = (
        f"China-like NIM zero; TFR {tfr_match_oadr:.3f}"
    )
    china_like_tdr = conditional_projection(tfr_match_tdr, 0.0)
    china_like_tdr["scenario"] = (
        f"China-like NIM zero; TFR {tfr_match_tdr:.3f}"
    )

    frontier_rows: list[dict[str, float]] = []
    for convergence_year in [2035, 2045, 2055, 2070]:
        def equations(parameters: np.ndarray) -> np.ndarray:
            tfr, nim_millions = parameters
            summary = conditional_projection(
                float(tfr),
                float(nim_millions) * 1_000_000.0,
                convergence_year,
            ).set_index("year")
            return np.array(
                [
                    (
                        summary.loc[2100, "old_age_dependency"]
                        - target_oadr
                    )
                    / 10.0,
                    (
                        summary.loc[2100, "total_dependency"]
                        - target_tdr
                    )
                    / 10.0,
                ]
            )

        solution = root(equations, np.array([1.25, -0.3]))
        if not solution.success:
            raise RuntimeError(solution.message)

        summary = conditional_projection(
            float(solution.x[0]),
            float(solution.x[1]) * 1_000_000.0,
            convergence_year,
        ).set_index("year")
        frontier_rows.append(
            {
                "convergence_year": convergence_year,
                "long_run_TFR": float(solution.x[0]),
                "long_run_net_migration": (
                    float(solution.x[1]) * 1_000_000.0
                ),
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

    timeseries = pd.concat(
        [
            baseline,
            restriction_census,
            restriction_cbo,
            zero_census,
            zero_cbo,
            china_like_oadr,
            china_like_tdr,
        ],
        ignore_index=True,
    )

    parameters = {
        "china_target_oadr_2100": target_oadr,
        "china_target_tdr_2100": target_tdr,
        "zero_nim_tfr_match_oadr": float(tfr_match_oadr),
        "zero_nim_tfr_match_tdr": float(tfr_match_tdr),
    }

    return ScenarioBundle(
        timeseries=timeseries,
        frontier=pd.DataFrame(frontier_rows),
        parameters=parameters,
    )
