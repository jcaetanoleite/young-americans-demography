from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

POP_YEARS = np.arange(2022, 2101)
MODEL_YEARS = np.arange(2026, 2101)
MODEL_INDEX = np.arange(2025, 2101)
AGES = np.arange(101)
FERTILITY_AGES = np.arange(14, 55)

SCENARIO_FILE_SUFFIX = {
    "zero": "zero",
    "low": "low",
    "main": "mid",
    "high": "hi",
}


@dataclass(frozen=True)
class ProjectionResult:
    year: np.ndarray
    population: np.ndarray
    births: np.ndarray
    migration: np.ndarray


class CohortComponentModel:
    """Annual two-sex cohort-component model, ages 0 to 100+.

    The engine uses Census survival ratios and age-specific fertility rates.
    The age-sex composition of net international migration is recovered from
    the four Census immigration variants and interpolated for intermediate
    aggregate NIM paths.

    This is a conditional scenario engine. It is not a probabilistic forecast.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.population_scenarios = self._load_population_scenarios()
        self.birth_scenarios = self._load_birth_scenarios()
        self.nim_scenarios = self._load_nim_scenarios()
        self.asfr_all = self._load_asfr()
        self.survival = self._load_survival_ratios()
        self.actual_2025 = self._load_actual_2025()

        self.asfr_base = self.asfr_all[2026 - 2023 :]
        self.transition_residuals: dict[str, np.ndarray] = {}
        self.birth_exposure_factors: dict[str, np.ndarray] = {}
        self.male_birth_shares: dict[str, np.ndarray] = {}
        self._calibrate_official_scenarios()

    def _load_population_scenarios(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        pop_columns = [f"POP_{age}" for age in AGES]

        for scenario, suffix in SCENARIO_FILE_SUFFIX.items():
            frame = pd.read_csv(self.data_dir / f"np2023_d1_{suffix}.csv")
            frame = frame[
                (frame["ORIGIN"] == 0)
                & (frame["RACE"] == 0)
                & (frame["SEX"].isin([1, 2]))
            ]

            array = np.zeros((len(POP_YEARS), 2, len(AGES)), dtype=float)
            for sex_index, sex_code in enumerate([1, 2]):
                sex_frame = frame[frame["SEX"] == sex_code].set_index("YEAR")
                array[:, sex_index, :] = sex_frame.loc[
                    POP_YEARS, pop_columns
                ].to_numpy(dtype=float)
            output[scenario] = array

        return output

    def _load_birth_scenarios(self) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for scenario, suffix in SCENARIO_FILE_SUFFIX.items():
            frame = pd.read_csv(self.data_dir / f"np2023_d2_{suffix}.csv")
            frame = frame[frame["RACE_HISP"] == 0]
            output[scenario] = frame.pivot(
                index="YEAR", columns="SEX", values="BIRTHS"
            ).sort_index()
        return output

    def _load_nim_scenarios(self) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for scenario, suffix in SCENARIO_FILE_SUFFIX.items():
            frame = pd.read_csv(self.data_dir / f"np2023_d4_{suffix}.csv")
            frame = frame[(frame["RACE_HISP"] == 0) & (frame["SEX"] == 0)]
            output[scenario] = frame.set_index("YEAR")["TOTAL_NIM"].sort_index()
        return output

    def _load_asfr(self) -> np.ndarray:
        frame = pd.read_csv(self.data_dir / "np2023_a1.csv")
        frame = frame[frame["GROUP"] == 0].set_index("YEAR")
        columns = [f"ASFR_{age}" for age in FERTILITY_AGES]
        return frame.loc[2023:2100, columns].to_numpy(dtype=float)

    def _load_survival_ratios(self) -> np.ndarray:
        frame = pd.read_csv(self.data_dir / "np2023_a4.csv")
        frame = frame[
            (frame["NATIVITY"] == 0)
            & (frame["GROUP"] == 0)
            & (frame["SEX"].isin([1, 2]))
        ]
        columns = [f"SRAT_{age}" for age in AGES]
        output = np.zeros((78, 2, 101), dtype=float)
        for sex_index, sex_code in enumerate([1, 2]):
            sex_frame = frame[frame["SEX"] == sex_code].set_index("YEAR")
            output[:, sex_index, :] = sex_frame.loc[
                2023:2100, columns
            ].to_numpy(dtype=float)
        return output

    def _load_actual_2025(self) -> np.ndarray:
        frame = pd.read_csv(self.data_dir / "nc-est2025-agesex-res.csv")
        frame = frame[
            (frame["SEX"].isin([1, 2])) & (frame["AGE"].between(0, 100))
        ]
        output = np.zeros((2, 101), dtype=float)
        for sex_index, sex_code in enumerate([1, 2]):
            sex_frame = frame[frame["SEX"] == sex_code].set_index("AGE")
            output[sex_index] = sex_frame.loc[
                AGES, "POPESTIMATE2025"
            ].to_numpy(dtype=float)
        return output

    def _calibrate_official_scenarios(self) -> None:
        for scenario in SCENARIO_FILE_SUFFIX:
            population = self.population_scenarios[scenario]
            births = self.birth_scenarios[scenario]

            residuals = np.zeros((78, 2, 101), dtype=float)
            exposure_factor = np.zeros(78, dtype=float)
            male_share = np.zeros(78, dtype=float)

            for year in range(2023, 2101):
                time_index = year - 2023
                current_index = year - 2022
                previous_index = current_index - 1
                survival = self.survival[time_index]

                total_births = float(births.loc[year, 0])
                male_births = float(births.loc[year, 1])
                female_births = float(births.loc[year, 2])
                birth_vector = np.array([male_births, female_births])

                female_exposure = np.dot(
                    self.asfr_all[time_index],
                    population[previous_index, 1, 14:55],
                )
                exposure_factor[time_index] = total_births / female_exposure
                male_share[time_index] = male_births / total_births

                current = population[current_index]
                previous = population[previous_index]

                residuals[time_index, :, 0] = (
                    current[:, 0] - birth_vector * survival[:, 0]
                )
                residuals[time_index, :, 1:100] = (
                    current[:, 1:100]
                    - previous[:, :99] * survival[:, 1:100]
                )
                residuals[time_index, :, 100] = (
                    current[:, 100]
                    - (previous[:, 99] + previous[:, 100])
                    * survival[:, 100]
                )

            self.transition_residuals[scenario] = residuals
            self.birth_exposure_factors[scenario] = exposure_factor
            self.male_birth_shares[scenario] = male_share

    def _interpolated_components(
        self, year: int, target_nim: float
    ) -> tuple[np.ndarray, float, float]:
        time_index = year - 2023
        scenario_order = ["zero", "low", "main", "high"]
        knots = np.array(
            [self.nim_scenarios[name].loc[year] for name in scenario_order],
            dtype=float,
        )
        ordering = np.argsort(knots)
        knots = knots[ordering]
        scenario_order = [scenario_order[index] for index in ordering]

        if target_nim <= knots[0]:
            lower, upper = 0, 1
        elif target_nim >= knots[-1]:
            lower, upper = len(knots) - 2, len(knots) - 1
        else:
            upper = int(np.searchsorted(knots, target_nim))
            lower = upper - 1

        denominator = knots[upper] - knots[lower]
        weight = 0.0 if denominator == 0 else (
            target_nim - knots[lower]
        ) / denominator

        low_name = scenario_order[lower]
        high_name = scenario_order[upper]

        residual = (
            (1.0 - weight) * self.transition_residuals[low_name][time_index]
            + weight * self.transition_residuals[high_name][time_index]
        )
        exposure_factor = (
            (1.0 - weight)
            * self.birth_exposure_factors[low_name][time_index]
            + weight
            * self.birth_exposure_factors[high_name][time_index]
        )
        male_share = (
            (1.0 - weight) * self.male_birth_shares[low_name][time_index]
            + weight * self.male_birth_shares[high_name][time_index]
        )
        return residual, float(exposure_factor), float(male_share)

    def project(
        self,
        nim_path: Mapping[int, float],
        asfr_path: np.ndarray,
        initial_population: np.ndarray | None = None,
    ) -> ProjectionResult:
        if asfr_path.shape != (len(MODEL_YEARS), len(FERTILITY_AGES)):
            raise ValueError(
                "asfr_path must have shape "
                f"{(len(MODEL_YEARS), len(FERTILITY_AGES))}"
            )

        population = np.zeros((len(MODEL_INDEX), 2, 101), dtype=float)
        population[0] = (
            self.actual_2025
            if initial_population is None
            else np.asarray(initial_population, dtype=float)
        )
        births = np.full(len(MODEL_INDEX), np.nan, dtype=float)
        migration = np.full(len(MODEL_INDEX), np.nan, dtype=float)

        for output_index, year in enumerate(MODEL_YEARS, start=1):
            target_nim = float(nim_path[year])
            residual, exposure_factor, male_share = (
                self._interpolated_components(year, target_nim)
            )
            survival = self.survival[year - 2023]

            total_births = exposure_factor * np.dot(
                asfr_path[output_index - 1],
                population[output_index - 1, 1, 14:55],
            )
            births_by_sex = np.array(
                [male_share * total_births, (1.0 - male_share) * total_births]
            )

            next_population = np.zeros((2, 101), dtype=float)
            next_population[:, 0] = (
                births_by_sex * survival[:, 0] + residual[:, 0]
            )
            next_population[:, 1:100] = (
                population[output_index - 1, :, :99]
                * survival[:, 1:100]
                + residual[:, 1:100]
            )
            next_population[:, 100] = (
                (
                    population[output_index - 1, :, 99]
                    + population[output_index - 1, :, 100]
                )
                * survival[:, 100]
                + residual[:, 100]
            )

            population[output_index] = np.maximum(next_population, 0.0)
            births[output_index] = total_births
            migration[output_index] = target_nim

        return ProjectionResult(
            year=MODEL_INDEX.copy(),
            population=population,
            births=births,
            migration=migration,
        )


def summarize_projection(
    result: ProjectionResult, scenario: str
) -> pd.DataFrame:
    by_age = result.population.sum(axis=1)
    total = by_age.sum(axis=1)
    young = by_age[:, :15].sum(axis=1)
    working_age = by_age[:, 15:65].sum(axis=1)
    elderly = by_age[:, 65:].sum(axis=1)

    median_age = []
    for distribution in by_age:
        cumulative = np.cumsum(distribution)
        cutoff = cumulative[-1] / 2.0
        age = int(np.searchsorted(cumulative, cutoff))
        if age == 0:
            interpolated = 0.0
        else:
            previous = cumulative[age - 1]
            share_within_age = (
                (cutoff - previous) / max(distribution[age], 1.0)
            )
            interpolated = age + float(np.clip(share_within_age, 0.0, 1.0))
        median_age.append(interpolated)

    return pd.DataFrame(
        {
            "scenario": scenario,
            "year": result.year,
            "population": total,
            "age_0_14": young,
            "age_15_64": working_age,
            "age_65_plus": elderly,
            "old_age_dependency": 100.0 * elderly / working_age,
            "total_dependency": 100.0 * (young + elderly) / working_age,
            "share_65_plus": 100.0 * elderly / total,
            "median_age": median_age,
            "births": result.births,
            "net_international_migration": result.migration,
        }
    )


def make_group_scaled_asfr(
    base_asfr: np.ndarray, cbo_assumptions: pd.DataFrame
) -> np.ndarray:
    assumptions = cbo_assumptions.set_index("year")
    output = np.zeros_like(base_asfr)
    under_30 = FERTILITY_AGES < 30
    age_30_plus = FERTILITY_AGES >= 30

    for index, year in enumerate(MODEL_YEARS):
        row = base_asfr[index].copy()
        output[index, under_30] = (
            row[under_30]
            * assumptions.loc[year, "tfr_under30"]
            / row[under_30].sum()
        )
        output[index, age_30_plus] = (
            row[age_30_plus]
            * assumptions.loc[year, "tfr_30plus"]
            / row[age_30_plus].sum()
        )
    return output


def make_tfr_scaled_asfr(
    reference_asfr: np.ndarray, tfr_path: Mapping[int, float]
) -> np.ndarray:
    output = reference_asfr.copy()
    for index, year in enumerate(MODEL_YEARS):
        output[index] *= float(tfr_path[year]) / output[index].sum()
    return output


def linear_convergence_path(
    near_term: Mapping[int, float],
    long_run_value: float,
    convergence_year: int,
    anchor_year: int = 2030,
) -> dict[int, float]:
    if convergence_year <= anchor_year:
        raise ValueError("convergence_year must be greater than anchor_year")

    output: dict[int, float] = {}
    anchor_value = float(near_term[anchor_year])
    for year in MODEL_YEARS:
        if year <= anchor_year:
            output[year] = float(near_term[year])
        elif year < convergence_year:
            weight = (year - anchor_year) / (
                convergence_year - anchor_year
            )
            output[year] = anchor_value + weight * (
                long_run_value - anchor_value
            )
        else:
            output[year] = float(long_run_value)
    return output
