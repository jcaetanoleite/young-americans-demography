from pathlib import Path

import numpy as np

from us_demography.model import MODEL_YEARS, CohortComponentModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"


def test_initial_population_is_positive():
    model = CohortComponentModel(DATA)
    assert model.actual_2025.shape == (2, 101)
    assert model.actual_2025.sum() > 300_000_000


def test_projection_is_nonnegative():
    model = CohortComponentModel(DATA)
    nim = {
        year: float(model.nim_scenarios["main"].loc[year])
        for year in MODEL_YEARS
    }
    result = model.project(nim, model.asfr_base)
    assert np.isfinite(result.population).all()
    assert (result.population >= 0).all()


def test_main_scenario_reproduces_official_transitions_when_not_rebased():
    model = CohortComponentModel(DATA)
    nim = {
        year: float(model.nim_scenarios["main"].loc[year])
        for year in MODEL_YEARS
    }
    official_initial = model.population_scenarios["main"][2025 - 2022]
    result = model.project(
        nim,
        model.asfr_base,
        initial_population=official_initial,
    )
    official = model.population_scenarios["main"][2025 - 2022 :]
    assert np.max(np.abs(result.population - official)) < 1e-6
