from pathlib import Path

from us_demography.scenarios import run_revised_scenarios


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"


def test_revised_headline_results_are_reproduced():
    bundle = run_revised_scenarios(DATA)
    summary = bundle.timeseries.set_index(["scenario", "year"])

    baseline = summary.loc[
        ("Census principal rebased para 2025", 2100), "population"
    ]
    low_cbo = summary.loc[
        ("NIM 574 mil; fecundidade CBO", 2100), "population"
    ]
    zero_cbo = summary.loc[
        ("Entrada estrangeira zero; fecundidade CBO", 2100),
        "population",
    ]

    assert 369e6 < baseline < 372e6
    assert 315e6 < low_cbo < 317e6
    assert 224e6 < zero_cbo < 226e6
