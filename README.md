# Young Americans

A reproducible cohort-component model of United States population dynamics through 2100, focused on immigration, fertility, population aging, and comparisons with the demographic trajectory projected for China.

The model follows the U.S. population annually by sex and single year of age, from age 0 to 100+, using births, survival ratios, and age-sex-specific net international migration. It combines data from the U.S. Census Bureau, the Congressional Budget Office, and the United Nations World Population Prospects 2024.

## Research questions

The repository examines three counterfactuals:

1. What happens to the U.S. population if net international migration remains permanently near 574,000 people per year?
2. What happens if new foreign-born immigration falls to zero?
3. Which combinations of long-run fertility and net migration would make the U.S. age structure in 2100 resemble the one projected for China?

## Main results

Under the assumptions implemented here:

- the rebased Census main scenario produces a U.S. population of about 370 million in 2100;
- permanent net international migration of 574,000 produces 316–328 million;
- zero new foreign-born immigration produces 225–235 million;
- matching China's projected 2100 age structure with long-run net migration equal to zero requires a total fertility rate near 1.12;
- that China-like scenario produces a U.S. population near 195 million and about 830,000 births in 2100.

These are conditional demographic scenarios, not official forecasts or probabilistic prediction intervals.

## Model

Let \(P_{a,s,t}\) denote the population of age \(a\), sex \(s\), in year \(t\). For ages below the open-ended group,

\[
P_{a+1,s,t+1}
=
S_{a,s,t}P_{a,s,t}
+
M_{a+1,s,t+1},
\]

where \(S_{a,s,t}\) is the survival ratio and \(M_{a+1,s,t+1}\) is age-sex-specific net international migration.

Births are

\[
B_t
=
\sum_{a=14}^{54}
f_{a,t}P_{a,F,t},
\]

where \(f_{a,t}\) is the age-specific fertility rate and \(P_{a,F,t}\) is the female population at age \(a\).

The aggregate population identity is

\[
N_{t+1}-N_t
=
B_t-D_t+NIM_t.
\]

The model also computes the old-age dependency ratio,

\[
OADR_t
=
100
\frac{P_{65+,t}}{P_{15-64,t}},
\]

and the total dependency ratio,

\[
TDR_t
=
100
\frac{P_{0-14,t}+P_{65+,t}}
{P_{15-64,t}}.
\]

## Quasi-Monte Carlo sensitivity analysis

The optional sensitivity exercise uses 2,048 Sobol draws over long-run fertility, long-run net migration, and convergence timing. Sobol sequences cover the parameter space more evenly than ordinary pseudo-random draws.

The parameter ranges are analyst-defined stress-test ranges. The resulting distributions are sensitivity results, not posterior distributions, confidence intervals, or official probabilistic forecasts.

## Repository structure

```text
src/us_demography/
    model.py          cohort-component engine
    scenarios.py      baseline and counterfactual scenarios
    qmc.py            Sobol quasi-Monte Carlo sensitivity analysis
    plotting.py       figure generation

scripts/
    run_scenarios.py
    run_qmc_sensitivity.py
    make_figures.py

tests/
    accounting and regression tests

data/raw/
    input data

data/processed/
    reproduced scenario outputs

outputs/figures/
    generated figures
```

## Installation

Python 3.11 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproduction

Run the main scenarios:

```bash
python scripts/run_scenarios.py
```

Generate the figures:

```bash
python scripts/make_figures.py
```

Run the optional Sobol sensitivity exercise:

```bash
python scripts/run_qmc_sensitivity.py --draws 2048
```

Run the tests:

```bash
pytest -q
```

Or reproduce the default workflow with:

```bash
make all
```

## Data sources

- U.S. Census Bureau, 2023 National Population Projections;
- U.S. Census Bureau, Vintage 2025 National Population Estimates;
- Congressional Budget Office, *The Demographic Outlook: 2026 to 2056*;
- United Nations, *World Population Prospects 2024*.

Additional source details are available in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Limitations

The model does not distinguish migrants by country of origin, education, visa category, employment status, or duration of stay. It does not include endogenous wages, capital accumulation, public finances, retirement choices, or behavioral fertility responses. Aggregate net international migration is therefore a demographic input, not a complete representation of immigration policy.

## Reproducibility

The main outputs are regenerated from the raw data by the scripts in this repository. The test suite checks population accounting, non-negativity, reproduction of official Census transitions, and the headline counterfactual results.

## License

The code is released under the MIT License. The original datasets remain subject to the terms of their respective providers.
