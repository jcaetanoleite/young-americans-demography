from pathlib import Path
import json

from us_demography.scenarios import run_revised_scenarios


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

bundle = run_revised_scenarios(DATA)
bundle.timeseries.to_csv(
    OUTPUT / "scenario_timeseries_revised.csv", index=False
)
bundle.frontier.to_csv(
    OUTPUT / "china_like_frontier_revised.csv", index=False
)
with open(OUTPUT / "scenario_parameters.json", "w", encoding="utf-8") as file:
    json.dump(bundle.parameters, file, indent=2, ensure_ascii=False)

selected = bundle.timeseries[
    bundle.timeseries["year"].isin([2025, 2050, 2056, 2100])
].copy()
selected["population_millions"] = selected["population"] / 1e6
selected["births_millions"] = selected["births"] / 1e6
selected.to_csv(OUTPUT / "scenario_summary_revised.csv", index=False)

print("Scenario outputs written to", OUTPUT)
