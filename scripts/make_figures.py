from pathlib import Path
import pandas as pd

from us_demography.plotting import make_figures


ROOT = Path(__file__).resolve().parents[1]
processed = ROOT / "data" / "processed"

make_figures(
    pd.read_csv(processed / "scenario_timeseries_revised.csv"),
    pd.read_csv(processed / "china_like_frontier_revised.csv"),
    pd.read_csv(ROOT / "data" / "raw" / "china_un_wpp_2024_targets.csv"),
    ROOT / "outputs" / "figures",
)
print("Figures written to", ROOT / "outputs" / "figures")
