from pathlib import Path
import argparse

from us_demography.qmc import run_qmc_sensitivity


parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=2048)
parser.add_argument("--seed", type=int, default=20260805)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "data" / "processed" / "qmc_sensitivity.csv"

draws = run_qmc_sensitivity(
    ROOT / "data" / "raw",
    draws=args.draws,
    seed=args.seed,
)
draws.to_csv(output, index=False)
print("QMC sensitivity output written to", output)
print(
    "Warning: these are design-based sensitivity draws, "
    "not probabilistic prediction intervals."
)
