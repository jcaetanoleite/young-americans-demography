.PHONY: install scenarios qmc figures test all clean

install:
	python -m pip install -e ".[dev]"

scenarios:
	python scripts/run_scenarios.py

qmc:
	python scripts/run_qmc_sensitivity.py --draws 2048

figures:
	python scripts/make_figures.py

test:
	pytest -q

all: scenarios figures test

clean:
	rm -f data/processed/qmc_sensitivity.csv
