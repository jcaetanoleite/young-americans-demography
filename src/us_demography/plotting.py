from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(
    timeseries: pd.DataFrame,
    frontier: pd.DataFrame,
    china_targets: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> pd.DataFrame:
        return timeseries[timeseries["scenario"] == name].sort_values("year")

    baseline = get("Census principal rebased para 2025")
    restriction_census = get("NIM 574 mil; fecundidade Census")
    restriction_cbo = get("NIM 574 mil; fecundidade CBO")
    zero_census = get("Entrada estrangeira zero; fecundidade Census")
    zero_cbo = get("Entrada estrangeira zero; fecundidade CBO")

    china_like_names = [
        name
        for name in timeseries["scenario"].unique()
        if name.startswith("China-like NIM zero")
    ]
    china_low = get(china_like_names[0])
    china_high = get(china_like_names[1])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11,
        }
    )

    def clean(axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)

    # Population
    fig, axis = plt.subplots(figsize=(9.3, 5.8))
    axis.plot(
        baseline["year"],
        baseline["population"] / 1e6,
        linewidth=2.2,
        label="Census principal",
    )
    axis.fill_between(
        restriction_cbo["year"],
        np.minimum(
            restriction_cbo["population"].to_numpy(),
            restriction_census["population"].to_numpy(),
        )
        / 1e6,
        np.maximum(
            restriction_cbo["population"].to_numpy(),
            restriction_census["population"].to_numpy(),
        )
        / 1e6,
        alpha=0.22,
        label="NIM 574 mil",
    )
    axis.fill_between(
        zero_cbo["year"],
        np.minimum(
            zero_cbo["population"].to_numpy(),
            zero_census["population"].to_numpy(),
        )
        / 1e6,
        np.maximum(
            zero_cbo["population"].to_numpy(),
            zero_census["population"].to_numpy(),
        )
        / 1e6,
        alpha=0.18,
        label="Entrada estrangeira zero",
    )
    axis.fill_between(
        china_low["year"],
        np.minimum(
            china_low["population"].to_numpy(),
            china_high["population"].to_numpy(),
        )
        / 1e6,
        np.maximum(
            china_low["population"].to_numpy(),
            china_high["population"].to_numpy(),
        )
        / 1e6,
        alpha=0.20,
        label="China-like, NIM zero",
    )
    axis.set_title("População dos Estados Unidos sob cenários demográficos")
    axis.set_xlabel("Ano")
    axis.set_ylabel("Milhões de habitantes")
    axis.legend(frameon=False)
    clean(axis)
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_1_population.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Old-age dependency
    fig, axis = plt.subplots(figsize=(9.3, 5.8))
    axis.plot(
        baseline["year"],
        baseline["old_age_dependency"],
        linewidth=2.2,
        label="Census principal",
    )
    axis.fill_between(
        restriction_cbo["year"],
        np.minimum(
            restriction_cbo["old_age_dependency"].to_numpy(),
            restriction_census["old_age_dependency"].to_numpy(),
        ),
        np.maximum(
            restriction_cbo["old_age_dependency"].to_numpy(),
            restriction_census["old_age_dependency"].to_numpy(),
        ),
        alpha=0.22,
        label="NIM 574 mil",
    )
    axis.fill_between(
        zero_cbo["year"],
        np.minimum(
            zero_cbo["old_age_dependency"].to_numpy(),
            zero_census["old_age_dependency"].to_numpy(),
        ),
        np.maximum(
            zero_cbo["old_age_dependency"].to_numpy(),
            zero_census["old_age_dependency"].to_numpy(),
        ),
        alpha=0.18,
        label="Entrada estrangeira zero",
    )
    axis.fill_between(
        china_low["year"],
        np.minimum(
            china_low["old_age_dependency"].to_numpy(),
            china_high["old_age_dependency"].to_numpy(),
        ),
        np.maximum(
            china_low["old_age_dependency"].to_numpy(),
            china_high["old_age_dependency"].to_numpy(),
        ),
        alpha=0.20,
        label="EUA China-like",
    )
    axis.plot(
        china_targets["Year"],
        china_targets["oadr"],
        linestyle="--",
        linewidth=2.0,
        label="China, ONU/WPP",
    )
    axis.set_title("Razão de dependência de idosos")
    axis.set_xlabel("Ano")
    axis.set_ylabel("Pessoas 65+ por 100 pessoas de 15–64")
    axis.legend(frameon=False)
    clean(axis)
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_2_old_age_dependency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Births
    fig, axis = plt.subplots(figsize=(9.3, 5.8))
    for frame, label, style in [
        (baseline, "Census principal", "-"),
        (restriction_cbo, "NIM 574 mil", "--"),
        (zero_cbo, "Entrada estrangeira zero", ":"),
        (china_low, "China-like, NIM zero", "-."),
    ]:
        selected = frame[frame["year"] >= 2026]
        axis.plot(
            selected["year"],
            selected["births"] / 1e6,
            linestyle=style,
            linewidth=2.1,
            label=label,
        )
    axis.set_title("Nascimentos anuais")
    axis.set_xlabel("Ano")
    axis.set_ylabel("Milhões de nascimentos")
    axis.legend(frameon=False)
    clean(axis)
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_3_births.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Frontier
    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    axis.plot(
        frontier["long_run_TFR"],
        frontier["long_run_net_migration"] / 1e6,
        marker="o",
        linewidth=2.0,
    )
    for _, row in frontier.iterrows():
        axis.annotate(
            str(int(row["convergence_year"])),
            (
                row["long_run_TFR"],
                row["long_run_net_migration"] / 1e6,
            ),
            xytext=(4, 5),
            textcoords="offset points",
        )
    zero_tfr = sorted(
        [
            float(name.split("TFR ")[1])
            for name in china_like_names
        ]
    )
    axis.axhline(0, linewidth=0.8)
    axis.axvspan(zero_tfr[0], zero_tfr[1], alpha=0.20)
    axis.set_title(
        "Fronteira fecundidade–migração para um cenário China-like"
    )
    axis.set_xlabel("Taxa de fecundidade total de longo prazo")
    axis.set_ylabel("NIM de longo prazo (milhões por ano)")
    clean(axis)
    fig.tight_layout()
    fig.savefig(
        output_dir / "figure_4_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
