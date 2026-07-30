#!/usr/bin/env python3
"""Generate the teacher-free method and verified four-model result figures.

Figure contract
---------------
Core conclusion: a randomly initialized, teacher-free physical-anchor flow
improves over direct lattice regression, while teacher guidance gives a small
cost advantage and continuity-aware selection can stabilize closed loop.
All quantitative panels use the complete 20,000-query result files without
filtering. Exports use a 183-mm double-column width, editable vector text, a
600-dpi TIFF, and a 300-dpi preview. Backend: Python/matplotlib only.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

import teacher_free_figure_impl as impl
import make_real_flight_figure as real_flight


width_mm = 183
OUT = impl.OUT

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def save_publication_bundle(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    """Export editable SVG/PDF, submission TIFF, and screen preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def main() -> None:
    # The implementation module contains only panel construction. Override its
    # exporter here so this independently auditable entry point owns all output.
    impl.save_pub_py = save_publication_bundle
    impl.width_mm = width_mm
    impl.make_method_figure()
    impl.make_results_figure()
    real_flight.main()
    print(f"Teacher-free figures written to {OUT}")


if __name__ == "__main__":
    main()
