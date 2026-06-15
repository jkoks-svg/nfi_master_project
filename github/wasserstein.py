# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 02:02:12 2026

@author: jortk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyse_wasserstein_column(
    wasserstein_column,
    output_dir=None,
    filename="wasserstein_distance_histogram.png",
    bins=30,
    show=True,
    save=False,
):
    """
    Analyse a column/Series/array containing Wasserstein distances.

    Parameters
    ----------
    wasserstein_column : array-like
        Column from percentile_df, for example:
        percentile_df["wasserstein_log10_LR"]

    output_dir : str or pathlib.Path, optional
        Directory where the histogram is saved.
        If None, the plot is only shown and not saved unless save=False.

    filename : str
        Filename for the saved histogram.

    bins : int
        Number of histogram bins.

    show : bool
        Whether to show the histogram.

    save : bool
        Whether to save the histogram.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame containing summary statistics.
    """

    values = pd.Series(wasserstein_column).astype(float)
    values = values[np.isfinite(values)]

    if values.empty:
        raise ValueError("No finite Wasserstein distance values found.")

    summary_df = pd.DataFrame([{
        "n": len(values),
        "mean": values.mean(),
        "variance": values.var(ddof=1),
        "std": values.std(ddof=1),
        "min": values.min(),
        "q25": values.quantile(0.25),
        "median": values.median(),
        "q75": values.quantile(0.75),
        "max": values.max(),
    }])

    print("\nWasserstein distance summary")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black")

    plt.axvline(values.mean(), linestyle="--", linewidth=2,
                label=f"Mean = {values.mean():.3f}")
    plt.axvline(values.median(), linestyle=":", linewidth=2,
                label=f"Std = {values.std(ddof=1):.3f}")

    plt.xlabel("Wasserstein distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Wasserstein distances")
    plt.legend()
    plt.tight_layout()

    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save=True.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig_path = output_dir / filename
        plt.savefig(fig_path, dpi=300)
        print(f"[INFO] Saved Wasserstein histogram -> {fig_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return summary_df