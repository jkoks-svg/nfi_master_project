# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:09:23 2026

@author: jortk
"""



import ast
import json
import re
import time
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
#from confidence import Configuration
from scipy.special import logsumexp, xlogy

import sys
#from pathlib import Path

PYTHON_SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(PYTHON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SCRIPTS_DIR))

from helpers.euroformix_gamma_model import log_likelihood_locus
from helpers.genotypes import (
    build_frequency_dict,
    generate_genotypes,
    build_genotypes,
    build_observed_alleles_from_dict,
    genotype_prior,
    canonical_allele_label,
)
from helpers.mcmc_parameters import (
    run_mcmc_for_parameters,
    run_mcmc_for_rework_parameters,
)
from pipeline_config import (
    ANALYSIS_MIXTURES_DIR,
    MCMC_CONFIG_YAML,
    MCMC_REWORK_OUTPUT_ROOT,
    MCMC_SINGLE_OUTPUT_ROOT,
    RESOURCES_PATH,
    combined_params_json,
    combined_reference_csv,
    reference_csv,
    rework_candidate_csv,
    single_candidate_csv,
    single_params_json,
)



def safe_log(series):
    arr = np.asarray(series, dtype=float)
    return np.where(arr > 0, np.log(arr), -np.inf)

def get_rework_mixture(mixture: str) -> str:
    return mixture.split("_", 1)[1] if "_" in mixture else mixture


def detect_allele_columns_for_filling(df):
    allele_cols = [c for c in df.columns if c.startswith("Allele")]
    height_cols = [c for c in df.columns if c.startswith("Height")]

    allele_cols = sorted(allele_cols, key=lambda x: int(x.split()[-1]))
    height_cols = sorted(height_cols, key=lambda x: int(x.split()[-1]))

    return allele_cols, height_cols


# def mixture_row_to_allele_dict(row, allele_cols, height_cols):
#     d = {}

#     for a_col, h_col in zip(allele_cols, height_cols):
#         allele = str(row[a_col]).strip()
#         height = str(row[h_col]).strip()

#         if allele != "":
#             d[allele] = height if height != "" else "0"

#     return d
def mixture_row_to_allele_dict(row, allele_cols, height_cols):
    d = {}

    for a_col, h_col in zip(allele_cols, height_cols):
        allele = str(row.get(a_col, "")).strip()
        height = str(row.get(h_col, "")).strip()

        if allele != "":
            d[allele] = height if height != "" else "0"

    return d

def sort_allele_for_filling(x):
    try:
        return float(x)
    except ValueError:
        return x


def rebuild_filled_mixture_row(sample_name, marker, allele_dict, max_slots):
    row = {
        "Sample Name": sample_name,
        "Marker": marker,
    }

    alleles = sorted(allele_dict.keys(), key=sort_allele_for_filling)

    for i in range(max_slots):
        a_col = f"Allele {i + 1}"
        h_col = f"Height {i + 1}"

        if i < len(alleles):
            allele = alleles[i]
            row[a_col] = allele
            row[h_col] = allele_dict[allele]
        else:
            row[a_col] = ""
            row[h_col] = ""

    return row


def fill_missing_alleles_in_replicate_dfs(dfs):
    """
    Fill missing alleles across three replicate mixture dataframes.

    If an allele occurs in one replicate at a marker but is missing
    in another replicate, the missing replicate gets that allele with
    height 0.

    No files or folders are created.
    """
    if len(dfs) != 3:
        raise ValueError(f"Expected exactly 3 replicate dataframes, got {len(dfs)}.")

    dfs = [df.copy().fillna("").astype(str) for df in dfs]

    # allele_cols, height_cols = detect_allele_columns_for_filling(dfs[0])
    # markers = dfs[0]["Marker"].tolist()
    
    allele_cols = sorted(
        {
            c
            for df in dfs
            for c in df.columns
            if c.startswith("Allele")
        },
        key=lambda x: int(x.split()[-1]),
    )
    
    height_cols = sorted(
        {
            c
            for df in dfs
            for c in df.columns
            if c.startswith("Height")
        },
        key=lambda x: int(x.split()[-1]),
    )
    
    markers = dfs[0]["Marker"].tolist()

    filled_rows_by_replicate = [[] for _ in range(3)]

    for marker in markers:
        rows = []

        for df in dfs:
            match = df[df["Marker"] == marker]

            if match.empty:
                raise ValueError(f"Marker {marker} is missing in one replicate.")

            rows.append(match.iloc[0])

        sample_names = [row["Sample Name"] for row in rows]

        allele_maps = [
            mixture_row_to_allele_dict(row, allele_cols, height_cols)
            for row in rows
        ]

        all_alleles = set()
        for d in allele_maps:
            all_alleles.update(d.keys())

        for d in allele_maps:
            for allele in all_alleles:
                if allele not in d:
                    d[allele] = "0"

        max_slots = max(len(allele_cols), len(all_alleles))

        for i in range(3):
            filled_rows_by_replicate[i].append(
                rebuild_filled_mixture_row(
                    sample_name=sample_names[i],
                    marker=marker,
                    allele_dict=allele_maps[i],
                    max_slots=max_slots,
                )
            )

    return [pd.DataFrame(rows) for rows in filled_rows_by_replicate]


def load_and_fill_rework_replicates(rework_mixture, mixture_folder):
    """
    Read 1_x, 2_x, 3_x from the original mixture folder and return
    filled dataframes in memory.
    """
    replicate_names = [
        f"{replicate}_{rework_mixture}"
        for replicate in [1, 2, 3]
    ]

    replicate_dfs = [
        pd.read_csv(
            mixture_folder / f"{mixture}.txt",
            sep="\t",
            dtype=str,
        ).fillna("")
        for mixture in replicate_names
    ]

    filled_dfs = fill_missing_alleles_in_replicate_dfs(replicate_dfs)

    return dict(zip(replicate_names, filled_dfs))


def allele_to_repeat_length(allele):
    """
    Converts allele labels like:
    10   -> 10.0
    10.1 -> 10.25
    10.2 -> 10.50
    10.3 -> 10.75

    Assumes the decimal part represents extra nucleotides
    out of a 4-nucleotide repeat.
    """
    allele_str = str(allele)

    if "." not in allele_str:
        return float(allele_str)

    whole, extra_nt = allele_str.split(".")
    return float(whole) + int(extra_nt) / 4

def avg_phi_allele_getter(observed_alleles, allele_freq_dict, locus):
    observed = {
        str(a)
        for a in observed_alleles.get(locus, {}).keys()
    }

    remaining = {
        allele: freq
        for allele, freq in allele_freq_dict.items()
        if str(allele) not in observed
    }

    if not remaining:
        raise ValueError(
            f"No alleles left after removing observed alleles for locus {locus}."
        )

    total_freq = sum(remaining.values())

    avg_phi_allele = sum(
        allele_to_repeat_length(allele) * (freq / total_freq)
        for allele, freq in remaining.items()
    )

    return avg_phi_allele


def plot_mcmc_traces(mcmc_result, mixture, output_folder=None):
    samples = mcmc_result["samples"]

    params_to_plot = ["mu", "sigma", "beta", "phi1", "phi2"]

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    for param in params_to_plot:
        values = samples[param]

        plt.figure(figsize=(10, 4))
        plt.plot(values, linewidth=0.8)
        plt.xlabel("Posterior sample index")
        plt.ylabel(param)
        plt.title(f"MCMC trace for {param} - {mixture}")
        plt.tight_layout()

        if output_folder is None:
            plt.show()
        else:
            out_file = output_folder / f"{mixture}_trace_{param}.png"
            plt.savefig(out_file, dpi=200)
            plt.close()

            print(f"Saved trace plot: {out_file}")
            
def plot_mcmc_empirical_distributions(mcmc_result, mixture, output_folder=None):
    params_to_plot = ["mu", "sigma", "beta", "phi1", "phi2"]

    samples = mcmc_result["samples"]  # already after burn-in and thinning

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    for param in params_to_plot:
        values = np.asarray(samples[param], dtype=float)

        plt.figure(figsize=(8, 4))
        plt.hist(values, bins=30, density=True, alpha=0.75)

        mean_val = np.mean(values)
        median_val = np.median(values)
        ci_low, ci_high = np.quantile(values, [0.025, 0.975])

        plt.axvline(mean_val, linestyle="--", linewidth=1, label=f"mean = {mean_val:.4g}")
        plt.axvline(median_val, linestyle=":", linewidth=1, label=f"median = {median_val:.4g}")
        plt.axvline(ci_low, linestyle="-.", linewidth=1, label=f"2.5% = {ci_low:.4g}")
        plt.axvline(ci_high, linestyle="-.", linewidth=1, label=f"97.5% = {ci_high:.4g}")

        plt.xlabel(param)
        plt.ylabel("Density")
        plt.title(f"Empirical posterior distribution for {param} - {mixture}")
        plt.legend()
        plt.tight_layout()

        if output_folder is None:
            plt.show()
        else:
            out_file = output_folder / f"{mixture}_posterior_distribution_{param}.png"
            plt.savefig(out_file, dpi=200)
            plt.close()
            print(f"Saved posterior distribution plot: {out_file}")
   
    
def plot_mcmc_parameter_correlations(
    mcmc_result,
    mixture,
    output_folder=None,
    params_to_plot=None,
):
    """
    Plots pairwise correlations between MCMC parameter samples.

    Creates:
    1. A correlation heatmap
    2. Pairwise scatter plots between all model parameters

    Parameters
    ----------
    mcmc_result : dict
        Output dictionary from your MCMC function.
        Must contain mcmc_result["samples"].

    mixture : str
        Name of the mixture, used in plot titles and filenames.

    output_folder : str or Path, optional
        If None, plots are shown.
        If given, plots are saved to this folder.

    params_to_plot : list[str], optional
        Parameters to include. If None, defaults to:
        ["mu", "sigma", "beta", "phi1", "phi2"].
    """

    samples = mcmc_result["samples"]

    if params_to_plot is None:
        params_to_plot = ["mu", "sigma", "beta", "phi1", "phi2"]

    # Keep only parameters that are actually present
    params_to_plot = [p for p in params_to_plot if p in samples]

    if len(params_to_plot) < 2:
        raise ValueError("Need at least two parameters to compute correlations.")

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Convert samples to dataframe
    df = pd.DataFrame({
        param: np.asarray(samples[param], dtype=float)
        for param in params_to_plot
    })

    # ------------------------------------------------------------
    # 1. Correlation heatmap
    # ------------------------------------------------------------
    corr = df.corr()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, vmin=-1, vmax=1)

    plt.colorbar(im, label="Pearson correlation")
    plt.xticks(range(len(params_to_plot)), params_to_plot, rotation=45)
    plt.yticks(range(len(params_to_plot)), params_to_plot)

    # Add correlation numbers inside cells
    for i in range(len(params_to_plot)):
        for j in range(len(params_to_plot)):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.title(f"MCMC parameter correlation matrix - {mixture}")
    plt.tight_layout()

    if output_folder is None:
        plt.show()
    else:
        out_file = output_folder / f"{mixture}_parameter_correlation_heatmap.png"
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f"Saved parameter correlation heatmap: {out_file}")

    # ------------------------------------------------------------
    # 2. Pairwise scatter plots
    # ------------------------------------------------------------
    n_params = len(params_to_plot)

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(3 * n_params, 3 * n_params),
    )

    for i, y_param in enumerate(params_to_plot):
        for j, x_param in enumerate(params_to_plot):
            ax = axes[i, j]

            if i == j:
                ax.hist(df[x_param], bins=30, density=True, alpha=0.75)
                ax.set_ylabel("Density")
            else:
                ax.scatter(
                    df[x_param],
                    df[y_param],
                    s=6,
                    alpha=0.35,
                )

                r = corr.loc[y_param, x_param]
                ax.set_title(f"r = {r:.2f}", fontsize=9)

            if i == n_params - 1:
                ax.set_xlabel(x_param)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(y_param)
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"Pairwise MCMC parameter correlations - {mixture}", y=1.02)
    plt.tight_layout()

    if output_folder is None:
        plt.show()
    else:
        out_file = output_folder / f"{mixture}_parameter_pairwise_correlations.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved pairwise parameter correlation plot: {out_file}")
   
    



# def joint_deconvolution_locus(
#     params_lst,
#     observed_alleles,
#     locus,
#     all_alleles,
#     threshold,
#     freq_dict,
#     RARE_FREQ,
#     N_contributors,
#     kit_properties_df,
#     candidate_pairs_by_locus=None,
# ):
#     observed_sets = build_observed_alleles_from_dict(observed_alleles)

#     results = []
#     allele_freq_dict = freq_dict[locus]
#     avg_phi_allele = avg_phi_allele_getter(observed_alleles, allele_freq_dict, locus)
#     prior_cache = {}

#     if candidate_pairs_by_locus is not None:
#         if N_contributors != 2:
#             raise NotImplementedError(
#                 "Candidate-pair filtering is currently implemented only for 2 contributors."
#             )

#         genotype_tuples = candidate_pairs_by_locus.get(locus)

#         if not genotype_tuples:
#             raise ValueError(
#                 f"No candidate genotype pairs found for locus {locus}. "
#                 "Check the candidate CSV."
#             )

#     else:
#         alleles2 = [
#             int(a) if float(a).is_integer() else float(a)
#             for a in (canonical_allele_label(x) for x in observed_sets[locus])
#         ]

#         genos = generate_genotypes(alleles2)
#         genotype_tuples = product(genos, repeat=N_contributors)   
    
   
    
#     for g_tuple in genotype_tuples:

#         genotypes = build_genotypes(g_tuple, locus)


#         prior = 1.0
#         for g in g_tuple:
#             if g not in prior_cache:
#                 prior_cache[g] = genotype_prior(
#                     g, locus, freq_dict, observed_sets, RARE_FREQ
#                 )
        
#             prior *= prior_cache[g]

#         # likelihood averaging over parameter sets
#         log_likelihoods = []
#         #likelihoods = []
#         for params in params_lst:
#             logL, dropin_set_size = log_likelihood_locus(
#                 params,
#                 observed_alleles,
#                 genotypes,
#                 locus,
#                 all_alleles,
#                 threshold,
#                 allele_freq_dict,
#                 RARE_FREQ,
#                 kit_properties_df,
#                 avg_phi_allele
#             )
#             log_likelihoods.append(float(logL))
#             #likelihoods.append(float(np.exp(logL)))
        
#         #safe averaging over very small likelihoods
#         log_likelihoods = np.array(log_likelihoods, dtype=float)
#         log_avg_L = logsumexp(log_likelihoods) - np.log(len(log_likelihoods))
#         avg_L = float(np.exp(log_avg_L))
#         posterior_raw = avg_L * prior
        
#         # avg_L = np.mean(likelihoods)
#         # posterior_raw = avg_L * prior

#         # build dynamic result dict
#         result_entry = {
#             "Locus": locus,
#         }

#         # add contributors dynamically
#         for i, g in enumerate(g_tuple):
#             result_entry[f"Unknown{i+1}"] = g
        
#         result_entry["Prior"] = prior
#         #result_entry['Dropin_set_size'] = dropin_set_size
#         #result_entry["Likelihoods"] = likelihoods
        
#         result_entry["Posterior_unnormalized"] = posterior_raw
#         result_entry["Log_likelihoods"] = log_likelihoods
#         result_entry["Average_likelihood"] = avg_L

#         results.append(result_entry)

#     # normalize
#     total = sum(r["Posterior_unnormalized"] for r in results)

#     if total <= 0 or not np.isfinite(total):
#         raise ValueError(
#             f"Cannot normalize posterior probabilities for locus {locus}: "
#             f"total posterior mass is {total}."
#         )

#     for r in results:
#         r["Probability"] = r["Posterior_unnormalized"] / total
#         #r["Posterior_unnormalized2"] = r["Posterior_unnormalized"]

#     return results

def joint_deconvolution_locus(
    params_lst,
    observed_alleles,
    locus,
    all_alleles,
    threshold,
    freq_dict,
    RARE_FREQ,
    N_contributors,
    kit_properties_df,
    candidate_pairs_by_locus=None,
):
    observed_sets = build_observed_alleles_from_dict(observed_alleles)

    results = []
    allele_freq_dict = freq_dict[locus]
    avg_phi_allele = avg_phi_allele_getter(observed_alleles, allele_freq_dict, locus)
    prior_cache = {}

    if candidate_pairs_by_locus is not None:
        if N_contributors != 2:
            raise NotImplementedError(
                "Candidate-pair filtering is currently implemented only for 2 contributors."
            )

        genotype_tuples = candidate_pairs_by_locus.get(locus)

        if not genotype_tuples:
            raise ValueError(
                f"No candidate genotype pairs found for locus {locus}. "
                "Check the candidate CSV."
            )

    else:
        alleles2 = [
            int(a) if float(a).is_integer() else float(a)
            for a in (canonical_allele_label(x) for x in observed_sets[locus])
        ]

        genos = generate_genotypes(alleles2)
        genotype_tuples = product(genos, repeat=N_contributors)

    # ------------------------------------------------------------
    # STEP 1: compute log prior + log likelihood vector
    # for every genotype tuple
    # ------------------------------------------------------------
    for g_tuple in genotype_tuples:

        genotypes = build_genotypes(g_tuple, locus)

        prior = 1.0
        for g in g_tuple:
            if g not in prior_cache:
                prior_cache[g] = genotype_prior(
                    g, locus, freq_dict, observed_sets, RARE_FREQ
                )
            prior *= prior_cache[g]

        if prior <= 0 or not np.isfinite(prior):
            log_prior = -np.inf
        else:
            log_prior = np.log(prior)

        log_likelihoods = []

        for params in params_lst:
            logL, dropin_set_size = log_likelihood_locus(
                params,
                observed_alleles,
                genotypes,
                locus,
                all_alleles,
                threshold,
                allele_freq_dict,
                RARE_FREQ,
                kit_properties_df,
                avg_phi_allele,
            )
            log_likelihoods.append(float(logL))

        log_likelihoods = np.asarray(log_likelihoods, dtype=float)

        # This is the unnormalised posterior mass per MCMC draw:
        # log [ P(E_l | g_l, theta_m) * P(g_l) ]
        log_posterior_unnormalized_draws = log_prior + log_likelihoods

        result_entry = {
            "Locus": locus,
        }

        for i, g in enumerate(g_tuple):
            result_entry[f"Unknown{i+1}"] = g

        result_entry["Prior"] = prior

        # Keep these as arrays for now.
        # They will be converted to JSON strings before writing to CSV.
        result_entry["Log_likelihoods"] = log_likelihoods
        result_entry["Log_posterior_unnormalized_draws"] = log_posterior_unnormalized_draws

        # # Optional diagnostic only; do not use this for Bayesian deconvolution.
        # result_entry["Average_likelihood"] = float(
        #     np.exp(logsumexp(log_likelihoods) - np.log(len(log_likelihoods)))
        # )

        results.append(result_entry)

    if not results:
        raise ValueError(f"No genotype results produced for locus {locus}.")

    # ------------------------------------------------------------
    # STEP 2: normalize separately for each MCMC draw
    # ------------------------------------------------------------
    log_raw_matrix = np.vstack([
        r["Log_posterior_unnormalized_draws"]
        for r in results
    ])
    # shape: n_genotype_tuples x n_mcmc_draws

    log_denominators = logsumexp(log_raw_matrix, axis=0)
    # one denominator per MCMC draw

    bad = ~np.isfinite(log_denominators)
    if np.any(bad):
        bad_idx = np.where(bad)[0][:10]
        raise ValueError(
            f"Cannot normalize posterior probabilities for locus {locus}. "
            f"Non-finite denominator for MCMC draw indices: {bad_idx}"
        )

    posterior_prob_matrix = np.exp(log_raw_matrix - log_denominators)
    # shape: n_genotype_tuples x n_mcmc_draws

    # ------------------------------------------------------------
    # STEP 3: store vector posterior probabilities per genotype tuple
    # ------------------------------------------------------------
    for i, r in enumerate(results):
        posterior_draws = posterior_prob_matrix[i, :]

        # This is the important new column:
        # vector P_m(g_l | E_l, H2) for each MCMC draw m.
        r["Posterior_probability_draws"] = posterior_draws
        
        # Scalar used only for comparison with DNAStatistX
        log_post_draws = r["Log_posterior_unnormalized_draws"]
        
        r["Posterior_unnormalized"] = float(
            np.exp(
                logsumexp(log_post_draws) -
                np.log(len(log_post_draws))
            )
        )

        # # Keep scalar mean for compatibility with old downstream code.
        # r["Probability"] = float(np.mean(posterior_draws))

        # # Old-style scalar posterior mass is no longer conceptually correct
        # # for Bayes deconvolution, but this compatibility column can be useful.
        # r["Posterior_unnormalized"] = float(np.mean(np.exp(r["Log_posterior_unnormalized_draws"])))

        # # Useful diagnostic: should sum to 1 over genotype tuples after averaging.
        # r["Posterior_probability_sd"] = float(np.std(posterior_draws, ddof=1))

    return results

def extract_observed_alleles_and_thresholds(mixture_df, thresholds_map, frac_threshold, mode):
    """
    Combines:
    - mixture_df_to_dict
    - compute_thresholds

    Returns:
        observed_alleles: dict[locus -> {allele: height}]
        thresholds: dict[locus -> effective threshold]
    """

    observed_alleles = {}
    thresholds = {}

    for _, row in mixture_df.iterrows():
        locus = row["Marker"]

        allele_dict_raw = {}

        # # -------------------------
        # # STEP 1: collect peaks
        # # -------------------------
        # for i in range(1, 11):
        #     allele_col = f"Allele {i}"
        #     height_col = f"Height {i}"

        #     allele = row.get(allele_col)
        #     height = row.get(height_col)

        #     if pd.notna(allele) and pd.notna(height):
        #         allele_val = float(allele)
        #         if allele_val.is_integer():
        #             allele_val = int(allele_val)

        #         allele_dict_raw[allele_val] = float(height)
                # -------------------------
        # STEP 1: collect peaks
        # -------------------------
        # allele_cols = [
        #     c for c in mixture_df.columns
        #     if c.startswith("Allele")
        # ]

        # allele_cols = sorted(
        #     allele_cols,
        #     key=lambda x: int(x.split()[-1])
        # )

        # for allele_col in allele_cols:
        #     idx = allele_col.split()[-1]
        #     height_col = f"Height {idx}"

        #     allele = row.get(allele_col, "")
        #     height = row.get(height_col, "")

        #     # Skip truly empty allele slots
        #     if pd.isna(allele) or str(allele).strip() == "":
        #         continue

        #     # Skip empty height slots
        #     if pd.isna(height) or str(height).strip() == "":
        #         continue

        #     allele_val = float(str(allele).strip())
        #     if allele_val.is_integer():
        #         allele_val = int(allele_val)

        #     allele_dict_raw[allele_val] = float(str(height).strip())
        
        allele_cols = sorted(
            [c for c in mixture_df.columns if c.startswith("Allele")],
            key=lambda x: int(x.split()[-1])
        )
        
        for allele_col in allele_cols:
            idx = allele_col.split()[-1]
            height_col = f"Height {idx}"
        
            allele = row.get(allele_col, "")
            height = row.get(height_col, "")
        
            if pd.isna(allele) or str(allele).strip() == "":
                continue
        
            if pd.isna(height) or str(height).strip() == "":
                continue
        
            allele_val = float(str(allele).strip())
            if allele_val.is_integer():
                allele_val = int(allele_val)
        
            allele_dict_raw[allele_val] = float(str(height).strip())

        if not allele_dict_raw:
            continue

        # -------------------------
        # STEP 2: compute threshold
        # -------------------------
        T = thresholds_map[locus]
        max_peak = max(allele_dict_raw.values())
        frac_T = frac_threshold * max_peak
        effective_T = int(max(T, frac_T))

        thresholds[locus] = effective_T
        
        if mode == "single":
            allele_dict_filtered = {
                a: h for a, h in allele_dict_raw.items()
                if h >= effective_T
            }
        
        elif mode == "rework":
            allele_dict_filtered = {
                a: h if h >= effective_T else 0.0
                #allele is treated as observed, but its allele height will be mapped to zero later in log_likelihood_locus function
                #this is necessary for combined analysis (cleaner solutions are probably possible as well)
                for a, h in allele_dict_raw.items()
            }
        
        else:
            raise ValueError(f"Unknown mode={mode}. Use 'single' or 'rework'.")
            
        observed_alleles[locus] = allele_dict_filtered

        if allele_dict_filtered:
            observed_alleles[locus] = allele_dict_filtered
        else:
            # still store empty locus if needed (optional)
            observed_alleles[locus] = {}

    return observed_alleles, thresholds

#def create_joint_df(params_lst, observed_alleles, all_alleles, thresholds, freq_dict, RARE_FREQ, N_contributors, kit_properties_df):
def create_joint_df(
    params_lst,
    observed_alleles,
    all_alleles,
    thresholds,
    freq_dict,
    RARE_FREQ,
    N_contributors,
    kit_properties_df,
    candidate_pairs_by_locus=None,
):    
    all_results = []

    for locus in observed_alleles:

        res = joint_deconvolution_locus(
            params_lst,
            observed_alleles,
            locus,
            all_alleles[locus],
            thresholds[locus],
            freq_dict,
            RARE_FREQ,
            N_contributors,
            kit_properties_df,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
        )
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    return df #df.sort_values(by=["Locus", "Probability"], ascending=[True, False])

# def create_marginal_df_combined(joint_df, observed, freq_dict, RARE_FREQ):
    
#     #contributor_cols = [col for col in joint_df.columns if col.startswith("Unknown")]
#     contributor_cols = [
#         col for col in joint_df.columns
#         if re.fullmatch(r"Unknown\d+", col)
#     ]
    
#     dfs = []

#     for col in contributor_cols:
#         df_marg = (
#             joint_df
#             .groupby(["Locus", col], as_index=False)["Posterior_unnormalized"]
#             .sum()
#             .rename(columns={
#                 col: "Genotype",
#                 "Posterior_unnormalized": "Marginal_posterior_unnormalized"
#             })
#         )
        
#         df_marg["Contributor"] = col
#         dfs.append(df_marg)

#     # combine everything
#     marginal_df = pd.concat(dfs)

#     # NORMALIZE PER (Locus, Contributor)
#     marginal_df["Marginal_probability"] = (
#         marginal_df["Marginal_posterior_unnormalized"] /
#         marginal_df.groupby(["Locus", "Contributor"])["Marginal_posterior_unnormalized"].transform("sum")
#     )
    
#     observed_sets = build_observed_alleles_from_dict(observed)
    

#     marginal_df["Genotype"] = marginal_df["Genotype"].apply(
#         lambda x: x if isinstance(x, tuple) else ast.literal_eval(x)
#     )

#     marginal_df["Prior"] = marginal_df.apply(
#         lambda row: genotype_prior(
#             row["Genotype"],
#             row["Locus"],
#             freq_dict,
#             observed_sets,
#             RARE_FREQ
#         ),
#         axis=1
#     )
    
#     #from postprocess_add_priors_helper
#     col_F = marginal_df["Marginal_probability"]
#     col_G = marginal_df["Marginal_posterior_unnormalized"]
#     col_I = marginal_df["Prior"]

#     marginal_df["P(E|a)"] = col_G / col_I
#     marginal_df["LR(a)_1"] = col_F / col_I

#     group_cols = ["Contributor", "Locus"]
#     sum_G_by_marker = marginal_df.groupby(group_cols)[col_G.name].transform("sum")

#     marginal_df["LR(a)_2"] = marginal_df["P(E|a)"] / sum_G_by_marker
    
#     return marginal_df

def create_marginal_df_combined(joint_df, observed, freq_dict, RARE_FREQ):
    contributor_cols = [
        col for col in joint_df.columns
        if re.fullmatch(r"Unknown\d+", col)
    ]

    if "Posterior_probability_draws" not in joint_df.columns:
        raise ValueError(
            "joint_df must contain Posterior_probability_draws. "
            "Use the updated Bayesian joint_deconvolution_locus first."
        )

    dfs = []

    for col in contributor_cols:
        rows = []

        for (locus, genotype), group in joint_df.groupby(["Locus", col]):
            # Sum posterior probability vectors over all joint genotype tuples
            # containing this marginal genotype.
            posterior_matrix = np.vstack([
                np.asarray(x, dtype=float)
                for x in group["Posterior_probability_draws"]
            ])

            marginal_draws = posterior_matrix.sum(axis=0)

            # rows.append({
            #     "Locus": locus,
            #     "Genotype": genotype,
            #     "Contributor": col,

            #     # vector over MCMC draws
            #     "Marginal_probability_draws": marginal_draws,

            #     # scalar summary for old downstream use
            #     "Marginal_probability": float(np.mean(marginal_draws)),
            #     "Marginal_probability_sd": float(np.std(marginal_draws, ddof=1)),
            # })
            
            rows.append({
                "Locus": locus,
                "Genotype": genotype,
                "Contributor": col,
                "Marginal_probability_draws": marginal_draws,
            })

        dfs.append(pd.DataFrame(rows))

    marginal_df = pd.concat(dfs, ignore_index=True)

    observed_sets = build_observed_alleles_from_dict(observed)

    marginal_df["Genotype"] = marginal_df["Genotype"].apply(
        lambda x: x if isinstance(x, tuple) else ast.literal_eval(x)
    )

    marginal_df["Prior"] = marginal_df.apply(
        lambda row: genotype_prior(
            row["Genotype"],
            row["Locus"],
            freq_dict,
            observed_sets,
            RARE_FREQ,
        ),
        axis=1,
    )

    # For Bayesian posterior draws, LR(a) is also draw-specific:
    # LR_m(a) = P_m(a | E) / P(a)
    marginal_df["LR(a)_draws"] = marginal_df.apply(
        lambda row: (
            np.asarray(row["Marginal_probability_draws"], dtype=float) / row["Prior"]
            if row["Prior"] > 0
            else np.full_like(
                np.asarray(row["Marginal_probability_draws"], dtype=float),
                np.nan,
                dtype=float,
            )
        ),
        axis=1,
    )

    # marginal_df["LR(a)_1"] = marginal_df.apply(
    #     lambda row: (
    #         row["Marginal_probability"] / row["Prior"]
    #         if row["Prior"] > 0
    #         else np.nan
    #     ),
    #     axis=1,
    # )

    return marginal_df

def prepare_marginal_df_for_csv(df):
    df = df.copy()

    array_cols = [
        "Marginal_probability_draws",
        "LR(a)_draws",
    ]

    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(np.asarray(x, dtype=float).tolist())
            )

    return df



def compare_with_reference_csv(joint_df, csv_path,tol, mixture):
    """
    Adds reference likelihoods from CSV and compares log-likelihoods.

    Parameters
    ----------
    joint_df : pandas.DataFrame
        Output from your model (must contain Locus, Unknown1, Unknown2, Average_likelihood)

    csv_path : str or Path
        Path to reference CSV file

    tol : float
        Absolute tolerance for log-likelihood comparison

    Returns
    -------
    merged_df : pandas.DataFrame
        joint_df with added columns:
        - Reference_Likelihood
        - logL_model
        - logL_ref
        - logL_match
        - logL_diff
    """
    # -------------------------
    # Load CSV
    # -------------------------
    csv_df = pd.read_csv(csv_path, sep="\t")

    # Fix encoding issue (Ã˜ → Ø)
    csv_df = csv_df.replace("Ã˜", "Ø")

    # -------------------------
    # Helper functions
    # -------------------------
    def make_tuple(a, b):
        return (str(a), str(b))

    def normalize_genotype(g):
        a, b = g
        return tuple(sorted([str(a), str(b)]))

    # -------------------------
    # Convert CSV genotypes
    # -------------------------
    csv_df["Unknown1"] = csv_df.apply(
        lambda r: make_tuple(r["Unknown1-Allele1"], r["Unknown1-Allele2"]),
        axis=1
    )

    csv_df["Unknown2"] = csv_df.apply(
        lambda r: make_tuple(r["Unknown2-Allele1"], r["Unknown2-Allele2"]),
        axis=1
    )

    # -------------------------
    # Normalize both dataframes
    # -------------------------
    joint_df = joint_df.copy()

    joint_df["Unknown1"] = joint_df["Unknown1"].apply(normalize_genotype)
    joint_df["Unknown2"] = joint_df["Unknown2"].apply(normalize_genotype)

    csv_df["Unknown1"] = csv_df["Unknown1"].apply(normalize_genotype)
    csv_df["Unknown2"] = csv_df["Unknown2"].apply(normalize_genotype)

    # -------------------------
    # Merge
    # -------------------------
    merged = joint_df.merge(
        csv_df[["Locus", "Unknown1", "Unknown2", "Likelihood"]],
        on=["Locus", "Unknown1", "Unknown2"],
        how="left"
    )

    # # Rename for clarity
    # merged["mixture"] = mixture
    # merged["replicate"] = mixture[0]
    # merged["mixture_combined"] = get_rework_mixture(mixture)
    
    merged["mixture"] = mixture
    merged["replicate"] = mixture.split("_", 1)[0] if "_" in mixture else np.nan
    merged["mixture_combined"] = get_rework_mixture(mixture)
    
    merged = merged.rename(columns={
        "Likelihood": "Reference_posterior_unnormalized"
    })

    # -------------------------
    # Compute log likelihoods
    # -------------------------

    # merged["log_post_model"] = np.log(merged["Posterior_unnormalized"])
    # merged["log_post_ref"] = np.log(merged["Reference_posterior_unnormalized"])
    merged["log_post_model"] = xlogy(1, merged["Posterior_unnormalized"])
    merged["log_post_ref"] = xlogy(1, merged["Reference_posterior_unnormalized"])
    # merged["log_post_model"] = safe_log(merged["Posterior_unnormalized"])
    # merged["log_post_ref"] = safe_log(merged["Reference_posterior_unnormalized"])

    # -------------------------
    # Compare
    # -------------------------
    merged["log_post_match"] = np.isclose(
        merged["log_post_model"],
        merged["log_post_ref"],
        atol=tol
    )

    # Difference (VERY useful for debugging)
    merged["abs_log_post_diff"] = np.abs(merged["log_post_model"] - merged["log_post_ref"])


    return merged

# import ast
# import pandas as pd

def split_and_sort_genotype(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    a, b = x

    def convert(v):
        try:
            v_float = float(v)
            return int(v_float) if v_float.is_integer() else v_float
        except:
            return str(v)  # e.g. 'Ã˜'

    a = convert(a)
    b = convert(b)

    # sorting rule:
    # - numbers first
    # - then strings like 'Ã˜'
    def sort_key(v):
        return (isinstance(v, str), v)

    a, b = sorted([a, b], key=sort_key)

    return pd.Series([a, b])


def kit_properties_parser(kit_properties_txt):
    tree = ET.parse(kit_properties_txt)
    root = tree.getroot()

    rows = []

    for kit in root.findall(".//Kit"):
        current_kit_name = kit.get("name")

        for locus_element in kit.findall("Locus"):
            rows.append({
                "Kit": current_kit_name,
                "Locus": locus_element.get("name"),
                "period": locus_element.get("period"),
                "slope": float(locus_element.get("slope")) if locus_element.get("slope") else None,
                "offset": float(locus_element.get("offset")) if locus_element.get("offset") else None,
                "ignore": locus_element.get("ignore") == "true"
            })

    kit_properties_df = pd.DataFrame(rows)
    return kit_properties_df

def build_kit_properties_lookup(kit_properties_df):
    lookup = {}

    for _, row in kit_properties_df.iterrows():
        key = (row["Kit"], row["Locus"])

        lookup[key] = {
            "period": float(row["period"]) if pd.notna(row["period"]) else None,
            "slope": float(row["slope"]) if pd.notna(row["slope"]) else None,
            "offset": float(row["offset"]) if pd.notna(row["offset"]) else None,
            "ignore": bool(row["ignore"]),
        }

    return lookup

# def copy_results_json(results_json_path, output_folder, mixture_name):
#     """
#     Copy DNAStatistX results.json into the same output location as the CSV files.

#     Output:
#         output_folder / mixtures / mixture_name / results.json
#     """
#     if output_folder is None:
#         return

#     results_json_path = Path(results_json_path)
#     output_folder = Path(output_folder)

#     if not results_json_path.exists():
#         raise FileNotFoundError(f"results.json not found: {results_json_path}")

#     out_file = output_folder / "mixtures" / mixture_name / "results.json"
#     out_file.parent.mkdir(parents=True, exist_ok=True)

#     shutil.copy2(results_json_path, out_file)
#     print(f"Copied results.json to: {out_file}")
    
def copy_results_json(results_json_path, output_folder, mixture_name):
    """
    Copy DNAStatistX results.json into the same output location as the CSV files.

    Output:
        output_folder / mixtures / mixture_name / results_dnax.json
    """
    if output_folder is None:
        return

    results_json_path = Path(results_json_path)
    output_folder = Path(output_folder)

    if not results_json_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_json_path}")

    out_file = output_folder / "mixtures" / mixture_name / "results_dnax.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(results_json_path, out_file)
    print(f"Copied DNAStatistX results.json to: {out_file}")
    

def make_json_serializable(obj):
    """
    Convert numpy/pandas objects to normal Python objects so json.dump works.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_serializable(v)
            for v in obj
        ]

    return obj


def save_mcmc_results_json(mcmc_result, output_folder, mixture_name):
    """
    Save the MCMC chain and diagnostics.

    Output:
        output_folder / mixtures / mixture_name / results_mcmc.json
    """
    if output_folder is None:
        return

    if mcmc_result is None:
        return

    output_folder = Path(output_folder)

    out_file = output_folder / "mixtures" / mixture_name / "results_mcmc.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = make_json_serializable(mcmc_result)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved MCMC results to: {out_file}")

# def create_deconvolution_tables(mixture_file, 
#                                 params_lst, 
#                                 RESOURCES_PATH, 
#                                 CONFIG_YAML, 
#                                 csv_path, 
#                                 tol, 
#                                 mixture, 
#                                 mode,
#                                 candidate_pairs_by_locus=None,):
      
#     mixture_df = pd.read_csv(mixture_file, sep = '\t')
    
def create_deconvolution_tables(
    mixture_df,
    params_lst,
    RESOURCES_PATH,
    CONFIG_YAML,
    csv_path,
    tol,
    mixture,
    mode,
    candidate_pairs_by_locus=None,
):
    mixture_df = mixture_df.copy()
 
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    RARE_FREQ = float(cfg['RARE_FREQ'])
    H2_FILTER_THRESHOLD = float(cfg['H2_FILTER_THRESHOLD'])
    freqs_file = RESOURCES_PATH / cfg['dnastatistx']["frequencies_file"]
    N_contributors = int(cfg['dnastatistx']['N_CONTRIBUTORS'])
    
    # kit_properties_txt = RESOURCES_PATH / cfg['kit_properties_txt']
    # kit_properties_df = kit_properties_parser(kit_properties_txt)
    kit_properties_txt = RESOURCES_PATH / cfg['kit_properties_txt']
    kit_properties_df = kit_properties_parser(kit_properties_txt)
    kit_properties_df = build_kit_properties_lookup(kit_properties_df) #overrule kit_properties_df
    
    if len(params_lst[0][0])!= N_contributors:
        raise ValueError("N_contributors not equal to length of phi vector")
    
    #thresholds_cfg = Configuration(cfg['dnastatistx']["thresholds"])
    #thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    thresholds_map = dict(cfg["dnastatistx"]["thresholds"])
    frac_threshold = float(cfg['dnastatistx']['frac_threshold'].replace('*=', '').strip())
    
    observed_alleles, thresholds = extract_observed_alleles_and_thresholds(
        mixture_df,
        thresholds_map,
        frac_threshold,
        mode
    )

    freq_df = pd.read_csv(freqs_file, sep = ',')
    freq_dict = build_frequency_dict(freq_df)
        
    all_alleles = {}

    for locus, allele_dict in freq_dict.items():
        alleles = []
        for allele in allele_dict.keys():
            # convert to int if possible, else float
            if '.' in allele:
                alleles.append(float(allele))
            else:
                alleles.append(int(allele))
        
        # optional: sort for consistency
        alleles.sort()
        #if locus == 'D1S1656':
        all_alleles[locus] = alleles
    #print(all_alleles)

    #joint_df = create_joint_df(params_lst, observed_alleles, all_alleles, thresholds, freq_dict, RARE_FREQ, N_contributors, kit_properties_df)
    joint_df = create_joint_df(
        params_lst,
        observed_alleles,
        all_alleles,
        thresholds,
        freq_dict,
        RARE_FREQ,
        N_contributors,
        kit_properties_df,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )
    #joint_df = joint_df.sort_values(by=["Locus", "Probability"], ascending=[True, True])
    joint_df["_posterior_probability_mean"] = joint_df[
        "Posterior_probability_draws"
    ].apply(lambda x: float(np.mean(np.asarray(x, dtype=float))))
    
    joint_df = joint_df.sort_values(
        by=["Locus", "_posterior_probability_mean"],
        ascending=[True, False],
    )
    joint_df[["Unknown1-Allele1", "Unknown1-Allele2"]] = joint_df["Unknown1"].apply(split_and_sort_genotype)
    joint_df[["Unknown2-Allele1", "Unknown2-Allele2"]] = joint_df["Unknown2"].apply(split_and_sort_genotype)
    
    #joint_df_clean = joint_df[(joint_df['Probability']>= H2_FILTER_THRESHOLD)] 
                              #& (joint_df['Locus']=='CSF1PO')]
    # joint_df_clean = (
    #     joint_df.loc[
    #         joint_df["Probability"] >= H2_FILTER_THRESHOLD,
    #         [
    #             "Locus",
    #             "Unknown1-Allele1",
    #             "Unknown1-Allele2",
    #             "Unknown2-Allele1",
    #             "Unknown2-Allele2",
    #             "Probability",
    #             #"Posterior_unnormalized",
    #         ],
    #     ]
    #     .copy()
    # )
    
    joint_df_clean = (
        joint_df.loc[
            :,#joint_df["Probability"] >= H2_FILTER_THRESHOLD,
            [
                "Locus",
                "Unknown1-Allele1",
                "Unknown1-Allele2",
                "Unknown2-Allele1",
                "Unknown2-Allele2",
                #"Probability",
                "Posterior_probability_draws",
                # "Posterior_probability_sd",
                # "Log_likelihoods",
                # "Log_posterior_unnormalized_draws",
                # "Average_likelihood",
                # "Prior",
            ],
        ]
        .copy()
    )
    
    marginal_df = create_marginal_df_combined(joint_df, observed_alleles,freq_dict, RARE_FREQ)
    #marginal_df = marginal_df.sort_values(by=["Locus", "Contributor", "Marginal_probability"],ascending=[True, True, False])
    marginal_df["_marginal_probability_mean"] = marginal_df[
        "Marginal_probability_draws"
    ].apply(lambda x: float(np.mean(np.asarray(x, dtype=float))))
    
    marginal_df = marginal_df.sort_values(
        by=["Locus", "Contributor", "_marginal_probability_mean"],
        ascending=[True, True, False],
    )
    
    marginal_df[["Allele 1", "Allele 2"]] = marginal_df['Genotype'].apply(split_and_sort_genotype)
    marginal_df["Hypothesis"] = 'H2'
    #marginal_df['Probability'] = marginal_df["Marginal_probability"]
    
    # marginal_df = (
    #     marginal_df.loc[
    #         :,
    #         [
    #             "Hypothesis",
    #             "Contributor",
    #             "Locus",
    #             "Allele 1",
    #             "Allele 2",
    #             "Probability",
    #             "LR(a)_1",
    #             #"Posterior_unnormalized",
    #         ],
    #     ]
    #     .copy()
    # )
    
    # marginal_df = (
    #     marginal_df.loc[
    #         :,
    #         [
    #             "Hypothesis",
    #             "Contributor",
    #             "Locus",
    #             "Allele 1",
    #             "Allele 2",
    #             "Probability",
    #             "Marginal_probability_draws",
    #             "Marginal_probability_sd",
    #             "LR(a)_1",
    #             "LR(a)_draws",
    #             "Prior",
    #         ],
    #     ]
    #     .copy()
    # )
    
    marginal_df = (
        marginal_df.loc[
            :,
            [
                "Hypothesis",
                "Contributor",
                "Locus",
                "Allele 1",
                "Allele 2",
                "Marginal_probability_draws",
                "LR(a)_draws",
                #"Prior",
            ],
        ]
        .copy()
    )
    
    
    #Hypothesis	Contributor	Locus	Allele 1	Allele 2	Probability	Likelihood	Genotype_prior_raw	Genotype_prior_fixed	P(E|a)	LR(a)_1	LR(a)_2

    
    
    # merged = compare_with_reference_csv(joint_df, csv_path, tol, mixture)
    # merged = merged.sort_values(by="abs_log_post_diff",ascending=False)
    # merged_clean = merged[(merged['Probability']>= H2_FILTER_THRESHOLD)] 
    # merged_clean = merged_clean.sort_values(by="abs_log_post_diff",ascending=False)
    
    merged = compare_with_reference_csv(
        joint_df,
        csv_path,
        tol,
        mixture,
    )
    
    merged = merged.sort_values(
        by="abs_log_post_diff",
        ascending=False,
    )
    
    merged_clean = merged[
        merged["_posterior_probability_mean"] >= H2_FILTER_THRESHOLD
    ].copy()
    
    merged_clean = merged_clean.sort_values(
        by="abs_log_post_diff",
        ascending=False,
    )
    
    # merged = pd.DataFrame()
    # merged_clean = pd.DataFrame()
    
    return joint_df, joint_df_clean, marginal_df, mixture_df, merged, merged_clean

def build_params_from_results_json(results_json_path, CONFIG_YAML):
    with open(results_json_path, "r") as f:
        data = json.load(f)

    modelParameters = data["hypothesesResults"]["H2"]["modelParameters"]

    mu = modelParameters["expectedPeakHeight"]
    sigma = modelParameters["peakHeightVariance"]
    phi0 = modelParameters["mixtureProportions"][0]
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    C = float(cfg['dnastatistx']['dropin_prob'])
    lam = float(cfg['dnastatistx']['dropin_lambda'])
    beta = modelParameters['degradationSlope']
    #beta = round(beta,9)
    if float(beta) != 0.0:    
        print('degradation ON')
    else:
        beta = 1 #DNAStatistx sets beta = 0 if degradation OFF
        print('degradation OFF')
    print('beta = ', beta)
    return ([phi0, 1 - phi0], mu, sigma, C, lam, beta)

def build_params_from_mcmc(
    mixture_file,
    RESOURCES_PATH,
    CONFIG_YAML,
    mode,
    n_iter,#=100,
    burnin,#=20,
    thin,#=5,
    proposal_scales,#=(0.12, 0.10, 0.10),
    seed,#=12345,
    max_posterior_draws,#=3,
    init_mu,#: float = 1500.0,
    init_sigma,#: float = 0.15,
    init_phi1,#: float = 0.7,
    init_beta,
    mu_1,
    sigma_1,
    candidate_pairs_by_locus=None,
):


    mixture_df = pd.read_csv(mixture_file, sep="\t")

    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    RARE_FREQ = float(cfg["RARE_FREQ"])
    N_contributors = int(cfg["dnastatistx"]["N_CONTRIBUTORS"])
    if N_contributors != 2:
        raise NotImplementedError("Current MCMC implementation is written for 2 contributors.")
        
    # kit_properties_txt = RESOURCES_PATH / cfg['kit_properties_txt']
    # kit_properties_df = kit_properties_parser(kit_properties_txt)
    kit_properties_txt = RESOURCES_PATH / cfg['kit_properties_txt']
    kit_properties_df = kit_properties_parser(kit_properties_txt)
    kit_properties_df = build_kit_properties_lookup(kit_properties_df) #overrule kit_properties_df

    # thresholds_cfg = Configuration(cfg["dnastatistx"]["thresholds"])
    # thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    thresholds_map = dict(cfg["dnastatistx"]["thresholds"])
    frac_threshold = float(cfg["dnastatistx"]["frac_threshold"].replace("*=", "").strip())

    observed_alleles, thresholds = extract_observed_alleles_and_thresholds(
        mixture_df,
        thresholds_map,
        frac_threshold,
        mode,
    )

    freqs_file = RESOURCES_PATH / cfg["dnastatistx"]["frequencies_file"]
    freq_df = pd.read_csv(freqs_file, sep=",")
    freq_dict = build_frequency_dict(freq_df)

    all_alleles = {}
    for locus, allele_dict in freq_dict.items():
        alleles = []
        for allele in allele_dict.keys():
            if "." in allele:
                alleles.append(float(allele))
            else:
                alleles.append(int(allele))
        alleles.sort()
        all_alleles[locus] = alleles

    C = float(cfg["dnastatistx"]["dropin_prob"])
    lam = float(cfg["dnastatistx"]["dropin_lambda"])

    mcmc_result = run_mcmc_for_parameters(
        observed_alleles=observed_alleles,
        all_alleles_by_locus=all_alleles,
        thresholds=thresholds,
        freq_dict=freq_dict,
        RARE_FREQ=RARE_FREQ,
        C=C,
        lam=lam,
        N_contributors=N_contributors,
        n_iter=n_iter,
        burnin=burnin,
        thin=thin,
        proposal_scales=proposal_scales,
        seed=seed,
        init_mu=init_mu,#: float = 1500.0,
        init_sigma=init_sigma,#: float = 0.15,
        init_phi1=init_phi1,#: float = 0.7,
        init_beta=init_beta,
        mu_1=mu_1,
        sigma_1=sigma_1,
        kit_properties_df=kit_properties_df,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )


    mu_samples = mcmc_result["samples"]["mu"]
    sigma_samples = mcmc_result["samples"]["sigma"]
    beta_samples = mcmc_result["samples"]["beta"]
    phi1_samples = mcmc_result["samples"]["phi1"]
    phi2_samples = mcmc_result["samples"]["phi2"]

    params_lst = [
        ([float(phi1), float(phi2)], float(mu), float(sigma), C, lam, float(beta))
        for mu, sigma, phi1, phi2, beta in zip(mu_samples, sigma_samples, phi1_samples, phi2_samples, beta_samples)
    ]

    if max_posterior_draws is not None and len(params_lst) > max_posterior_draws:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(params_lst), size=max_posterior_draws, replace=False)
        params_lst = [params_lst[i] for i in idx]

    return params_lst, mcmc_result

def build_params_for_single_mixture(
    parameter_method,
    mixture,
    mixture_file,
    resources_path,
    config_yaml,
    mode,
    params_json_getter,
    mcmc_config_dict=None,
    candidate_pairs_by_locus=None,
):
    """
    Returns:
        params_lst, mcmc_result

    For MLE:
        params_lst = [MLE params]
        mcmc_result = None

    For MCMC:
        params_lst = posterior parameter draws
        mcmc_result = full MCMC output
    """
    parameter_method = parameter_method.upper()

    if parameter_method == "MLE":
        params_json_path = params_json_getter(mixture)
        params_lst = [build_params_from_results_json(params_json_path, config_yaml)]
        return params_lst, None

    if parameter_method == "MCMC":
        if mcmc_config_dict is None:
            raise ValueError("mcmc_config_dict is required when parameter_method='MCMC'.")

        params_json_path = params_json_getter(mixture)
        init_params_lst = build_params_from_results_json(params_json_path, config_yaml)

        init_mu = init_params_lst[1]
        init_sigma = init_params_lst[2]
        init_phi1 = init_params_lst[0][0]
        init_beta = init_params_lst[5]

        print("Initial MCMC values")
        print("init_mu", init_mu)
        print("init_sigma", init_sigma)
        print("init_phi1", init_phi1)
        print("init_beta", init_beta)

        print("Building posterior parameter samples with MCMC...")
        t_mcmc0 = time.perf_counter()

        params_lst, mcmc_result = build_params_from_mcmc(
            mixture_file=mixture_file,
            RESOURCES_PATH=resources_path,
            CONFIG_YAML=config_yaml,
            mode=mode,
            n_iter=mcmc_config_dict["n_iter"],
            burnin=mcmc_config_dict["burnin"],
            thin=mcmc_config_dict["thin"],
            proposal_scales=mcmc_config_dict["proposal_scales"],
            seed=mcmc_config_dict["seed"],
            init_mu=init_mu,
            init_sigma=init_sigma,
            init_phi1=init_phi1,
            init_beta=init_beta,
            max_posterior_draws=mcmc_config_dict["max_posterior_draws"],
            mu_1=mcmc_config_dict["mu_1"],
            sigma_1=mcmc_config_dict["sigma_1"],
            candidate_pairs_by_locus=candidate_pairs_by_locus,
        )

        t_mcmc1 = time.perf_counter()
        print(f"MCMC finished in {t_mcmc1 - t_mcmc0:.2f} seconds")
        print(
            f"MCMC {mixture}: "
            f"acc={mcmc_result['acceptance_rate']:.3f}, "
            f"mu={mcmc_result['posterior_mean']['mu']:.2f}, "
            f"sigma={mcmc_result['posterior_mean']['sigma']:.4f}, "
            f"phi={mcmc_result['posterior_mean']['phi']}, "
            f"beta={mcmc_result['posterior_mean']['beta']:.4f}"
        )
        print(f"95-% confidence interval = {mcmc_result['posterior_ci95']}")

        return params_lst, mcmc_result

    raise ValueError(
        f"Unknown parameter_method={parameter_method}. "
        "Use 'MLE' or 'MCMC'."
    )


def build_common_inputs_for_mcmc(
    RESOURCES_PATH,
    CONFIG_YAML,
):
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    RARE_FREQ = float(cfg["RARE_FREQ"])
    N_contributors = int(cfg["dnastatistx"]["N_CONTRIBUTORS"])

    # thresholds_cfg = Configuration(cfg["dnastatistx"]["thresholds"])
    # thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    thresholds_map = dict(cfg["dnastatistx"]["thresholds"])
    frac_threshold = float(
        cfg["dnastatistx"]["frac_threshold"].replace("*=", "").strip()
    )

    freqs_file = RESOURCES_PATH / cfg["dnastatistx"]["frequencies_file"]
    freq_df = pd.read_csv(freqs_file, sep=",")
    freq_dict = build_frequency_dict(freq_df)

    all_alleles = {}
    for locus, allele_dict in freq_dict.items():
        alleles = []
        for allele in allele_dict.keys():
            if "." in allele:
                alleles.append(float(allele))
            else:
                alleles.append(int(allele))
        alleles.sort()
        all_alleles[locus] = alleles

    C = float(cfg["dnastatistx"]["dropin_prob"])
    lam = float(cfg["dnastatistx"]["dropin_lambda"])

    kit_properties_txt = RESOURCES_PATH / cfg["kit_properties_txt"]
    kit_properties_df = kit_properties_parser(kit_properties_txt)
    kit_properties_df = build_kit_properties_lookup(kit_properties_df)

    return {
        "cfg": cfg,
        "RARE_FREQ": RARE_FREQ,
        "N_contributors": N_contributors,
        "thresholds_map": thresholds_map,
        "frac_threshold": frac_threshold,
        "freq_dict": freq_dict,
        "all_alleles": all_alleles,
        "C": C,
        "lam": lam,
        "kit_properties_df": kit_properties_df,
    }


def build_rework_params_from_mcmc(
    rework_mixture,
    filled_replicate_dfs_by_mixture, #mixture_folder,
    RESOURCES_PATH,
    CONFIG_YAML,
    mode,
    mcmc_config_dict,
    init_params_json_getter,
    candidate_pairs_by_locus=None,
):
    """
    Estimate parameters for one combined rework mixture.

    The MCMC target is:

        p(theta | E_1, E_2, E_3)

    where E_1, E_2, E_3 are the three replicate profiles.
    """
    common = build_common_inputs_for_mcmc(
        RESOURCES_PATH=RESOURCES_PATH,
        CONFIG_YAML=CONFIG_YAML,
    )

    if common["N_contributors"] != 2:
        raise NotImplementedError(
            "Current MCMC implementation is written for 2 contributors."
        )

    observed_alleles_list = []
    thresholds_list = []

    for replicate in [1, 2, 3]:
        mixture = f"{replicate}_{rework_mixture}"
        # mixture_file = mixture_folder / f"{mixture}.txt"

        # mixture_df = pd.read_csv(mixture_file, sep="\t")
        
        mixture_df = filled_replicate_dfs_by_mixture[mixture].copy()

        observed_alleles, thresholds = extract_observed_alleles_and_thresholds(
            mixture_df,
            common["thresholds_map"],
            common["frac_threshold"],
            mode,
        )

        observed_alleles_list.append(observed_alleles)
        thresholds_list.append(thresholds)

    # Use the combined/rework MLE as starting point
    init_params = build_params_from_results_json(
        init_params_json_getter(rework_mixture),
        CONFIG_YAML,
    )

    init_mu = init_params[1]
    init_sigma = init_params[2]
    init_phi1 = init_params[0][0]
    init_beta = init_params[5]

    print("Initial rework values")
    print("init_mu", init_mu)
    print("init_sigma", init_sigma)
    print("init_phi1", init_phi1)
    print("init_beta", init_beta)

    print("Building rework posterior parameter samples with MCMC...")
    t_mcmc0 = time.perf_counter()

    mcmc_result = run_mcmc_for_rework_parameters(
        observed_alleles_list=observed_alleles_list,
        all_alleles_by_locus=common["all_alleles"],
        thresholds_list=thresholds_list,
        freq_dict=common["freq_dict"],
        RARE_FREQ=common["RARE_FREQ"],
        C=common["C"],
        lam=common["lam"],
        N_contributors=common["N_contributors"],
        n_iter=mcmc_config_dict["n_iter"],
        burnin=mcmc_config_dict["burnin"],
        thin=mcmc_config_dict["thin"],
        proposal_scales=mcmc_config_dict["proposal_scales"],
        seed=mcmc_config_dict["seed"],
        init_mu=init_mu,
        init_sigma=init_sigma,
        init_phi1=init_phi1,
        init_beta=init_beta,
        mu_1=mcmc_config_dict["mu_1"],
        sigma_1=mcmc_config_dict["sigma_1"],
        kit_properties_df=common["kit_properties_df"],
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )

    t_mcmc1 = time.perf_counter()
    print(f"Rework MCMC finished in {t_mcmc1 - t_mcmc0:.2f} seconds")

    print(
        f"REWORK MCMC {rework_mixture}: "
        f"acc={mcmc_result['acceptance_rate']:.3f}, "
        f"mu={mcmc_result['posterior_mean']['mu']:.2f}, "
        f"sigma={mcmc_result['posterior_mean']['sigma']:.4f}, "
        f"phi={mcmc_result['posterior_mean']['phi']}, "
        f"beta={mcmc_result['posterior_mean']['beta']:.4f}"
    )
    print(f"95-% confidence interval = {mcmc_result['posterior_ci95']}")

    mu_samples = mcmc_result["samples"]["mu"]
    sigma_samples = mcmc_result["samples"]["sigma"]
    beta_samples = mcmc_result["samples"]["beta"]
    phi1_samples = mcmc_result["samples"]["phi1"]
    phi2_samples = mcmc_result["samples"]["phi2"]

    params_lst = [
        ([float(phi1), float(phi2)], float(mu), float(sigma), common["C"], common["lam"], float(beta))
        for mu, sigma, phi1, phi2, beta in zip(
            mu_samples,
            sigma_samples,
            phi1_samples,
            phi2_samples,
            beta_samples,
        )
    ]

    max_posterior_draws = mcmc_config_dict["max_posterior_draws"]

    if max_posterior_draws is not None and len(params_lst) > max_posterior_draws:
        rng = np.random.default_rng(mcmc_config_dict["seed"])
        idx = rng.choice(len(params_lst), size=max_posterior_draws, replace=False)
        params_lst = [params_lst[i] for i in idx]

    return params_lst, mcmc_result

def get_combined_observed_alleles_for_rework(
    rework_mixture,
    filled_replicate_dfs_by_mixture, #mixture_folder,
    thresholds_map,
    frac_threshold,
):
    combined_observed = {}

    for replicate in [1, 2, 3]:
        mixture = f"{replicate}_{rework_mixture}"
        # mixture_file = mixture_folder / f"{mixture}.txt"

        # mixture_df = pd.read_csv(mixture_file, sep="\t")
        mixture_df = filled_replicate_dfs_by_mixture[mixture].copy()

        observed_alleles, _ = extract_observed_alleles_and_thresholds(
            mixture_df,
            thresholds_map,
            frac_threshold,
            mode="rework",
        )

        for locus, allele_dict in observed_alleles.items():
            combined_observed.setdefault(locus, {})

            for allele, height in allele_dict.items():
                old_height = combined_observed[locus].get(allele, 0)
                combined_observed[locus][allele] = max(old_height, height)

    return combined_observed

def parse_candidate_allele(x):
    """
    Convert allele values from DNAStatistX CSV into the same style used
    elsewhere in the pipeline: int, float, or 'Ø'.
    """
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Fix encoding issue
    s = s.replace("Ã˜", "Ø")

    if s == "Ø":
        return "Ø"

    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except ValueError:
        return s


def canonical_genotype_pair(a, b):
    """
    Sort alleles inside a genotype so that (12, 10) becomes (10, 12),
    while keeping 'Ø' after numeric alleles.
    """
    a = parse_candidate_allele(a)
    b = parse_candidate_allele(b)

    def sort_key(v):
        return (isinstance(v, str), v)

    return tuple(sorted((a, b), key=sort_key))


def load_candidate_genotype_pairs(
    csv_path,
    min_probability=0.0,
    max_rows_per_locus=None,
    add_contributor_swaps=False,
):
    """
    Reads results_joint_deconvolution_H2_clean.csv and returns:

        dict[locus] -> list[((u1a, u1b), (u2a, u2b))]

    These tuples can directly replace product(genos, repeat=2).
    """
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.replace("Ã˜", "Ø")

    if "Probability" in df.columns:
        df = df[df["Probability"] >= min_probability].copy()

    if max_rows_per_locus is not None:
        df = (
            df.sort_values(["Locus", "Probability"], ascending=[True, False])
              .groupby("Locus", as_index=False)
              .head(max_rows_per_locus)
              .copy()
        )

    required_cols = [
        "Locus",
        "Unknown1-Allele1",
        "Unknown1-Allele2",
        "Unknown2-Allele1",
        "Unknown2-Allele2",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {missing}")

    candidate_pairs_by_locus = {}

    for _, row in df.iterrows():
        locus = row["Locus"]

        g1 = canonical_genotype_pair(
            row["Unknown1-Allele1"],
            row["Unknown1-Allele2"],
        )
        g2 = canonical_genotype_pair(
            row["Unknown2-Allele1"],
            row["Unknown2-Allele2"],
        )

        candidate_pairs_by_locus.setdefault(locus, []).append((g1, g2))

        if add_contributor_swaps and g1 != g2:
            candidate_pairs_by_locus[locus].append((g2, g1))

    # Remove duplicates while preserving order
    for locus, pairs in candidate_pairs_by_locus.items():
        seen = set()
        unique_pairs = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        candidate_pairs_by_locus[locus] = unique_pairs

    return candidate_pairs_by_locus


def prepare_deconvolution_df_for_csv(df):
    """
    Convert array-valued columns to JSON strings before writing to CSV.
    This preserves the full MCMC vector in a readable/reloadable format.
    """
    df = df.copy()

    array_cols = [
        "Log_likelihoods",
        "Log_posterior_unnormalized_draws",
        "Posterior_probability_draws",
    ]

    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(np.asarray(x, dtype=float).tolist())
            )

    return df


def run_mode_for_mixtures(
    mixtures,
    mixture_folder,
    resources_path,
    config_yaml,
    tol,
    mode,
    parameter_method,
    params_json_getter,
    reference_csv_getter,
    candidate_csv_getter=None,
    mcmc_config_dict=None,
    joint_output_folder=None,
    marginal_output_folder=None,
    plot_mcmc=False,
):
    df_single_mixtures = pd.DataFrame()

    parameter_method = parameter_method.upper()

    if joint_output_folder is not None:
        joint_output_folder = Path(joint_output_folder)

    if marginal_output_folder is not None:
        marginal_output_folder = Path(marginal_output_folder)
    

    print("mode =", mode)
    print("parameter_method =", parameter_method)

    for mixture in mixtures:
        print("mixture =", mixture)

        # mixture_file = mixture_folder / f"{mixture}.txt"
        # reference_csv_path = reference_csv_getter(mixture)
        mixture_file = mixture_folder / f"{mixture}.txt"
        mixture_df = pd.read_csv(mixture_file, sep="\t")
        reference_csv_path = reference_csv_getter(mixture)

        candidate_pairs_by_locus = None

        if candidate_csv_getter is not None:
            candidate_pairs_by_locus = load_candidate_genotype_pairs(
                candidate_csv_getter(mixture),
                min_probability=0.0,
                max_rows_per_locus=None,
                add_contributor_swaps=False,
            )

        params_lst, mcmc_result = build_params_for_single_mixture(
            parameter_method=parameter_method,
            mixture=mixture,
            mixture_file=mixture_file,
            resources_path=resources_path,
            config_yaml=config_yaml,
            mode=mode,
            params_json_getter=params_json_getter,
            mcmc_config_dict=mcmc_config_dict,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
        )

        if parameter_method == "MCMC" and plot_mcmc:
            plot_mcmc_traces(
                mcmc_result=mcmc_result,
                mixture=mixture,
                output_folder=None,
            )

            plot_mcmc_empirical_distributions(
                mcmc_result=mcmc_result,
                mixture=mixture,
                output_folder=None,
            )
            plot_mcmc_parameter_correlations(
                mcmc_result=mcmc_result,
                mixture=mixture,
                output_folder=None,
            )

        (
            joint_df,
            joint_df_clean,
            marginal_df,
            mixture_df,
            merged,
            merged_clean,
        ) = create_deconvolution_tables(
            mixture_df=mixture_df,#mixture_file=mixture_file,
            params_lst=params_lst,
            RESOURCES_PATH=resources_path,
            CONFIG_YAML=config_yaml,
            csv_path=reference_csv_path,
            tol=tol,
            mixture=mixture,
            mode=mode,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
        )

        if joint_output_folder is not None:
            joint_file = (
                joint_output_folder
                / "mixtures"
                / mixture
                / "results_joint_deconvolution_H2_clean.csv"
            )
            joint_file.parent.mkdir(parents=True, exist_ok=True)
            joint_df_clean_csv = prepare_deconvolution_df_for_csv(joint_df_clean)
            joint_df_clean_csv.to_csv(joint_file, index=False)

        if marginal_output_folder is not None:
            marginal_file = (
                marginal_output_folder
                / "mixtures"
                / mixture
                / "results_marginal_prior_LR.csv"
            )
            marginal_file.parent.mkdir(parents=True, exist_ok=True)
            #marginal_df.to_csv(marginal_file, index=False)
            marginal_df_csv = prepare_marginal_df_for_csv(marginal_df)
            marginal_df_csv.to_csv(marginal_file, index=False)
            
        # Copy results.json to the same output folder(s)
        params_json_path = params_json_getter(mixture)
        
        copy_results_json(
            results_json_path=params_json_path,
            output_folder=joint_output_folder,
            mixture_name=mixture,
        )
        save_mcmc_results_json(
            mcmc_result=mcmc_result,
            output_folder=joint_output_folder,
            mixture_name=mixture,
        )
        
        # if marginal_output_folder != joint_output_folder:
        #     copy_results_json(
        #         results_json_path=params_json_path,
        #         output_folder=marginal_output_folder,
        #         mixture_name=mixture,
        #     )

        # df_single_mixtures = pd.concat(
        #     [df_single_mixtures, merged],
        #     ignore_index=True,
        # )
        if merged is not None and len(merged) > 0:
            df_single_mixtures = pd.concat(
                [df_single_mixtures, merged],
                ignore_index=True,
            )

    return df_single_mixtures


def run_mode_for_rework_mixtures(
    mixtures,
    mixture_folder,
    resources_path,
    config_yaml,
    tol,
    parameter_method,
    init_params_json_getter,
    reference_csv_getter,
    candidate_csv_getter=None,
    mcmc_config_dict=None,
    plot_mcmc=False,
):
    """
    Rework analysis.

    For MLE:
        Uses one combined/rework MLE parameter set per rework mixture.

    For MCMC:
        Runs one rework MCMC jointly on the three replicate profiles.

    Then each replicate is evaluated with the same parameter list,
    and build_combined_analysis_df() combines the replicate likelihoods.
    """
    df_single_mixtures_rework = pd.DataFrame()
    filled_replicate_dfs_by_rework_mixture = {}
    mcmc_results_by_rework_mixture = {}

    parameter_method = parameter_method.upper()

    rework_mixtures = sorted({get_rework_mixture(m) for m in mixtures})

    print("rework_mixtures =", rework_mixtures)
    print("parameter_method =", parameter_method)

    for rework_mixture in rework_mixtures:
        print("Running rework analysis for", rework_mixture)
        
        filled_replicate_dfs_by_mixture = load_and_fill_rework_replicates(
            rework_mixture=rework_mixture,
            mixture_folder=mixture_folder,
        )

        filled_replicate_dfs_by_rework_mixture[rework_mixture] = (
            filled_replicate_dfs_by_mixture
        )

        candidate_pairs_by_locus = None

        if candidate_csv_getter is not None:
            candidate_pairs_by_locus = load_candidate_genotype_pairs(
                candidate_csv_getter(rework_mixture),
                min_probability=0.0,
                max_rows_per_locus=None,
                add_contributor_swaps=False,
            )

        if parameter_method == "MLE":
            params_lst = [
                build_params_from_results_json(
                    init_params_json_getter(rework_mixture),
                    config_yaml,
                )
            ]
            mcmc_result = None

        elif parameter_method == "MCMC":
            if mcmc_config_dict is None:
                raise ValueError("mcmc_config_dict is required when parameter_method='MCMC'.")

            params_lst, mcmc_result = build_rework_params_from_mcmc(
                rework_mixture=rework_mixture,
                filled_replicate_dfs_by_mixture=filled_replicate_dfs_by_mixture, #mixture_folder=mixture_folder,
                RESOURCES_PATH=resources_path,
                CONFIG_YAML=config_yaml,
                mode="rework",
                mcmc_config_dict=mcmc_config_dict,
                init_params_json_getter=init_params_json_getter,
                candidate_pairs_by_locus=candidate_pairs_by_locus,
            )

            if plot_mcmc:
                plot_mcmc_traces(
                    mcmc_result=mcmc_result,
                    mixture=rework_mixture,
                    output_folder=None,
                )

                plot_mcmc_empirical_distributions(
                    mcmc_result=mcmc_result,
                    mixture=rework_mixture,
                    output_folder=None,
                )
                plot_mcmc_parameter_correlations(
                    mcmc_result=mcmc_result,
                    mixture=rework_mixture,
                    output_folder=None,
                )

        else:
            raise ValueError(
                f"Unknown parameter_method={parameter_method}. "
                "Use 'MLE' or 'MCMC'."
            )
        
        mcmc_results_by_rework_mixture[rework_mixture] = mcmc_result
        
        for replicate in [1, 2, 3]:
            mixture = f"{replicate}_{rework_mixture}"
            print("Creating rework replicate deconvolution for", mixture)

            # mixture_file = mixture_folder / f"{mixture}.txt"
            # reference_csv_path = reference_csv_getter(mixture)
            
            mixture_df = filled_replicate_dfs_by_mixture[mixture].copy()
            reference_csv_path = reference_csv_getter(mixture)

            (
                joint_df,
                joint_df_clean,
                marginal_df,
                mixture_df,
                merged,
                merged_clean,
            ) = create_deconvolution_tables(
                mixture_df=mixture_df, #mixture_file=mixture_file,
                params_lst=params_lst,
                RESOURCES_PATH=resources_path,
                CONFIG_YAML=config_yaml,
                csv_path=reference_csv_path,
                tol=tol,
                mixture=mixture,
                mode="rework",
                candidate_pairs_by_locus=candidate_pairs_by_locus,
            )
                
            joint_df = joint_df.copy()

            joint_df["mixture"] = mixture
            joint_df["replicate"] = replicate
            joint_df["mixture_combined"] = rework_mixture
            
            df_single_mixtures_rework = pd.concat(
                [df_single_mixtures_rework, joint_df],
                ignore_index=True,
            )

            # df_single_mixtures_rework = pd.concat(
            #     [df_single_mixtures_rework, merged],
            #     ignore_index=True,
            # )

    #return df_single_mixtures_rework, filled_replicate_dfs_by_rework_mixture
    return (
        df_single_mixtures_rework,
        filled_replicate_dfs_by_rework_mixture,
        mcmc_results_by_rework_mixture,
    )


def build_combined_analysis_df(df_single_mixtures_rework):
    group_cols = ["mixture_combined", "Locus", "Unknown1", "Unknown2", "Prior"]

    df_rework_mixtures = (
        df_single_mixtures_rework
        .groupby(group_cols, as_index=False)
        .agg({"Log_likelihoods": list})
    )

    # For each genotype tuple:
    # combine replicate likelihoods per MCMC draw by summing log-likelihoods.
    df_rework_mixtures["Log_likelihoods"] = (
        df_rework_mixtures["Log_likelihoods"].apply(
            lambda rows: np.asarray(
                [sum(values) for values in zip(*rows)],
                dtype=float,
            )
        )
    )

    df_rework_mixtures["Log_posterior_unnormalized_draws"] = (
        df_rework_mixtures.apply(
            lambda row: np.log(row["Prior"]) + row["Log_likelihoods"]
            if row["Prior"] > 0
            else np.full_like(row["Log_likelihoods"], -np.inf, dtype=float),
            axis=1,
        )
    )

    # Normalize per mixture/locus/MCMC draw
    out_parts = []

    for (mixture_combined, locus), group in df_rework_mixtures.groupby(
        ["mixture_combined", "Locus"]
    ):
        group = group.copy()

        log_raw_matrix = np.vstack(
            group["Log_posterior_unnormalized_draws"].to_numpy()
        )

        log_denominators = logsumexp(log_raw_matrix, axis=0)

        if np.any(~np.isfinite(log_denominators)):
            raise ValueError(
                f"Cannot normalize rework posterior probabilities for "
                f"{mixture_combined}, locus {locus}."
            )

        posterior_prob_matrix = np.exp(log_raw_matrix - log_denominators)
        
        group["Posterior_probability_draws"] = [
            posterior_prob_matrix[i, :]
            for i in range(posterior_prob_matrix.shape[0])
        ]
        
        # Scalar used only for comparison with DNAStatistX
        group["Posterior_unnormalized"] = group[
            "Log_posterior_unnormalized_draws"
        ].apply(
            lambda x: float(
                np.exp(
                    logsumexp(x) -
                    np.log(len(x))
                )
            )
        )
        
        out_parts.append(group)

        # group["Posterior_probability_draws"] = [
        #     posterior_prob_matrix[i, :]
        #     for i in range(posterior_prob_matrix.shape[0])
        # ]

        # group["Probability"] = group["Posterior_probability_draws"].apply(
        #     lambda x: float(np.mean(x))
        # )

        # # group["Posterior_probability_sd"] = group["Posterior_probability_draws"].apply(
        # #     lambda x: float(np.std(x, ddof=1))
        # # )

        # # # Compatibility/debug only
        # # group["Posterior_unnormalized"] = group[
        # #     "Log_posterior_unnormalized_draws"
        # # ].apply(
        # #     lambda x: float(np.mean(np.exp(x)))
        # # )

        # out_parts.append(group)

    return pd.concat(out_parts, ignore_index=True)





# def build_combined_analysis_df(df_single_mixtures_rework):
#     group_cols = ["mixture_combined", "Locus", "Unknown1", "Unknown2", "Prior"]

#     df_rework_mixtures = (
#         df_single_mixtures_rework
#         .groupby(group_cols, as_index=False)
#         .agg({"Log_likelihoods": list})
#     )

#     df_rework_mixtures["log_sum_vec"] = (
#         df_rework_mixtures["Log_likelihoods"].apply(
#             lambda rows: [sum(values) for values in zip(*rows)]
#         )
#     )

#     df_rework_mixtures["log_mean_float"] = (
#         df_rework_mixtures["log_sum_vec"].apply(
#             lambda x: logsumexp(x) - np.log(len(x))
#         )
#     )

#     df_rework_mixtures["log_posterior_unnormalized"] = (
#         np.log(df_rework_mixtures["Prior"]) +
#         df_rework_mixtures["log_mean_float"]
#     )

#     df_rework_mixtures["Posterior_unnormalized"] = np.exp(
#         df_rework_mixtures["log_posterior_unnormalized"]
#     )

#     df_rework_mixtures["Probability"] = (
#         df_rework_mixtures["Posterior_unnormalized"] /
#         df_rework_mixtures
#         .groupby(["mixture_combined", "Locus"])["Posterior_unnormalized"]
#         .transform("sum")
#     )

#     return df_rework_mixtures


# def build_rework_comparison_df(
#     df_rework_input,
#     combined_reference_csv_getter,
#     tol,
#     mixture_folder=None,
#     resources_path=None,
#     config_yaml=None,
#     joint_output_folder=None,
#     marginal_output_folder=None,
#     combined_params_json_getter=None,
#     filled_replicate_dfs_by_rework_mixture=None,
# ):
    

def build_rework_comparison_df(
    df_rework_input,
    combined_reference_csv_getter=None,
    tol=None,
    mixture_folder=None,
    resources_path=None,
    config_yaml=None,
    joint_output_folder=None,
    marginal_output_folder=None,
    combined_params_json_getter=None,
    filled_replicate_dfs_by_rework_mixture=None,
    mcmc_results_by_rework_mixture=None,
):
    """
    Build combined rework Bayesian deconvolution outputs.

    This no longer compares to DNAStatistX scalar output.
    It combines replicate log-likelihood vectors per MCMC draw,
    normalizes per draw, and writes vector-valued joint and marginal CSVs.
    """
    all_results = []

    if df_rework_input is None or len(df_rework_input) == 0:
        print("No rework input rows available.")
        return pd.DataFrame()

    need_exports = (
        joint_output_folder is not None or
        marginal_output_folder is not None
    )

    if need_exports:
        if resources_path is None or config_yaml is None:
            raise ValueError(
                "resources_path and config_yaml are required when exporting rework files."
            )

        common = build_common_inputs_for_mcmc(
            RESOURCES_PATH=resources_path,
            CONFIG_YAML=config_yaml,
        )

        H2_FILTER_THRESHOLD = float(common["cfg"]["H2_FILTER_THRESHOLD"])

    if marginal_output_folder is not None and filled_replicate_dfs_by_rework_mixture is None:
        raise ValueError(
            "filled_replicate_dfs_by_rework_mixture is required when marginal_output_folder is provided."
        )

    if joint_output_folder is not None:
        joint_output_folder = Path(joint_output_folder)

    if marginal_output_folder is not None:
        marginal_output_folder = Path(marginal_output_folder)

    for rework_mixture, group_df in df_rework_input.groupby("mixture_combined"):
        
        reference_csv_path = combined_reference_csv_getter(
            rework_mixture
        )

        df_rework_posteriors = build_combined_analysis_df(group_df)

        # Internal temporary scalar for filtering/sorting only.
        # Do not export this as the main result.
        df_rework_posteriors["_posterior_probability_mean"] = (
            df_rework_posteriors["Posterior_probability_draws"]
            .apply(lambda x: float(np.mean(np.asarray(x, dtype=float))))
        )

        # -------------------------
        # Save joint rework table
        # -------------------------
        if joint_output_folder is not None:
            joint_df_clean = df_rework_posteriors.copy()

            joint_df_clean[["Unknown1-Allele1", "Unknown1-Allele2"]] = (
                joint_df_clean["Unknown1"].apply(split_and_sort_genotype)
            )
            joint_df_clean[["Unknown2-Allele1", "Unknown2-Allele2"]] = (
                joint_df_clean["Unknown2"].apply(split_and_sort_genotype)
            )

            joint_df_clean = (
                joint_df_clean.loc[
                    joint_df_clean["_posterior_probability_mean"] >= H2_FILTER_THRESHOLD,
                    [
                        "Locus",
                        "Unknown1-Allele1",
                        "Unknown1-Allele2",
                        "Unknown2-Allele1",
                        "Unknown2-Allele2",
                        "Posterior_probability_draws",
                    ],
                ]
                .copy()
                .sort_values(by=["Locus"], ascending=[True])
            )

            joint_file = (
                joint_output_folder
                / "mixtures"
                / rework_mixture
                / "results_joint_deconvolution_H2_clean.csv"
            )
            joint_file.parent.mkdir(parents=True, exist_ok=True)

            joint_df_clean_csv = prepare_deconvolution_df_for_csv(joint_df_clean)
            joint_df_clean_csv.to_csv(joint_file, index=False)

        # -------------------------
        # Save marginal rework table
        # -------------------------
        if marginal_output_folder is not None:
            observed_alleles = get_combined_observed_alleles_for_rework(
                rework_mixture=rework_mixture,
                filled_replicate_dfs_by_mixture=filled_replicate_dfs_by_rework_mixture[rework_mixture],
                thresholds_map=common["thresholds_map"],
                frac_threshold=common["frac_threshold"],
            )

            marginal_df = create_marginal_df_combined(
                df_rework_posteriors,
                observed_alleles,
                common["freq_dict"],
                common["RARE_FREQ"],
            )

            marginal_df["_marginal_probability_mean"] = (
                marginal_df["Marginal_probability_draws"]
                .apply(lambda x: float(np.mean(np.asarray(x, dtype=float))))
            )

            marginal_df.sort_values(
                by=["Locus", "Contributor", "_marginal_probability_mean"],
                ascending=[True, True, False],
                inplace=True,
            )

            marginal_df[["Allele 1", "Allele 2"]] = (
                marginal_df["Genotype"].apply(split_and_sort_genotype)
            )

            marginal_df["Hypothesis"] = "H2"

            marginal_df = (
                marginal_df.loc[
                    :,
                    [
                        "Hypothesis",
                        "Contributor",
                        "Locus",
                        "Allele 1",
                        "Allele 2",
                        "Marginal_probability_draws",
                        "LR(a)_draws",
                    ],
                ]
                .copy()
            )

            marginal_file = (
                marginal_output_folder
                / "mixtures"
                / rework_mixture
                / "results_marginal_prior_LR.csv"
            )
            marginal_file.parent.mkdir(parents=True, exist_ok=True)

            marginal_df_csv = prepare_marginal_df_for_csv(marginal_df)
            marginal_df_csv.to_csv(marginal_file, index=False)

        # Copy DNAStatistX parameter json, renamed to results_dnax.json
        if combined_params_json_getter is not None:
            params_json_path = combined_params_json_getter(rework_mixture)

            copy_results_json(
                results_json_path=params_json_path,
                output_folder=joint_output_folder,
                mixture_name=rework_mixture,
            )

        # Save MCMC chain
        if mcmc_results_by_rework_mixture is not None:
            save_mcmc_results_json(
                mcmc_result=mcmc_results_by_rework_mixture.get(rework_mixture),
                output_folder=joint_output_folder,
                mixture_name=rework_mixture,
            )
    
        df_rework_comparison = compare_with_reference_csv(
            df_rework_posteriors,
            reference_csv_path,
            tol,
            rework_mixture,
        ).sort_values(
            by="abs_log_post_diff",
            ascending=False,
        )
        all_results.append(df_rework_comparison)
        #all_results.append(df_rework_posteriors)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)    
    

# def build_rework_comparison_df(
#     df_rework_input,
#     combined_reference_csv_getter,
#     tol,
#     mixture_folder=None,
#     resources_path=None,
#     config_yaml=None,
#     joint_output_folder=None,
#     marginal_output_folder=None,
#     combined_params_json_getter=None,
#     filled_replicate_dfs_by_rework_mixture=None,
#     mcmc_results_by_rework_mixture=None,
# ):
#     all_results = []

#     need_exports = (
#         joint_output_folder is not None or
#         marginal_output_folder is not None
#     )

#     if need_exports:
#         if resources_path is None or config_yaml is None:
#             raise ValueError(
#                 "resources_path and config_yaml are required when exporting rework files."
#             )

#         common = build_common_inputs_for_mcmc(
#             RESOURCES_PATH=resources_path,
#             CONFIG_YAML=config_yaml,
#         )

#         H2_FILTER_THRESHOLD = float(common["cfg"]["H2_FILTER_THRESHOLD"])

#     # if marginal_output_folder is not None and mixture_folder is None:
#     #     raise ValueError(
#     #         "mixture_folder is required when marginal_output_folder is provided."
#     #     )
#     if marginal_output_folder is not None and filled_replicate_dfs_by_rework_mixture is None:
#         raise ValueError(
#             "filled_replicate_dfs_by_rework_mixture is required when marginal_output_folder is provided."
#         )

#     if joint_output_folder is not None:
#         joint_output_folder = Path(joint_output_folder)

#     if marginal_output_folder is not None:
#         marginal_output_folder = Path(marginal_output_folder)

#     for rework_mixture, group_df in df_rework_input.groupby("mixture_combined"):
#         reference_csv_path = combined_reference_csv_getter(rework_mixture)

#         df_rework_posteriors = build_combined_analysis_df(group_df)

#         # -------------------------
#         # Save joint rework table
#         # -------------------------
#         if joint_output_folder is not None:
#             joint_df_clean = df_rework_posteriors.copy()

#             joint_df_clean[["Unknown1-Allele1", "Unknown1-Allele2"]] = (
#                 joint_df_clean["Unknown1"].apply(split_and_sort_genotype)
#             )
#             joint_df_clean[["Unknown2-Allele1", "Unknown2-Allele2"]] = (
#                 joint_df_clean["Unknown2"].apply(split_and_sort_genotype)
#             )

#             joint_df_clean = (
#                 joint_df_clean.loc[
#                     joint_df_clean["Probability"] >= H2_FILTER_THRESHOLD,
#                     [
#                         "Locus",
#                         "Unknown1-Allele1",
#                         "Unknown1-Allele2",
#                         "Unknown2-Allele1",
#                         "Unknown2-Allele2",
#                         "Probability",
#                     ],
#                 ]
#                 .copy()
#                 .sort_values(by=["Locus", "Probability"], ascending=[True, False])
#             )

#             joint_file = (
#                 joint_output_folder
#                 / "mixtures"
#                 / rework_mixture
#                 / "results_joint_deconvolution_H2_clean.csv"
#             )
#             joint_file.parent.mkdir(parents=True, exist_ok=True)
#             #joint_df_clean.to_csv(joint_file, index=False)
#             joint_df_clean_csv = prepare_deconvolution_df_for_csv(joint_df_clean)
#             joint_df_clean_csv.to_csv(joint_file, index=False)

#         # -------------------------
#         # Save marginal rework table
#         # -------------------------
#         if marginal_output_folder is not None:
#             observed_alleles = get_combined_observed_alleles_for_rework(
#                 rework_mixture=rework_mixture,
#                 #mixture_folder=mixture_folder,
#                 filled_replicate_dfs_by_mixture=filled_replicate_dfs_by_rework_mixture[rework_mixture],
#                 thresholds_map=common["thresholds_map"],
#                 frac_threshold=common["frac_threshold"],
#             )

#             marginal_df = create_marginal_df_combined(
#                 df_rework_posteriors,
#                 observed_alleles,
#                 common["freq_dict"],
#                 common["RARE_FREQ"],
#             )

#             marginal_df.sort_values(
#                 by=["Locus", "Contributor", "Marginal_probability"],
#                 ascending=[True, True, False],
#                 inplace=True,
#             )

#             marginal_df[["Allele 1", "Allele 2"]] = (
#                 marginal_df["Genotype"].apply(split_and_sort_genotype)
#             )
#             marginal_df["Hypothesis"] = "H2"
#             marginal_df["Probability"] = marginal_df["Marginal_probability"]

#             marginal_df = (
#                 marginal_df.loc[
#                     :,
#                     [
#                         "Hypothesis",
#                         "Contributor",
#                         "Locus",
#                         "Allele 1",
#                         "Allele 2",
#                         "Probability",
#                         "LR(a)_1",
#                     ],
#                 ]
#                 .copy()
#             )

#             marginal_file = (
#                 marginal_output_folder
#                 / "mixtures"
#                 / rework_mixture
#                 / "results_marginal_prior_LR.csv"
#             )
#             marginal_file.parent.mkdir(parents=True, exist_ok=True)
#             #marginal_df.to_csv(marginal_file, index=False)
#             marginal_df_csv = prepare_marginal_df_for_csv(marginal_df)
#             marginal_df_csv.to_csv(marginal_file, index=False)

#         # df_rework_comparison = compare_with_reference_csv(
#         #     df_rework_posteriors,
#         #     reference_csv_path,
#         #     tol,
#         #     rework_mixture,
#         # ).sort_values(by="abs_log_post_diff", ascending=False)

#         # all_results.append(df_rework_comparison)
        
#         df_rework_comparison = compare_with_reference_csv(
#             df_rework_posteriors,
#             reference_csv_path,
#             tol,
#             rework_mixture,
#         ).sort_values(by="abs_log_post_diff", ascending=False)
        
#         # Copy combined/rework results.json to the same output folder(s)
#         if combined_params_json_getter is not None:
#             params_json_path = combined_params_json_getter(rework_mixture)
        
#             copy_results_json(
#                 results_json_path=params_json_path,
#                 output_folder=joint_output_folder,
#                 mixture_name=rework_mixture,
#             )
#         if mcmc_results_by_rework_mixture is not None:
#             save_mcmc_results_json(
#                 mcmc_result=mcmc_results_by_rework_mixture.get(rework_mixture),
#                 output_folder=joint_output_folder,
#                 mixture_name=rework_mixture,
#             )
                
#             # if marginal_output_folder != joint_output_folder:
#             #     copy_results_json(
#             #         results_json_path=params_json_path,
#             #         output_folder=marginal_output_folder,
#             #         mixture_name=rework_mixture,
#             #     )
        
#         all_results.append(df_rework_comparison)

#     return pd.concat(all_results, ignore_index=True)


def main():
    resources_path = RESOURCES_PATH
    config_yaml = MCMC_CONFIG_YAML

    mixture_folder = ANALYSIS_MIXTURES_DIR
    mixtures = [p.stem for p in mixture_folder.glob("*.txt")]
    #mixtures = ["1_1B2", "2_1B2", "3_1B2"]
    
    single_mixture_folder = mixture_folder
    rework_mixture_folder = mixture_folder
    
    single_output_folder = MCMC_SINGLE_OUTPUT_ROOT
    rework_output_folder = MCMC_REWORK_OUTPUT_ROOT
    
    # --------------------------------------------------
    # MAIN SWITCHES
    # --------------------------------------------------
    
    #parameter_method = "MLE"
    parameter_method = "MCMC"

    run_single_analysis = True
    run_rework_analysis = True #False

    use_candidate_pairs_single = True
    use_candidate_pairs_rework = True
    plot_mcmc = True

    #degr_mode = "on"
    tol = 1 if parameter_method == "MLE" else 10
    
    # --------------------------------------------------
    # MCMC settings
    # --------------------------------------------------
    mcmc_config_dict = {
        "n_iter": 500,
        "burnin": 100,
        "thin": 10,
        "proposal_scales": (0.12, 0.10, 0.10, 0.10),
        "seed": 12345,
        "max_posterior_draws": 5,
        "mu_1": 40000,
        "sigma_1": 1,
    }


    candidate_csv_getter_single = (
        single_candidate_csv if use_candidate_pairs_single else None
    )

    candidate_csv_getter_rework = (
        rework_candidate_csv if use_candidate_pairs_rework else None
    )

    df_single = None
    df_rework_input = None
    df_rework_comparison = None

    # --------------------------------------------------
    # Single mixture analysis
    # --------------------------------------------------
    if run_single_analysis:
        df_single = run_mode_for_mixtures(
            mixtures=mixtures,
            mixture_folder=single_mixture_folder,
            resources_path=resources_path,
            config_yaml=config_yaml,
            tol=tol,
            mode="single",
            parameter_method=parameter_method,
            params_json_getter=single_params_json,
            reference_csv_getter=reference_csv,
            candidate_csv_getter=candidate_csv_getter_single,
            mcmc_config_dict=mcmc_config_dict,
            joint_output_folder = single_output_folder, 
            marginal_output_folder= single_output_folder,
            plot_mcmc=plot_mcmc,
        )

    # --------------------------------------------------
    # Rework mixture analysis
    # --------------------------------------------------
    if run_rework_analysis:
        df_rework_input, filled_replicate_dfs_by_rework_mixture, mcmc_results_by_rework_mixture = run_mode_for_rework_mixtures(
            mixtures=mixtures,
            mixture_folder=rework_mixture_folder,
            resources_path=resources_path,
            config_yaml=config_yaml,
            tol=tol,
            parameter_method=parameter_method,
            init_params_json_getter=combined_params_json,
            reference_csv_getter=reference_csv,
            candidate_csv_getter=candidate_csv_getter_rework,
            mcmc_config_dict=mcmc_config_dict,
            plot_mcmc=plot_mcmc,
        )

        df_rework_comparison = build_rework_comparison_df(
            df_rework_input=df_rework_input,
            combined_reference_csv_getter=combined_reference_csv,
            tol=tol,
            mixture_folder=rework_mixture_folder,
            resources_path=resources_path,
            config_yaml=config_yaml,
            joint_output_folder=rework_output_folder,
            marginal_output_folder=rework_output_folder, 
            combined_params_json_getter=combined_params_json,
            filled_replicate_dfs_by_rework_mixture=filled_replicate_dfs_by_rework_mixture,
            mcmc_results_by_rework_mixture=mcmc_results_by_rework_mixture,
        )

    return {
        "parameter_method": parameter_method,
        "single": df_single,
        "rework_input": df_rework_input,
        "rework_comparison": df_rework_comparison,
    }



if __name__ == "__main__":
    start = time.perf_counter()

    results = main()

    df_single = results["single"]
    df_rework_input = results["rework_input"]
    df_rework_comparison = results["rework_comparison"]

    if df_rework_comparison is not None and len(df_rework_comparison) > 0:
        df = df_rework_comparison
    elif df_single is not None and len(df_single) > 0:
        df = df_single
    else:
        df = None
        
    if df is not None and len(df) > 0 and "log_post_match" in df.columns:
        true = len(df[df["log_post_match"] == True]) / len(df)
        print("matches " + str(round((true * 100), 2)) + "%")
    
        total_error = sum(df["abs_log_post_diff"])
        print("total error = ", round(total_error, 6))
    else:
        print("No scalar DNAStatistX comparison was performed for Bayesian vector output.")

    # if df is not None:
    #     true = len(df[df["log_post_match"] == True]) / len(df)
    #     print("matches " + str(round((true * 100), 2)) + "%")

    #     total_error = sum(df["abs_log_post_diff"])
        #print("total error = ", round(total_error, 6))

    elapsed = time.perf_counter() - start

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Total runtime: {hours}h {minutes}m {seconds:.2f}s")











