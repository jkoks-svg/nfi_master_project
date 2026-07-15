# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:54:17 2026

@author: jortk
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import platform
import sys
import hashlib
from copy import deepcopy
import os
import glob

import yaml
import numpy as np
import pandas as pd
#from confidence import Configuration

# ----------------------------------------------------------------------
# Genotype prior post-processing (from add_genotype_prior_fixed.py)
# ----------------------------------------------------------------------

def canonical_allele_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in {"Ø"}:
        return s
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        else:
            return str(v)
    except ValueError:
        return s


def build_frequency_map(freq_df: pd.DataFrame):
    allele_nums = freq_df["Allele"]
    allele_labels = [canonical_allele_label(a) for a in allele_nums]

    freq_map = {}
    for locus in freq_df.columns[1:]:
        locus_map = {}
        for label, val in zip(allele_labels, freq_df[locus]):
            if pd.notna(val) and label is not None:
                locus_map[label] = float(val)
        freq_map[locus] = locus_map
    return freq_map


def build_observed_alleles(results_df: pd.DataFrame):
    observed = {}
    null_labels = {"Ø", "Ã˜", "0", "", "NA", "NaN", None}

    for locus, a1, a2 in zip(results_df["Locus"], results_df["Allele 1"], results_df["Allele 2"]):
        loc = str(locus)
        obs_set = observed.setdefault(loc, set())
        for a in (a1, a2):
            label = canonical_allele_label(a)
            if label is None or label in null_labels:
                continue
            obs_set.add(label)
    return observed


def genotype_prior_raw(freq_map, observed_alleles, locus: str, a1, a2) -> float:
    null_labels = {"Ø", "Ã˜", "0", "", "NA", "NaN", None}

    loc = str(locus)
    locus_map = freq_map.get(loc)
    if not locus_map:
        return 0.0

    obs_set = observed_alleles.get(loc, set())

    pU = 0.0
    for allele_label, freq in locus_map.items():
        if allele_label not in obs_set:
            pU += freq

    la1 = canonical_allele_label(a1)
    la2 = canonical_allele_label(a2)

    is_null1 = (la1 is None) or (la1 in null_labels)
    is_null2 = (la2 is None) or (la2 in null_labels)

    if is_null1 and is_null2:
        return pU ** 2

    if is_null1 and not is_null2:
        p_obs = locus_map.get(la2)
        if p_obs is None:
            return 0.0
        return 2.0 * p_obs * pU

    if is_null2 and not is_null1:
        p_obs = locus_map.get(la1)
        if p_obs is None:
            return 0.0
        return 2.0 * p_obs * pU

    p1 = locus_map.get(la1)
    p2 = locus_map.get(la2)
    if p1 is None or p2 is None:
        return 0.0

    if la1 == la2:
        return p1 ** 2
    else:
        return 2.0 * p1 * p2


def build_rare_per_locus(freq_map, RARE_FREQ):
    rare_per_locus = {}
    for locus, locus_map in freq_map.items():
        positives = [f for f in locus_map.values() if f > 0]
        if positives:
            rare_per_locus[locus] = min(min(positives), RARE_FREQ)
        else:
            rare_per_locus[locus] = RARE_FREQ
    return rare_per_locus


def safe_allele_freq(locus_map, locus, allele_label, rare_per_locus, RARE_FREQ):
    EPS = RARE_FREQ**2
    f = locus_map.get(allele_label)
    if f is None or f <= 0 or not np.isfinite(f):
        f = rare_per_locus.get(locus, RARE_FREQ)
    return max(float(f), EPS)


def genotype_prior_fixed(freq_map, observed_alleles, rare_per_locus, locus: str, a1, a2, RARE_FREQ) -> float:
    EPS = RARE_FREQ**2
    null_labels = {"Ø", "Ã˜", "0", "", "NA", "NaN", None}

    loc = str(locus)
    locus_map = freq_map.get(loc)
    if not locus_map:
        return EPS

    obs_set = observed_alleles.get(loc, set())

    pU = 0.0
    for allele_label, freq in locus_map.items():
        if allele_label not in obs_set and freq > 0:
            pU += freq

    if pU <= 0 or not np.isfinite(pU):
        pU = rare_per_locus.get(loc, RARE_FREQ)
    pU = max(float(pU), EPS)

    la1 = canonical_allele_label(a1)
    la2 = canonical_allele_label(a2)

    is_null1 = (la1 is None) or (la1 in null_labels)
    is_null2 = (la2 is None) or (la2 in null_labels)

    if is_null1 and is_null2:
        return max(pU ** 2, EPS)

    if is_null1 and not is_null2:
        p_obs = safe_allele_freq(locus_map, loc, la2, rare_per_locus, RARE_FREQ)
        return max(2.0 * p_obs * pU, EPS)

    if is_null2 and not is_null1:
        p_obs = safe_allele_freq(locus_map, loc, la1, rare_per_locus, RARE_FREQ)
        return max(2.0 * p_obs * pU, EPS)

    p1 = safe_allele_freq(locus_map, loc, la1, rare_per_locus, RARE_FREQ)
    p2 = safe_allele_freq(locus_map, loc, la2, rare_per_locus, RARE_FREQ)

    gp = (p1 ** 2) if (la1 == la2) else (2.0 * p1 * p2)
    return max(float(gp), EPS)

def filter_small_probabilities(
    input_csv,
    output_csv,
    threshold,
    prob_column="Probability",
):
    # Read CSV (let pandas handle scientific notation)
    df = pd.read_csv(input_csv, sep = '\t')

    # Ensure Probability column is numeric
    df[prob_column] = pd.to_numeric(df[prob_column], errors="coerce")

    # Filter rows
    df_filtered = df[df[prob_column] >= threshold]

    # Write output CSV
    df_filtered.to_csv(output_csv, sep = '\t', index=False)
    
def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  🗑️ Deleted: {file_path}")
    except Exception as e:
        print(f"  ⚠️ Could not delete {file_path}: {e}")


def postprocess_add_priors_helper(root_mixtures_dir: Path, freq_csv: Path, RARE_FREQ, H2_FILTER_THRESHOLD, DELETE_INTERMEDIATE_FILES) -> tuple[int, int]:
    """
    For each subfolder in root_mixtures_dir:
      - reads results_marginal_deconvolutions.csv
      - writes results_marginal_deconvolutions_with_prior_and_LR.csv

    Returns: (n_ok, n_skipped_or_failed)
    """
    EPS = RARE_FREQ**2
    if not freq_csv.exists():
        raise FileNotFoundError(f"Frequency CSV not found: {freq_csv}")

    freq_df = pd.read_csv(freq_csv, encoding="utf-8")
    freq_df.rename(columns={"PentaD": "Penta D", "PentaE": "Penta E"}, inplace=True)
    freq_map = build_frequency_map(freq_df)
    rare_per_locus = build_rare_per_locus(freq_map, RARE_FREQ)

    #mixture_dirs = [d for d in glob.glob(os.path.join(str(root_mixtures_dir), "*")) if os.path.isdir(d)]
    
    
    
    
    root_mixtures_dir = Path(root_mixtures_dir)

    if (root_mixtures_dir / "results_marginal_deconvolutions.csv").exists():
        # root_mixtures_dir is already one specific result folder
        mixture_dirs = [str(root_mixtures_dir)]
    else:
        # root_mixtures_dir contains multiple result folders
        mixture_dirs = [
            d for d in glob.glob(os.path.join(str(root_mixtures_dir), "*"))
            if os.path.isdir(d)
        ]
        
        
        
    
    print("\n=== Post-processing priors ===")
    print(f"Found {len(mixture_dirs)} mixture folders under {root_mixtures_dir}")

    n_ok = 0
    n_bad = 0

    for mixdir in mixture_dirs:
        in_file = os.path.join(mixdir, "results_marginal_deconvolutions.csv")
        out_file = os.path.join(mixdir, "results_marginal_prior_LR.csv")
        
        # in_file_joint_H2 = 
        # out_file_joint_H2 = 
        filter_small_probabilities(
            input_csv= os.path.join(mixdir, "results_joint_deconvolution_H2.csv"),
            output_csv=os.path.join(mixdir, "results_joint_deconvolution_H2_clean.csv"),
            threshold=H2_FILTER_THRESHOLD
        )

        if not os.path.exists(in_file):
            print(f"Skipping {mixdir}: no results_marginal_deconvolutions.csv")
            n_bad += 1
            continue

        print(f"\nProcessing priors: {mixdir}")
        try:
            results_df = pd.read_csv(in_file, sep="\t", encoding="utf-8")
            observed = build_observed_alleles(results_df)

            priors_raw = []
            priors_fixed = []

            for locus, a1, a2 in zip(results_df["Locus"], results_df["Allele 1"], results_df["Allele 2"]):
                gp_raw = genotype_prior_raw(freq_map, observed, locus, a1, a2)
                gp_fixed = genotype_prior_fixed(freq_map, observed, rare_per_locus, locus, a1, a2, RARE_FREQ)
                priors_raw.append(gp_raw)
                priors_fixed.append(gp_fixed)

            results_df["Genotype_prior_raw"] = priors_raw
            results_df["Genotype_prior_fixed"] = np.maximum(priors_fixed, EPS)

            col_F = results_df["Probability"]
            col_G = results_df["Likelihood"]
            col_I = results_df["Genotype_prior_fixed"]

            results_df["P(E|a)"] = col_G / col_I
            results_df["LR(a)_1"] = col_F / col_I

            group_cols = [results_df.columns[0], results_df.columns[1], results_df.columns[2]]
            sum_G_by_marker = results_df.groupby(group_cols)[col_G.name].transform("sum")

            results_df["LR(a)_2"] = results_df["P(E|a)"] / sum_G_by_marker

            results_df.to_csv(out_file, sep="\t", index=False, encoding="utf-8")
            print(f"  ✔ Output written to {out_file}")
            
            if DELETE_INTERMEDIATE_FILES:
                # Remove large unnecessary files
                safe_remove(os.path.join(mixdir, "results_joint_deconvolution_H1.csv"))
                safe_remove(os.path.join(mixdir, "results_joint_deconvolution_H2.csv"))
                safe_remove(os.path.join(mixdir, "results_marginal_deconvolutions.csv"))  
            
            n_ok += 1
        except Exception as e:
            print(f"  ❌ Failed for {mixdir}: {e}")
            n_bad += 1

    return n_ok, n_bad

