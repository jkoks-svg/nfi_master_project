# -*- coding: utf-8 -*-
"""
read_csv_batch_clean.py (CLEANED + MODE SWITCH)

What this script does
---------------------
For each mixture:
  1) Detect "bad loci" (drop-in / drop-out) using RAW peaks + known donors.
  2) Compute TRUE donor log10LR from marginal deconvolution output (with Ø fallbacks).
  3) Sample N genotype draws from posterior per locus and sum to totals (ON).
  4) Apply one or more "bad loci handling" methods to create corrected totals:

Bad loci handling modes
-----------------------
- "zero"   : remove bad loci by setting per-locus contribution to 0 (old behavior).
- "impute" : replace bad loci by the ON-sample mean per locus (current behavior).
- "both"   : run both corrections after ON (back-to-back, same ON draws).

Outputs
-------
- Per-mixture per-contributor percentiles for ON and chosen correction(s).
- Optional per-locus check table to verify the correction.
- Optional per-mixture histograms of ON vs corrected totals.
- Percentile histograms across mixtures for ON and each correction method.

Notes
-----
- ON samples are never mutated. Corrections operate on OFF copies.
- For percentile histograms, corrected methods exclude mixtures with too many bad loci
  (MAX_BAD_LOCI_FOR_HIST), while ON includes all mixtures by default.
  You can switch to "paired inclusion" if you want ON and corrected to use same set.

"""

import json
#import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(12345)

# =========================
# CONFIG
# =========================
SHOW_MIXTURE_HISTOGRAMS = True     # per-mixture ON vs corrected totals
SHOW_PERCENTILE_HISTS = True      # across-mixture percentile histograms

N_GLOBAL = 1000
HYPOTHESIS = "H2"
DELTA_THRESHOLD = np.inf

# ---- bad locus detection switches ----
DETECT_BAD_LOCI = False
FLAG_DROPOUT = True
FLAG_DROPIN = True
ANALYTICAL_THRESHOLD = 0  # RFU threshold in raw to consider an allele observed

# ---- histogram distortion filter ----
MAX_BAD_LOCI_FOR_HIST = 50  # corrected methods only; ON includes all by default

# ---- choose correction mode ----
# "zero" | "impute" | "both"
BAD_LOCI_MODE = "both"

# ---- debug / verification ----
PRINT_LOCUS_CHECK_TABLE = True
PRINT_MAX_LOCI = None            # None = all
ASSERT_NON_TOUCHED_UNCHANGED = True

ASSERT_ZERO_MODE_IS_ZERO = True          # for mode="zero"
ASSERT_IMPUTE_MODE_IS_MEAN = True       # for mode="impute"


# ---- degenerate sampling filter (percentile histograms) ----
EXCLUDE_DEGENERATE_FOR_HIST = False   # NEW: remove mixtures where all sampled totals == true total
DEGENERATE_ATOL = 1e-12             # NEW: tolerance for float equality
DEGENERATE_RTOL = 0.0               # NEW: keep strict unless you want relative tolerance too

# ---- KS uniformity stats on percentile histograms ----
COMPUTE_KS_FOR_PERCENTILE_HISTS = True
ANNOTATE_KS_ON_PLOTS = True
SAVE_KS_TABLE = True


# =========================
# DATASET SETTINGS
# =========================
TRUE_DONORS = {
    1: ("1A", "1B"),
    2: ("2F", "2G"),
    3: ("3K", "3L"),
    4: ("4P", "4Q"),
    5: ("5U", "5V"),
    6: ("6Z", "6AA"),
}

RUN_MODE = "batch"  # "batch" or "single"
if RUN_MODE == "batch":
    MIXTURE_TYPES = ["A", "B", "C", "D", "E"]
    DATASETS = [1, 2, 3, 4, 5, 6]
    REPLICATES = [1, 2, 3]
else:
    MIXTURE_TYPES = ["E"]
    DATASETS = [1]
    REPLICATES = [1,2,3]

# -------------------------
# PATHS (EDIT BASE)
# -------------------------
SAVE_FILES = True
version = 'v3'
BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
#"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p\output_HT_2p5p_frac_threshold_yes"
#"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p\output_HT_2p5p_frac_threshold_yes"
#"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p\output_HT_2p5p_runA"
#"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p_nodropin_auto_dropped\output_HT_2p5p_nodropin_runC"
#OUT_ROOT = BASE /"inputs" /"input_HT_2p_nodropin"/"output_HT_2p_nodropin_runF_100simulations"

# #Console 6/A
# OUT_ROOT = BASE /"inputs" /"input_HT_2p_nodropin"/"output_DNASx_singles_frac_thresh_on_degr_on"
# RAW_MIX_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\mixtures_2p")

#Console 7/A
#OUT_ROOT = BASE/'inputs'/'input_2p_Bayes'/'output_single_MLE_2p_v2'
OUT_ROOT = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2")
RAW_MIX_DIR = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\mixtures_2p") 
#Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\mixtures_2p")

# #Console 14/A
# OUT_ROOT = BASE/'inputs'/'input_2p_Bayes'/'output_MCMC_degr_on_500iters_fast' #output_single_MLE_degr_on'
# RAW_MIX_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\mixtures_2p")


RESULTS_DIR = OUT_ROOT / f'results_{version}'
if SAVE_FILES:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DONOR_DIR = BASE / "Dataset PP6FC Mixtures" / "Donoren" 
#RAW_MIX_DIR = BASE / "Dataset PP6FC Mixtures" / "HT_2p5p"
#RAW_MIX_DIR = BASE / "inputs" / "input_HT_2p_nodropin" / "mixtures"



# -------------------------
# Columns in results file
# -------------------------
COL_HYP = "Hypothesis"
COL_CONTR = "Contributor"
COL_MARKER = "Locus"
COL_A1 = "Allele 1"
COL_A2 = "Allele 2"
COL_POST = "Probability"
COL_LR = "LR(a)_1"

# =========================
# CACHES
# =========================
_raw_mix_cache = {}    # mixture -> raw df
_donor_cache = {}      # donor_id -> donor df


from datetime import datetime
import platform
import sys
#import textwrap

def _as_pretty_lines(d: dict) -> str:
    """Pretty key: value lines, stable order, handles lists/paths nicely."""
    lines = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (list, tuple)):
            v_str = "[" + ", ".join(map(str, v)) + "]"
        else:
            v_str = str(v)
        lines.append(f"{k}: {v_str}")
    return "\n".join(lines)

def write_run_parameters_txt(output_dir: Path, params: dict, filename: str = "run_parameters.txt"):
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / filename

    meta = {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
        "random_seed": 12345,  # keep in sync with np.random.seed(...)
    }

    content = []
    content.append("RUN METADATA")
    content.append("-" * 80)
    content.append(_as_pretty_lines(meta))
    content.append("\nCONFIG / PARAMETERS")
    content.append("-" * 80)
    content.append(_as_pretty_lines(params))

    outpath.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"[INFO] Saved run parameters -> {outpath}")
    return outpath

def write_run_parameters_json(output_dir: Path, params: dict, filename: str = "run_parameters.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / filename

    payload = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
        "params": params,
    }

    outpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[INFO] Saved run parameters (json) -> {outpath}")
    return outpath








# =========================
# HELPERS: canonical alleles
# =========================
def canonical_allele(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "Ø":
        return s
    if s.endswith(".0"):
        s = s[:-2]
    return s


def canonical_pair(a1, a2):
    return tuple(sorted([canonical_allele(a1), canonical_allele(a2)]))


# =========================
# HELPERS: sampling + metrics
# =========================
def locus_dominance_metrics(locus_log10_list):
    vals = np.array(locus_log10_list, dtype=float)
    abs_vals = np.abs(vals)
    L1_max = float(abs_vals.max() / abs_vals.sum()) if abs_vals.sum() > 0 else np.nan

    sq_vals = vals ** 2
    L2_max = float(sq_vals.max() / sq_vals.sum()) if sq_vals.sum() > 0 else np.nan
    return L1_max, L2_max


def sample_N_genotypes_with_locus_log10(df, hypothesis, contributor, N=1000):
    """
    Returns:
      total_log10_LRs: (N,) array
      locus_log10: dict {locus_name: (N,) array of per-locus log10LR samples}
    """
    sub = df[(df[COL_HYP] == hypothesis) & (df[COL_CONTR] == contributor)].copy()
    sub = sort_genotype_df(sub) 

    grouped = []
    for locus, g in sub.groupby(COL_MARKER):
        w = g[COL_POST].astype(float).to_numpy()
        w = w / w.sum()
        lrs = g[COL_LR].astype(float).to_numpy()
        grouped.append((locus, w, lrs))

    locus_log10 = {locus: np.empty(N, dtype=float) for locus, *_ in grouped}
    total_log10 = np.empty(N, dtype=float)

    for i in range(N):
        s = 0.0
        for locus, w, lrs in grouped:
            j = np.random.choice(len(lrs), p=w)
            lr_val = float(lrs[j])
            if not np.isfinite(lr_val) or lr_val <= 0:
                raise ValueError(f"Non-finite or non-positive LR {lr_val} at locus {locus}")
            log10v = np.log10(lr_val)
            locus_log10[locus][i] = log10v
            s += log10v
        total_log10[i] = s

    return total_log10, locus_log10


# =========================
# TRUE donor LR computation (with Ø fallbacks)
# =========================
def compute_true_donor_LR(results_df, hypothesis, contributor, donor_df,
                          donor_locus_col="Marker", donor_a1_col="Allele1", donor_a2_col="Allele2"):
    sub = results_df[(results_df[COL_HYP] == hypothesis) & (results_df[COL_CONTR] == contributor)].copy()
    sub["canon_pair"] = sub.apply(lambda r: canonical_pair(r[COL_A1], r[COL_A2]), axis=1)

    log10_lr_values = []
    missing_markers = []

    n_fallback = 0
    n_full_drop = 0
    locus_details = []

    for _, row in donor_df.iterrows():
        locus = row[donor_locus_col]
        a1 = canonical_allele(row[donor_a1_col])
        a2 = canonical_allele(row[donor_a2_col])
        donor_pair = canonical_pair(a1, a2)

        locus_rows = sub[sub[COL_MARKER] == locus]

        # 1) Exact
        exact = locus_rows[locus_rows["canon_pair"] == donor_pair]
        if not exact.empty:
            best = exact.iloc[0]
            lr_val = float(best[COL_LR])
            log10_val = np.log10(lr_val)
            log10_lr_values.append(log10_val)
            locus_details.append({
                "marker": locus, "donor_pair": donor_pair, "used_pair": best["canon_pair"],
                "match_type": "exact", "log10_LR": log10_val, "LR": lr_val
            })
            continue

        # 2) Fallback (a,Ø)
        fallback_pairs = [canonical_pair(a1, "Ø"), canonical_pair(a2, "Ø")]
        fallback = locus_rows[locus_rows["canon_pair"].isin(fallback_pairs)]
        if not fallback.empty:
            best = fallback.sort_values(COL_LR, ascending=False).iloc[0]
            lr_val = float(best[COL_LR])
            log10_val = np.log10(lr_val)
            log10_lr_values.append(log10_val)
            n_fallback += 1
            locus_details.append({
                "marker": locus, "donor_pair": donor_pair, "used_pair": best["canon_pair"],
                "match_type": "fallback_oneØ", "log10_LR": log10_val, "LR": lr_val
            })
            continue

        # 3) Full dropout (Ø,Ø)
        full = locus_rows[locus_rows["canon_pair"] == canonical_pair("Ø", "Ø")]
        if not full.empty:
            best = full.iloc[0]
            lr_val = float(best[COL_LR])
            log10_val = np.log10(lr_val)
            log10_lr_values.append(log10_val)
            n_full_drop += 1
            locus_details.append({
                "marker": locus, "donor_pair": donor_pair, "used_pair": best["canon_pair"],
                "match_type": "full_dropØØ", "log10_LR": log10_val, "LR": lr_val
            })
            continue

        # 4) No match
        missing_markers.append(locus)
        locus_details.append({
            "marker": locus, "donor_pair": donor_pair, "used_pair": None,
            "match_type": "no_match", "log10_LR": np.nan, "LR": np.nan
        })

    if missing_markers:
        print(f"[WARNING] No matching genotype rows for markers: {missing_markers}")

    log10_total = float(np.nansum(log10_lr_values))
    LR_total = 10 ** log10_total
    locus_log10_list = [v for v in log10_lr_values if np.isfinite(v)]
    return log10_total, LR_total, n_fallback, n_full_drop, locus_details, locus_log10_list


# =========================
# RAW mixture + bad loci detection
# =========================
def load_raw_mixture_txt(mixture: str) -> pd.DataFrame:
    if mixture in _raw_mix_cache:
        return _raw_mix_cache[mixture]

    path = RAW_MIX_DIR / f"{mixture}.txt"
    if not path.exists():
        print(f"[WARNING] Raw mixture file not found: {path}")
        _raw_mix_cache[mixture] = pd.DataFrame()
        return _raw_mix_cache[mixture]

    df = pd.read_csv(path, sep="\t", dtype=str)
    _raw_mix_cache[mixture] = df
    return df


def _to_float_safe(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(",", ".")
        return float(s)
    except Exception:
        return np.nan


def extract_observed_raw_alleles_above_AT(raw_row: pd.Series, AT: float) -> set:
    cols = list(raw_row.index)
    allele_cols = [c for c in cols if "allele" in str(c).lower()]
    height_cols = [c for c in cols if "height" in str(c).lower()]

    def suffix_digits(name):
        s = str(name)
        digits = ""
        for ch in reversed(s):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        return digits

    allele_by_k, height_by_k = {}, {}
    for ac in allele_cols:
        allele_by_k.setdefault(suffix_digits(ac), []).append(ac)
    for hc in height_cols:
        height_by_k.setdefault(suffix_digits(hc), []).append(hc)

    observed = set()

    for k in set(allele_by_k.keys()) & set(height_by_k.keys()):
        ac = allele_by_k[k][0]
        hc = height_by_k[k][0]
        a = canonical_allele(raw_row.get(ac))
        h = _to_float_safe(raw_row.get(hc))
        if a not in (None, "", "Ø") and np.isfinite(h) and h >= AT:
            observed.add(a)

    if not observed and allele_cols and height_cols:
        for ac, hc in zip(sorted(allele_cols), sorted(height_cols)):
            a = canonical_allele(raw_row.get(ac))
            h = _to_float_safe(raw_row.get(hc))
            if a not in (None, "", "Ø") and np.isfinite(h) and h >= AT:
                observed.add(a)

    return observed


def load_donor(donor_id: str) -> pd.DataFrame:
    if donor_id not in _donor_cache:
        path = DONOR_DIR / f"{donor_id}.csv"
        df = pd.read_csv(path, sep=";")
        df["Allele1"] = df["Allele1"].apply(canonical_allele)
        df["Allele2"] = df["Allele2"].apply(canonical_allele)
        _donor_cache[donor_id] = df
    return _donor_cache[donor_id]


def get_expected_union_by_marker(major_donor_id: str, minor_donor_id: str) -> dict:
    exp = {}
    for did in (major_donor_id, minor_donor_id):
        d = load_donor(did)
        for _, r in d.iterrows():
            m = r["Marker"]
            a1 = canonical_allele(r["Allele1"])
            a2 = canonical_allele(r["Allele2"])
            s = exp.get(m, set())
            if a1 not in (None, "", "Ø"):
                s.add(a1)
            if a2 not in (None, "", "Ø"):
                s.add(a2)
            exp[m] = s
    return exp


def get_bad_loci_dropin_dropout_for_mixture(mixture: str, major_donor_id: str, minor_donor_id: str,
                                            AT: float, flag_dropout: bool = True, flag_dropin: bool = True) -> set:
    raw = load_raw_mixture_txt(mixture)
    if raw.empty or "Marker" not in raw.columns:
        print(f"[WARNING] Cannot determine bad loci for {mixture} (raw missing/empty).")
        return set()

    expected_union = get_expected_union_by_marker(major_donor_id, minor_donor_id)
    bad = set()

    for marker, g in raw.groupby("Marker"):
        raw_row = g.iloc[0]
        observed = extract_observed_raw_alleles_above_AT(raw_row, AT)
        expected = expected_union.get(marker, set())
        if not expected:
            continue

        if flag_dropout and (not expected.issubset(observed)):
            bad.add(marker)
        if flag_dropin and any(a not in expected for a in observed):
            bad.add(marker)

    return bad


# =========================
# BAD LOCI HANDLING METHODS
# =========================
def apply_zero_mode(loci_bad: set, true_log10: float, locus_df: pd.DataFrame,
                    totals: np.ndarray, locus_samples: dict):
    """Set bad loci contribution to 0 by removing them from true + totals, then zeroing arrays."""
    if not loci_bad:
        return float(true_log10), totals

    # True: subtract the true per-locus contributions at bad loci
    remove_true = 0.0
    if "marker" in locus_df.columns and "log10_LR" in locus_df.columns:
        mask = locus_df["marker"].isin(loci_bad)
        remove_true = float(np.nansum(locus_df.loc[mask, "log10_LR"].to_numpy(dtype=float)))
    true_adj = float(true_log10 - remove_true)

    # Samples: subtract per draw contributions and set those arrays to 0
    totals_adj = totals.astype(float).copy()
    for locus in loci_bad:
        if locus in locus_samples:
            totals_adj -= locus_samples[locus]
            locus_samples[locus] = np.zeros_like(locus_samples[locus])

    return true_adj, totals_adj


def apply_impute_mode(loci_bad: set, true_log10: float, locus_df: pd.DataFrame,
                      totals: np.ndarray, locus_samples_on: dict, locus_samples_off: dict):
    """Replace bad loci by ON-sample mean per locus."""
    if not loci_bad:
        return float(true_log10), totals

    true_adj = float(true_log10)
    totals_adj = totals.astype(float).copy()

    for locus in loci_bad:
        if locus not in locus_samples_on:
            continue

        mu = float(np.nanmean(np.asarray(locus_samples_on[locus], dtype=float)))

        # True: replace per-locus true contribution with mu
        mask = (locus_df["marker"] == locus) if ("marker" in locus_df.columns) else None
        if mask is not None and mask.any() and ("log10_LR" in locus_df.columns):
            true_val = float(locus_df.loc[mask, "log10_LR"].iloc[0])
            true_adj += (mu - true_val)

        # Samples: replace each draw value with mu
        if locus in locus_samples_off:
            old_vec = np.asarray(locus_samples_off[locus], dtype=float)
            totals_adj += (mu - old_vec)
            locus_samples_off[locus] = np.full_like(old_vec, mu)

    return true_adj, totals_adj


def apply_bad_loci_handling(mode: str, loci_bad: set, true_log10_on: float, locus_df: pd.DataFrame,
                           totals_on: np.ndarray, locus_samples_on: dict):
    """
    Returns:
      true_log10_corr, totals_corr, locus_samples_corr, label
    Does not mutate ON objects.
    """
    mode = mode.lower().strip()
    if mode not in {"zero", "impute"}:
        raise ValueError(f"Unknown mode: {mode}")

    totals_corr = totals_on.astype(float).copy()
    locus_corr = {k: v.copy() for k, v in locus_samples_on.items()}
    true_corr = float(true_log10_on)

    if mode == "zero":
        true_corr, totals_corr = apply_zero_mode(
            loci_bad=loci_bad,
            true_log10=true_corr,
            locus_df=locus_df,
            totals=totals_corr,
            locus_samples=locus_corr,
        )
        return true_corr, totals_corr, locus_corr, "ZEROED"

    if mode == "impute":
        true_corr, totals_corr = apply_impute_mode(
            loci_bad=loci_bad,
            true_log10=true_corr,
            locus_df=locus_df,
            totals=totals_corr,
            locus_samples_on=locus_samples_on,
            locus_samples_off=locus_corr,
        )
        return true_corr, totals_corr, locus_corr, "IMPUTED"

    raise RuntimeError("unreachable")


# =========================
# CHECK TABLES (mode-specific)
# =========================
def _mean_per_locus_dict(locus_samples: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"marker": loc, "mean": float(np.nanmean(np.asarray(arr, dtype=float)))}
        for loc, arr in locus_samples.items()
    ])


def print_locus_check_table_zero(mixture, contr, donor_id, loci_bad, locus_df,
                                true_on, true_corr, totals_on, totals_corr,
                                locus_on, locus_corr):
    true_per_locus = (
        locus_df[["marker", "log10_LR"]]
        .copy()
        .rename(columns={"log10_LR": "true_log10LR_on"})
    )

    samp_on = _mean_per_locus_dict(locus_on).rename(columns={"mean": "samp_mean_on"})
    samp_corr = _mean_per_locus_dict(locus_corr).rename(columns={"mean": "samp_mean_zeroed"})

    check = true_per_locus.merge(samp_on, on="marker", how="outer").merge(samp_corr, on="marker", how="outer")
    check["bad_locus"] = check["marker"].isin(loci_bad)
    check["true_log10LR_zeroed"] = np.where(check["bad_locus"], 0.0, check["true_log10LR_on"])

    check = check.sort_values(["bad_locus", "marker"], ascending=[False, True])
    check_to_print = check.head(int(PRINT_MAX_LOCI)) if PRINT_MAX_LOCI is not None else check

    print("\n" + "-" * 80)
    print(f"[LOCUS CHECK ZERO] {mixture} | {contr} -> {donor_id} | bad_loci={len(loci_bad)}")
    print(f"True total ON     = {true_on:.6f}")
    print(f"True total ZEROED = {true_corr:.6f}")
    print(f"Sample mean ON    = {float(np.mean(totals_on)):.6f}")
    print(f"Sample mean ZEROED= {float(np.mean(totals_corr)):.6f}")
    print("\nPer-locus values (bad loci first):")
    print(check_to_print[[
        "marker", "bad_locus", "true_log10LR_on", "true_log10LR_zeroed", "samp_mean_on", "samp_mean_zeroed"
    ]].to_string(index=False))
    print("-" * 80 + "\n")

    if ASSERT_NON_TOUCHED_UNCHANGED:
        unchanged = check.loc[~check["bad_locus"]].dropna(subset=["samp_mean_on", "samp_mean_zeroed"])
        if not unchanged.empty:
            diff = (unchanged["samp_mean_on"] - unchanged["samp_mean_zeroed"]).abs().max()
            if pd.notna(diff) and diff > 1e-12:
                raise RuntimeError(f"[ASSERT FAIL] Non-bad loci changed (ZERO) in {mixture} {contr}. max diff={diff}")

    if ASSERT_ZERO_MODE_IS_ZERO and loci_bad:
        turned = check.loc[check["bad_locus"]].dropna(subset=["samp_mean_zeroed"])
        if not turned.empty:
            max_abs = turned["samp_mean_zeroed"].abs().max()
            if pd.notna(max_abs) and max_abs > 1e-12:
                raise RuntimeError(f"[ASSERT FAIL] Bad loci not ~0 (ZERO) in {mixture} {contr}. max abs={max_abs}")


def print_locus_check_table_impute(mixture, contr, donor_id, loci_bad, locus_df,
                                  true_on, true_corr, totals_on, totals_corr,
                                  locus_on, locus_corr):
    true_per_locus = (
        locus_df[["marker", "log10_LR"]]
        .copy()
        .rename(columns={"log10_LR": "true_log10LR_on"})
    )

    samp_on = _mean_per_locus_dict(locus_on).rename(columns={"mean": "samp_mean_on"})
    samp_corr = _mean_per_locus_dict(locus_corr).rename(columns={"mean": "samp_mean_imputed"})

    check = true_per_locus.merge(samp_on, on="marker", how="outer").merge(samp_corr, on="marker", how="outer")
    check["bad_locus"] = check["marker"].isin(loci_bad)
    check["imputed_mu_on"] = check["samp_mean_imputed"]
    check["true_log10LR_imputed"] = np.where(check["bad_locus"], check["imputed_mu_on"], check["true_log10LR_on"])
    #check["imputed_mu_used"] = check["samp_mean_imputed"]   # this is the actual value used after correction
    #check["true_log10LR_imputed"] = np.where(check["bad_locus"], check["imputed_mu_used"], check["true_log10LR_on"])


    check = check.sort_values(["bad_locus", "marker"], ascending=[False, True])
    check_to_print = check.head(int(PRINT_MAX_LOCI)) if PRINT_MAX_LOCI is not None else check

    print("\n" + "-" * 80)
    print(f"[LOCUS CHECK IMPUTE] {mixture} | {contr} -> {donor_id} | bad_loci={len(loci_bad)}")
    print(f"True total ON      = {true_on:.6f}")
    print(f"True total IMPUTED = {true_corr:.6f}")
    print(f"Sample mean ON     = {float(np.mean(totals_on)):.6f}")
    print(f"Sample mean IMPUTED= {float(np.mean(totals_corr)):.6f}")
    print("\nPer-locus values (bad loci first):")
    print(check_to_print[[
        "marker", "bad_locus", "true_log10LR_on", "imputed_mu_on", "true_log10LR_imputed",
        "samp_mean_on", "samp_mean_imputed"
    ]].to_string(index=False))
    print("-" * 80 + "\n")

    if ASSERT_NON_TOUCHED_UNCHANGED:
        unchanged = check.loc[~check["bad_locus"]].dropna(subset=["samp_mean_on", "samp_mean_imputed"])
        if not unchanged.empty:
            diff = (unchanged["samp_mean_on"] - unchanged["samp_mean_imputed"]).abs().max()
            if pd.notna(diff) and diff > 1e-12:
                raise RuntimeError(f"[ASSERT FAIL] Non-bad loci changed (IMPUTE) in {mixture} {contr}. max diff={diff}")

    if ASSERT_IMPUTE_MODE_IS_MEAN and loci_bad:
        turned = check.loc[check["bad_locus"]].dropna(subset=["samp_mean_imputed", "imputed_mu_on"])
        if not turned.empty:
            diff = (turned["samp_mean_imputed"] - turned["imputed_mu_on"]).abs().max()
            if pd.notna(diff) and diff > 1e-12:
                raise RuntimeError(f"[ASSERT FAIL] Imputed loci != mu_on in {mixture} {contr}. max diff={diff}")


# =========================
# OUTLIER EXPORT HELPERS (kept, simplified)
# =========================
# def donor_locus_table(donor_id: str, role_prefix: str) -> pd.DataFrame:
#     d = load_donor(donor_id).copy()
#     out = d.rename(columns={"Marker": "marker", "Allele1": "a1", "Allele2": "a2"})[["marker", "a1", "a2"]].copy()
#     out["a1"] = out["a1"].apply(canonical_allele)
#     out["a2"] = out["a2"].apply(canonical_allele)
#     out[f"{role_prefix}_donor_id"] = donor_id
#     out[f"{role_prefix}_pair"] = out.apply(lambda r: canonical_pair(r["a1"], r["a2"]), axis=1)
#     return out[["marker", f"{role_prefix}_donor_id", f"{role_prefix}_pair"]]

def donor_locus_table(donor_id: str, role_prefix: str) -> pd.DataFrame:
    """
    Returns one row per marker with:
      marker,
      {role_prefix}_donor_id,
      {role_prefix}_a1, {role_prefix}_a2,
      {role_prefix}_pair
    """
    d = load_donor(donor_id).copy()

    out = d.rename(columns={"Marker": "marker", "Allele1": "a1", "Allele2": "a2"})[["marker", "a1", "a2"]].copy()
    out["a1"] = out["a1"].apply(canonical_allele)
    out["a2"] = out["a2"].apply(canonical_allele)

    out[f"{role_prefix}_donor_id"] = donor_id
    out[f"{role_prefix}_a1"] = out["a1"]
    out[f"{role_prefix}_a2"] = out["a2"]
    out[f"{role_prefix}_pair"] = out.apply(lambda r: canonical_pair(r["a1"], r["a2"]), axis=1)

    return out[["marker",
                f"{role_prefix}_donor_id",
                f"{role_prefix}_a1", f"{role_prefix}_a2",
                f"{role_prefix}_pair"]]



def build_raw_locus_table_for_mixture(mixture: str) -> pd.DataFrame:
    raw = load_raw_mixture_txt(mixture)
    if raw.empty or "Marker" not in raw.columns:
        return pd.DataFrame()

    raw2 = raw.copy()
    raw2["mixture"] = mixture
    raw2 = raw2.rename(columns={"Marker": "marker"})
    join_keys = {"mixture", "marker"}
    rename_map = {c: f"raw_{c}" for c in raw2.columns if c not in join_keys}
    raw2 = raw2.rename(columns=rename_map)
    raw2 = raw2.drop_duplicates(subset=["mixture", "marker"], keep="first")
    return raw2

def is_degenerate_sample(totals: np.ndarray, true_val: float, atol=1e-12, rtol=0.0) -> bool:
    """
    Degenerate if:
      (a) all sampled totals are (nearly) identical, AND
      (b) that constant equals the true value (within tolerance)
    """
    if totals is None:
        return False
    arr = np.asarray(totals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False

    const = np.allclose(arr, arr[0], atol=atol, rtol=rtol)
    if not const:
        return False

    return np.isclose(arr[0], float(true_val), atol=atol, rtol=rtol)

def ks_D_uniform_01_from_percentiles(percentiles_0_100: np.ndarray) -> float:
    """
    Kolmogorov–Smirnov D statistic against Uniform(0,1) for percentile values in [0,100].

    Returns:
      D = sup_x |F_n(x) - x|  (with x in [0,1])

    Notes:
    - This computes D only (no p-value), and does not require SciPy.
    - If array is empty -> np.nan
    """
    x = np.asarray(percentiles_0_100, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    # convert to [0,1]
    u = x / 100.0

    # clamp just in case of tiny numerical overshoots
    u = np.clip(u, 0.0, 1.0)

    # empirical CDF at sorted points
    u_sorted = np.sort(u)
    n = u_sorted.size
    i = np.arange(1, n + 1)

    # KS components
    D_plus = np.max(i / n - u_sorted)
    D_minus = np.max(u_sorted - (i - 1) / n)
    D = float(max(D_plus, D_minus))
    return D

def sort_genotype_df(df):
    def allele_sort_value(x):
        try:
            return float(x)
        except:
            return float("inf")

    return (
        df
        .assign(
            Allele1_sort=df["Allele 1"].apply(allele_sort_value),
            Allele2_sort=df["Allele 2"].apply(allele_sort_value),
        )
        .sort_values(
            by=["Locus", "Allele1_sort", "Allele2_sort"]
        )
        .drop(columns=["Allele1_sort", "Allele2_sort"])
    )
# =========================
# PROCESS ONE MIXTURE
# =========================
def process_mixture(mixture_type, dataset_number, replicate_number):
    mixture = f"{replicate_number}_{dataset_number}{mixture_type}2"
    output_folder = OUT_ROOT / "mixtures" / mixture

    if not output_folder.is_dir():
        print(f"⚠ Skipping mixture {mixture} — folder not found.")
        return None

    print("\n" + "=" * 60)
    print(f"PROCESSING MIX {mixture}: dataset {dataset_number}, replicate {replicate_number}")
    print("=" * 60)

    true_major, true_minor = TRUE_DONORS[dataset_number]
    df_major = load_donor(true_major)
    df_minor = load_donor(true_minor)

    # # Check mixture proportions (ensure donor1 is major under H2)
    # json_path = output_folder / "results.json"
    # with open(json_path, "r") as f:
    #     res = json.load(f)
    # mp = res["hypothesesResults"]["H2"]["modelParameters"]["mixtureProportions"]
    # max_idx = max(range(len(mp)), key=lambda i: mp[i])

    # print(f"H2 mixture proportions: {mp}")
    # if max_idx != 0:
    #     raise RuntimeError(
    #         f"❌ FATAL: In mixture {mixture}, true donor 1 is NOT the major under H2.\n"
    #         f"    Largest mixture proportion is C{max_idx+1}={mp[max_idx]:.6f}."
    #     )
    # print("✔ Correct: true donor 1 is the major contributor under H2.")

    # Detect bad loci once per mixture
    bad_loci = set()
    if DETECT_BAD_LOCI:
        bad_loci = get_bad_loci_dropin_dropout_for_mixture(
            mixture=mixture,
            major_donor_id=true_major,
            minor_donor_id=true_minor,
            AT=ANALYTICAL_THRESHOLD,
            flag_dropout=FLAG_DROPOUT,
            flag_dropin=FLAG_DROPIN,
        )
        print(f"[INFO] {mixture}: bad_loci={len(bad_loci)} (dropin/dropout by raw+donors)")
        if bad_loci:
            print(f"       bad_loci = {sorted(bad_loci)}")

    # Load marginal results
    results_path = output_folder / "results_marginal_prior_LR.csv"
    try:
        results_df = pd.read_csv(results_path, sep='\t') #turn off if key error Allele 1
        results_df[COL_A1] = results_df[COL_A1].apply(canonical_allele)
    except:
        results_df = pd.read_csv(results_path)
        results_df[COL_A1] = results_df[COL_A1].apply(canonical_allele)
    results_df[COL_A2] = results_df[COL_A2].apply(canonical_allele)
       
    
    # Choose mapping (Unknown1/Unknown2 -> donor IDs) by max total log10LR
    mappings_to_try = [
        [("Unknown1", true_major, df_major), ("Unknown2", true_minor, df_minor)],
        [("Unknown1", true_minor, df_minor), ("Unknown2", true_major, df_major)],
    ]

    mapping_results = []
    for mapping in mappings_to_try:
        total_log10 = 0.0
        infos = []
        for contr, donor_id, donor_df in mapping:
            log10LR, LR, n_fb, n_fd, locus_details, locus_log10_list = compute_true_donor_LR(
                results_df, hypothesis=HYPOTHESIS, contributor=contr, donor_df=donor_df
            )
            total_log10 += log10LR
            infos.append({
                "contributor": contr,
                "donor_id": donor_id,
                "donor_df": donor_df,
                "log10_LR": log10LR,
                "LR": LR,
                "n_fallback": n_fb,
                "n_full_drop": n_fd,
                "locus_details": locus_details,
                "locus_log10_list": locus_log10_list,
            })
        mapping_results.append({"mapping": mapping, "total_log10": total_log10, "infos": infos})

    best = max(mapping_results, key=lambda m: m["total_log10"])
    chosen = [(c, d) for (c, d, _) in best["mapping"]]
    default = [("Unknown1", true_major), ("Unknown2", true_minor)]
    is_swap = chosen != default

    swap_record = None
    if is_swap:
        alt = min(mapping_results, key=lambda m: m["total_log10"])
        swap_record = {
            "mixture": mixture,
            "dataset": dataset_number,
            "replicate": replicate_number,
            "mixture_type": mixture_type,
            "selected_mapping": chosen,
            "total_log10_best": best["total_log10"],
            "total_log10_alt": alt["total_log10"],
        }
        print("🔄 SWAP occurred: contributors reassigned based on LR.")
    else:
        print("✔ No swap needed (default mapping chosen).")

    print(f"Mixture type {mixture_type}: selected mapping:")
    for info in best["infos"]:
        print(f"  {info['contributor']} -> {info['donor_id']} (log10 LR = {info['log10_LR']:.3f})")

    # Determine which correction modes to run
    if BAD_LOCI_MODE == "both":
        modes_to_run = ["zero", "impute"]
    else:
        modes_to_run = [BAD_LOCI_MODE]

    percentile_rows = []
    fallback_rows = []
    outlier_dfs = []

    # Process each contributor
    for info in best["infos"]:
        contr = info["contributor"]
        donor_id = info["donor_id"]
        donor_df = info["donor_df"]
        true_log10_on = info["log10_LR"]
        true_LR_on = info["LR"]
        n_fallback = info["n_fallback"]
        n_full_drop = info["n_full_drop"]
        locus_details = info["locus_details"]
        locus_log10_list = info["locus_log10_list"]

        locus_df = pd.DataFrame(locus_details)

        # dominance
        L1_max, L2_max = locus_dominance_metrics(locus_log10_list)

        # role label
        true_donor_label = "true_donor1" if donor_id == TRUE_DONORS[dataset_number][0] else "true_donor2"

        print(f"\nTrue donor LR ({contr} -> {donor_id} / {true_donor_label}): log10 LR = {true_log10_on:.3f}, LR = {true_LR_on:.3e}")
        print(f"    Fallback loci (a,Ø): {n_fallback}, full-drop loci (Ø,Ø): {n_full_drop}")
        print(f"    Locus dominance L1_max = {L1_max:.3f}, L2_max = {L2_max:.3f}")

        fallback_rows.append({
            "mixture": mixture,
            "mixture_type": mixture_type,
            "dataset": dataset_number,
            "replicate": replicate_number,
            "contributor": contr,
            "true_donor": true_donor_label,
            "donor_id": donor_id,
            "n_fallback": n_fallback,
            "n_full_drop": n_full_drop,
            "L1_max": L1_max,
            "L2_max": L2_max,
            "log10_LR_on": true_log10_on,
        })
        
        #results_df = sort_genotype_df(results_df) 

        # ---- sample ON once ----
        totals_on, locus_on = sample_N_genotypes_with_locus_log10(
            results_df, hypothesis=HYPOTHESIS, contributor=contr, N=N_GLOBAL
        )

        # per-locus mu/sd (for outliers table only)
        rows = []
        for locus, arr in locus_on.items():
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            mu = float(np.mean(arr)) if arr.size else np.nan
            sd_raw = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
            sd = max(0.01, sd_raw) if np.isfinite(sd_raw) else np.nan
            rows.append({"marker": locus, "mu": mu, "sd": sd})
        ci_df = pd.DataFrame(rows)
        locus_df = locus_df.merge(ci_df, on="marker", how="left")

        locus_df["mixture"] = mixture
        locus_df["dataset"] = dataset_number
        locus_df["replicate"] = replicate_number
        locus_df["mixture_type"] = mixture_type
        locus_df["contributor"] = contr
        locus_df["true_donor"] = true_donor_label
        locus_df["donor_id"] = donor_id
        locus_df["donor_role"] = "major" if true_donor_label == "true_donor1" else "minor"

        locus_df["z"] = (locus_df["log10_LR"] - locus_df["mu"]) / locus_df["sd"]
        locus_df["delta"] = locus_df["log10_LR"] - locus_df["mu"]

        outliers = locus_df[locus_df["delta"] <= DELTA_THRESHOLD].copy()
        if not outliers.empty:
            outliers["method"] = "ON"   # NEW (so merge can use method if you want)
            outlier_dfs.append(outliers)

        # ---- ON percentile row ----
        
        deg_on = is_degenerate_sample(totals_on, true_log10_on, atol=DEGENERATE_ATOL, rtol=DEGENERATE_RTOL)
        pct_on = 100.0 * np.mean(totals_on <= true_log10_on)
        mu_mixture_on = np.mean(totals_on)
        sigma_mixture_on = np.std(totals_on, ddof=1)
        percentile_rows.append({
            "mixture": mixture,
            "dataset": dataset_number,
            "replicate": replicate_number,
            "mixture_type": mixture_type,
            "contributor": contr,
            "true_donor": true_donor_label,
            "donor_id": donor_id,
            "method": "ON",
            "percentile": pct_on,
            "true_log10_LR": true_log10_on,
            "mu_mixture":mu_mixture_on,
            "sigma_mixture":sigma_mixture_on,
            "n_bad_loci": len(bad_loci),
            "degenerate_sample": deg_on,   # NEW
        })
        print(f"Percentile (ON) {true_donor_label} ({contr}) in {mixture}: {pct_on:.1f}th")

        # ---- correction runs ----
        for mode in modes_to_run:
            true_corr, totals_corr, locus_corr, label = apply_bad_loci_handling(
                mode=mode,
                loci_bad=bad_loci,
                true_log10_on=true_log10_on,
                locus_df=locus_df,
                totals_on=totals_on,
                locus_samples_on=locus_on,
            )
            deg_corr = is_degenerate_sample(totals_corr, true_corr, atol=DEGENERATE_ATOL, rtol=DEGENERATE_RTOL)

            pct_corr = 100.0 * np.mean(totals_corr <= true_corr)
            mu_mixture_corr = np.mean(totals_corr)
            sigma_mixture_corr = np.std(totals_corr, ddof=1)

            percentile_rows.append({
                "mixture": mixture,
                "dataset": dataset_number,
                "replicate": replicate_number,
                "mixture_type": mixture_type,
                "contributor": contr,
                "true_donor": true_donor_label,
                "donor_id": donor_id,
                "method": label,  # "ZEROED" or "IMPUTED"
                "percentile": pct_corr,
                "true_log10_LR": true_corr,
                "mu_mixture" : mu_mixture_corr,
                "sigma_mixture":sigma_mixture_corr,
                "n_bad_loci": len(bad_loci),
                "degenerate_sample": deg_corr,   # NEW
            })
            print(f"Percentile ({label}) {true_donor_label} ({contr}) in {mixture}: {pct_corr:.1f}th")

            # ---- check tables ----
            if PRINT_LOCUS_CHECK_TABLE:
                if mode == "zero":
                    print_locus_check_table_zero(
                        mixture, contr, donor_id, bad_loci, locus_df,
                        true_log10_on, true_corr, totals_on, totals_corr, locus_on, locus_corr
                    )
                elif mode == "impute":
                    print_locus_check_table_impute(
                        mixture, contr, donor_id, bad_loci, locus_df,
                        true_log10_on, true_corr, totals_on, totals_corr, locus_on, locus_corr
                    )

            # ---- per-mixture histogram ----
            if SHOW_MIXTURE_HISTOGRAMS:
                plt.figure(figsize=(10, 6))
                ax = plt.gca()
                ax.hist(totals_on, bins=60, edgecolor="black", alpha=0.35, label="Sampled donor logLRs")
                #ax.hist(totals_corr, bins=60, edgecolor="black", alpha=0.35, label=f"Sampled totals ({label})")
                # ax.axvline(true_log10_on, linestyle="--", linewidth=2, color="red",
                #            label=f"True total (ON) = {true_log10_on:.2f}")
                # ax.axvline(true_corr, linestyle="--", linewidth=2, color="green",
                #            label=f"True total ({label}) = {true_corr:.2f}")
                # ax.axvline(mu_mixture_on, linestyle="--", linewidth=2, color="blue",
                #            label=f"E[logLR] = {mu_mixture_on:.2f}")
                ax.axvline(true_log10_on, linestyle="--", linewidth=2, color="red",
                           label=f"True donor logLR = {true_log10_on:.2f}")
                # ax.axvline(true_corr, linestyle="--", linewidth=2, color="green",
                #            label=f"True total ({label}) = {true_corr:.2f}")
                ax.set_xlabel("log10 LR")
                ax.set_ylabel("Count")
                #ax.set_title(f"{mixture} – {contr} (bad loci: {len(bad_loci)} | {label})")
                ax.set_title(f"Single mixture {mixture} – {contr} (minor donor)")
                ax.legend()
                plt.show()

    return {
        "mixture": mixture,
        "percentile_rows": percentile_rows,
        "fallback_rows": fallback_rows,
        "swap_record": swap_record,
        "outlier_dfs": outlier_dfs,
        "bad_loci_count": len(bad_loci),
    }


# =========================
# MAIN
# =========================
def main():
    #start = time.time()
    
    
    
# def main():
    # --- record run params ---
    run_params = {
        # plotting / outputs
        "SHOW_MIXTURE_HISTOGRAMS": SHOW_MIXTURE_HISTOGRAMS,
        "SHOW_PERCENTILE_HISTS": SHOW_PERCENTILE_HISTS,
        "SAVE_FILES": SAVE_FILES,
        "version": version,

        # core LR sampling
        "N_GLOBAL": N_GLOBAL,
        "HYPOTHESIS": HYPOTHESIS,
        "DELTA_THRESHOLD": DELTA_THRESHOLD,

        # bad loci detection
        "DETECT_BAD_LOCI": DETECT_BAD_LOCI,
        "FLAG_DROPOUT": FLAG_DROPOUT,
        "FLAG_DROPIN": FLAG_DROPIN,
        "ANALYTICAL_THRESHOLD": ANALYTICAL_THRESHOLD,

        # bad loci handling
        "MAX_BAD_LOCI_FOR_HIST": MAX_BAD_LOCI_FOR_HIST,
        "BAD_LOCI_MODE": BAD_LOCI_MODE,

        # debug flags
        "PRINT_LOCUS_CHECK_TABLE": PRINT_LOCUS_CHECK_TABLE,
        "PRINT_MAX_LOCI": PRINT_MAX_LOCI,
        "ASSERT_NON_TOUCHED_UNCHANGED": ASSERT_NON_TOUCHED_UNCHANGED,
        "ASSERT_ZERO_MODE_IS_ZERO": ASSERT_ZERO_MODE_IS_ZERO,
        "ASSERT_IMPUTE_MODE_IS_MEAN": ASSERT_IMPUTE_MODE_IS_MEAN,

        # run selection
        "RUN_MODE": RUN_MODE,
        "MIXTURE_TYPES": MIXTURE_TYPES,
        "DATASETS": DATASETS,
        "REPLICATES": REPLICATES,

        # key paths (stringify Paths so json works too)
        "BASE": str(BASE),
        "OUT_ROOT": str(OUT_ROOT),
        "RESULTS_DIR": str(RESULTS_DIR),
        "DONOR_DIR": str(DONOR_DIR),
        "RAW_MIX_DIR": str(RAW_MIX_DIR),
        
        # ---- degenerate sampling filter (percentile histograms) ----
        "EXCLUDE_DEGENERATE_FOR_HIST": str(EXCLUDE_DEGENERATE_FOR_HIST), # = True   # NEW: remove mixtures where all sampled totals == true total
        "DEGENERATE_ATOL" : str(DEGENERATE_ATOL), # = 1e-12             # NEW: tolerance for float equality
        "DEGENERATE_RTOL" : str(DEGENERATE_RTOL), # = 0.0               # NEW: keep strict unless you want relative tolerance too

"COMPUTE_KS_FOR_PERCENTILE_HISTS": str(COMPUTE_KS_FOR_PERCENTILE_HISTS),# = True
"ANNOTATE_KS_ON_PLOTS": str(ANNOTATE_KS_ON_PLOTS), # = True
"SAVE_KS_TABLE": str(SAVE_KS_TABLE), # = True

    }

    if SAVE_FILES:
        write_run_parameters_txt(RESULTS_DIR, run_params, filename=f"run_params_read_csv_{version}.txt")
        write_run_parameters_json(RESULTS_DIR, run_params, filename=f"run_params_read_csv_{version}.json")

    # ... rest of your existing main() ...
   
    

    all_percentile_rows = []
    all_fallback_rows = []
    swap_records = []
    outlier_loci_dfs = []

    for mixture_type in MIXTURE_TYPES:
        for dataset_number in DATASETS:
            for replicate_number in REPLICATES:
                result = process_mixture(mixture_type, dataset_number, replicate_number)
                if result is None:
                    continue

                all_percentile_rows.extend(result["percentile_rows"])
                all_fallback_rows.extend(result["fallback_rows"])
                if result["swap_record"] is not None:
                    swap_records.append(result["swap_record"])
                outlier_loci_dfs.extend(result["outlier_dfs"])

    percentile_df = pd.DataFrame(all_percentile_rows)
    fallback_df = pd.DataFrame(all_fallback_rows)

    # Compare per-mixture ZEROED vs IMPUTED percentiles
    wide = (percentile_df
            .pivot_table(index=["mixture","contributor","true_donor","donor_id","n_bad_loci"],
                         columns="method",
                         values="percentile",
                         aggfunc="first")
            .reset_index())
    
    # Only where both exist
    wide = wide.dropna(subset=["ZEROED","IMPUTED"])
    
    wide["abs_diff"] = (wide["IMPUTED"] - wide["ZEROED"]).abs()
    
    print("\nMinor (true_donor2) ZEROED vs IMPUTED percentile diffs:")
    minor = wide[wide["true_donor"] == "true_donor2"]
    print(minor["abs_diff"].describe())
    print("Count nonzero diffs:", (minor["abs_diff"] > 0).sum(), "out of", len(minor))
    
    print("\nMajor (true_donor1) ZEROED vs IMPUTED percentile diffs:")
    major = wide[wide["true_donor"] == "true_donor1"]
    print(major["abs_diff"].describe())
    print("Count nonzero diffs:", (major["abs_diff"] > 0).sum(), "out of", len(major))



    # -------- Swap summary --------
    print("\n=======================================")
    print("         SWAP SUMMARY")
    print("=======================================")
    print(f"Total number of swaps: {len(swap_records)}")
    if swap_records:
        for rec in swap_records:
            print(f"  {rec['mixture']} (dataset {rec['dataset']}, rep {rec['replicate']}): {rec['selected_mapping']}")
            print(f"     total log10 LR (chosen)   = {rec['total_log10_best']:.3f}")
            print(f"     total log10 LR (rejected) = {rec['total_log10_alt']:.3f}")

    # -------- Percentile histograms across mixtures --------
    if SHOW_PERCENTILE_HISTS and not percentile_df.empty:
        methods_present = ["ON"]
        if BAD_LOCI_MODE in {"zero", "both"}:
            methods_present.append("ZEROED")
        if BAD_LOCI_MODE in {"impute", "both"}:
            methods_present.append("IMPUTED")

        # By default: ON includes all mixtures; corrected excludes those with too many bad loci
        # def get_vals(true_donor_label: str, method: str):
        #     sub = percentile_df[(percentile_df["true_donor"] == true_donor_label) & (percentile_df["method"] == method)]
        #     if method != "ON":
        #         sub = sub[sub["n_bad_loci"] <= MAX_BAD_LOCI_FOR_HIST]
        #     return sub["percentile"].to_numpy(), len(sub)
        
        def get_vals(true_donor_label: str, method: str):
            sub = percentile_df[
                (percentile_df["true_donor"] == true_donor_label) &
                (percentile_df["method"] == method)
            ].copy()
        
            # existing rule: corrected exclude too many bad loci
            if method != "ON":
                sub = sub[sub["n_bad_loci"] <= MAX_BAD_LOCI_FOR_HIST]
        
            # NEW rule: optionally exclude degenerate-sampled mixtures from the histogram
            if EXCLUDE_DEGENERATE_FOR_HIST and "degenerate_sample" in sub.columns:
                sub = sub[~sub["degenerate_sample"].fillna(False)]
        
            return sub["percentile"].to_numpy(), len(sub)

        if EXCLUDE_DEGENERATE_FOR_HIST and "degenerate_sample" in percentile_df.columns:
            print("\n[INFO] Degenerate-sample counts (excluded from percentile histograms):")
            print(
                percentile_df.groupby(["true_donor", "method"])["degenerate_sample"]
                .apply(lambda s: int(np.sum(s.fillna(False))))
                .sort_index()
                .to_string()
            )
        
        ks_rows = []  # collect per-panel KS stats

        # Make a grid: rows = donors (major/minor), cols = methods
        ncols = len(methods_present)
        fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8), sharex=True, sharey=True)

        bins = np.linspace(0, 100, 11)

        donors = [("true_donor1", "major"), ("true_donor2", "minor")]
        for r, (dlabel, drole) in enumerate(donors):
            for c, method in enumerate(methods_present):
                ax = axes[r, c] if ncols > 1 else axes[r]
                # vals, n = get_vals(dlabel, method)
                # if n > 0:
                #     ax.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
                #     ax.text(0.98, 0.95, f"n={n}", transform=ax.transAxes, ha="right", va="top")
                # else:
                #     ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                vals, n = get_vals(dlabel, method)

                D = np.nan
                if COMPUTE_KS_FOR_PERCENTILE_HISTS and n > 0:
                    D = ks_D_uniform_01_from_percentiles(vals)
                
                # store summary row
                if COMPUTE_KS_FOR_PERCENTILE_HISTS:
                    ks_rows.append({
                        "true_donor": dlabel,           # "true_donor1" / "true_donor2"
                        "donor_role": drole,            # "major" / "minor"
                        "method": method,               # "ON" / "ZEROED" / "IMPUTED"
                        "n_used": int(n),
                        "ks_D_uniform01": D,
                        "excluded_rule_corrected_max_bad_loci": int(MAX_BAD_LOCI_FOR_HIST),
                        "exclude_degenerate_for_hist": bool(EXCLUDE_DEGENERATE_FOR_HIST),
                    })
                
                if n > 0:
                    ax.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
                
                    # n + D annotation
                    ax.text(0.98, 0.95, f"n={n}", transform=ax.transAxes, ha="right", va="top")
                
                    if ANNOTATE_KS_ON_PLOTS and np.isfinite(D):
                        ax.text(0.98, 0.87, f"D={D:.3f}", transform=ax.transAxes, ha="right", va="top")
                else:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

                ax.set_title(f"{dlabel} ({drole}) – {method}")
                ax.set_xlim(0, 100)
                ax.set_xlabel("Percentile of true donor LR among sampled LRs")
                ax.set_ylabel("Number of mixtures")

        fig.suptitle(
            f"Percentile histograms (mode={BAD_LOCI_MODE}; corrected exclude n_bad_loci>{MAX_BAD_LOCI_FOR_HIST})",
            y=1.02
        )
        plt.tight_layout()
        if SAVE_FILES:
            plt.savefig(RESULTS_DIR/ f"percentile_histograms_{BAD_LOCI_MODE}_{version}.png", dpi=300, bbox_inches="tight")
            print("[INFO] Saved percentile histogram")
        plt.show()
        
        if COMPUTE_KS_FOR_PERCENTILE_HISTS:
            ks_df = pd.DataFrame(ks_rows)

            print("\n=======================================")
            print("   KS D vs Uniform(0,1) for percentiles")
            print("=======================================")
            if not ks_df.empty:
                # nice pivot for quick inspection
                pretty = (ks_df
                          .pivot_table(index=["donor_role", "true_donor"],
                                       columns="method",
                                       values=["ks_D_uniform01", "n_used"],
                                       aggfunc="first"))
                print(pretty.to_string())
            else:
                print("No KS rows computed (no data).")

            if SAVE_FILES and SAVE_KS_TABLE and not ks_df.empty:
                ks_out = RESULTS_DIR / f"ks_uniformity_percentiles_{BAD_LOCI_MODE}_{version}.csv"
                ks_df.to_csv(ks_out, index=False)
                print(f"[INFO] Saved KS uniformity table -> {ks_out}")


    # # -------- Outlier summary (kept) --------
    # print("\n=======================================")
    # print(f"   OUTLIER LOCI (delta <= {DELTA_THRESHOLD})")
    # print("=======================================")

    # if not outlier_loci_dfs:
    #     print("No outlier loci found.")
    # else:
    #     outlier_df = pd.concat(outlier_loci_dfs, ignore_index=True)
    #     print(f"Total outlier locus instances: {len(outlier_df)}")
    #     print("\nTop loci by frequency:")
    #     print(outlier_df["marker"].value_counts().head(15))

    # print(f"\nDone. Runtime: {time.time() - start:.1f}s")

    # # If you want these dataframes accessible after running in Spyder:
    # return percentile_df, fallback_df, outlier_df
    
    # -------- Outlier summary (OLD-SCRIPT STYLE) --------
    print("\n=======================================")
    print(f"   OUTLIER LOCI (delta <= {DELTA_THRESHOLD})")
    print("=======================================")
    
    outlier_df = pd.DataFrame()
    
    if not outlier_loci_dfs:
        print("No outlier loci found.")
    else:
        outlier_df = pd.concat(outlier_loci_dfs, ignore_index=True)
    
    
        # 1) Attach mixture-level ON percentile + ON true LR to each outlier locus
        pct_keys = ["mixture", "dataset", "replicate", "mixture_type", "contributor", "true_donor"]
        
        pct_on = percentile_df[percentile_df["method"] == "ON"].copy()
        
        # Ensure uniqueness: 1 row per key (prevents row multiplication)
        pct_on = pct_on.sort_values(pct_keys).drop_duplicates(subset=pct_keys, keep="first")
        
        outlier_df = outlier_df.merge(
            pct_on[pct_keys + ["percentile", "true_log10_LR"]],
            on=pct_keys,
            how="left"
        )
        
        # Rename to old-style names if you want
        outlier_df = outlier_df.rename(columns={
            "percentile": "mixture_percentile",
            "true_log10_LR": "mixture_true_log10_LR",
        })
        outlier_df["mixture_true_LR"] = 10 ** outlier_df["mixture_true_log10_LR"]

    
    
    
    
        # # 1) Add mixture-level percentile + mixture true LR columns to each outlier locus
        # #    (This matches your old merge: outlier_df.merge(percentile_df[merge_cols], ...))
        # merge_cols = [
        #     "mixture", "dataset", "replicate", "mixture_type", "contributor", "true_donor",
        #     #"method",            # keep method if your new script has ON/ZEROED/IMPUTED
        #     "percentile",        # your new script uses "percentile"
        #     "true_log10_LR",     # your new script uses "true_log10_LR"
        # ]
    
        # # Only keep columns that actually exist (robust if you renamed anything)
        # merge_cols = [c for c in merge_cols if c in percentile_df.columns]
    
        # outlier_df = outlier_df.merge(
        #     percentile_df[merge_cols],
        #     on=[c for c in ["mixture", "dataset", "replicate", "mixture_type", "contributor", "true_donor", "method"]
        #         if c in merge_cols],
        #     how="left"
        # )
    
        # # Optional: create old-style names too (if you want exact old column names)
        # if "percentile" in outlier_df.columns and "mixture_percentile" not in outlier_df.columns:
        #     outlier_df = outlier_df.rename(columns={
        #         "percentile": "mixture_percentile",
        #         "true_log10_LR": "mixture_true_log10_LR",
        #     })
    
        # # If you also want mixture_true_LR like in old script:
        # if "mixture_true_log10_LR" in outlier_df.columns and "mixture_true_LR" not in outlier_df.columns:
        #     outlier_df["mixture_true_LR"] = 10 ** outlier_df["mixture_true_log10_LR"]
            
            
            
    
        # 2) Add major/minor donor IDs based on dataset mapping (exactly as old script)
        outlier_df["major_donor_id"] = outlier_df["dataset"].map(lambda d: TRUE_DONORS[int(d)][0])
        outlier_df["minor_donor_id"] = outlier_df["dataset"].map(lambda d: TRUE_DONORS[int(d)][1])
    
        # 3) Merge donor genotype tables (major/minor alleles + pairs) onto outlier_df
        major_ids = sorted(outlier_df["major_donor_id"].dropna().unique().tolist())
        minor_ids = sorted(outlier_df["minor_donor_id"].dropna().unique().tolist())
    
        major_tables = [donor_locus_table(did, "major") for did in major_ids]
        minor_tables = [donor_locus_table(did, "minor") for did in minor_ids]
    
        major_df = pd.concat(major_tables, ignore_index=True).drop_duplicates(["major_donor_id", "marker"])
        minor_df = pd.concat(minor_tables, ignore_index=True).drop_duplicates(["minor_donor_id", "marker"])
    
        outlier_df = outlier_df.merge(
            major_df,
            left_on=["major_donor_id", "marker"],
            right_on=["major_donor_id", "marker"],
            how="left",
        )
    
        outlier_df = outlier_df.merge(
            minor_df,
            left_on=["minor_donor_id", "marker"],
            right_on=["minor_donor_id", "marker"],
            how="left",
        )
    
        # 4) Attach raw locus data (raw_* columns) for mixtures present in outliers
        mixtures_needed = sorted(outlier_df["mixture"].dropna().unique().tolist())
    
        raw_tables = []
        for mix in mixtures_needed:
            t = build_raw_locus_table_for_mixture(mix)
            if not t.empty:
                raw_tables.append(t)
    
        if raw_tables:
            raw_locus_df = pd.concat(raw_tables, ignore_index=True)
            outlier_df = outlier_df.merge(raw_locus_df, on=["mixture", "marker"], how="left")
        else:
            print("[WARNING] No raw locus tables could be built (missing .txt files?).")
    
    
        # 5) Drop all-NaN columns (exactly as you did)
        #outlier_df = outlier_df[outlier_df["method"] == "ON"].copy()
        outlier_df = outlier_df.loc[:, outlier_df.notna().any()]
    
        print(f"Total outlier locus instances: {len(outlier_df)}")
        print("\nTop loci by frequency:")
        print(outlier_df["marker"].value_counts().head(15))
    
    return percentile_df, fallback_df, outlier_df



if __name__ == "__main__":
    
    percentile_df, fallback_df, outlier_df = main()
    if SAVE_FILES:
        percentile_df.to_csv(RESULTS_DIR / f'percentile_df_{version}.csv',index=False)
        outlier_df.to_csv(RESULTS_DIR / f"outlier_df_{version}.csv", index=False)