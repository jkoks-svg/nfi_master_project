
import json
import yaml
#import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import platform
import sys
#import textwrap
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance

#mle_or_dnax = 'dnax'  # used only for simulated-theta_s filtered folders
#ACTUAL_REWORK_LR_MODE = 'bayesian_vector_draws'

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


def parse_vector_cell(x):
    """
    Parse a CSV cell containing a JSON vector, e.g. "[1.2, 3.4, 5.6]".
    """
    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, list):
        return np.asarray(x, dtype=float)

    if pd.isna(x):
        return np.asarray([], dtype=float)

    return np.asarray(json.loads(str(x)), dtype=float)


def log10_mean_exp_from_log10(log10_values):
    """
    Stable computation of:

        log10(mean(10 ** log10_values))

    This is used to average full-profile LRs over MCMC draws
    after loci have already been combined.
    """
    x = np.asarray(log10_values, dtype=float)

    if x.size == 0:
        return np.nan

    finite = np.isfinite(x)
    if not np.any(finite):
        return -np.inf

    x_finite = x[finite]
    xmax = np.max(x_finite)

    return float(xmax + np.log10(np.mean(10 ** (x_finite - xmax))))


def lr_draws_to_log10(lr_draws):
    """
    Convert LR(a)_draws to log10 scale.
    Zero LR becomes -inf.
    Negative LR is invalid.
    """
    lr_draws = np.asarray(lr_draws, dtype=float)

    if np.any(lr_draws < 0):
        raise ValueError("LR draws contain negative values, which is invalid.")

    return np.where(lr_draws > 0, np.log10(lr_draws), -np.inf)



def sample_N_genotypes_with_locus_log10(df, hypothesis, contributor, N=1000):
    """
    Bayesian-vector version for rework mixtures.

    For each sampled profile:
      1) sample one MCMC draw theta_m;
      2) for every locus, sample a genotype using that draw's marginal posterior probabilities;
      3) collect the selected genotype's full LR(a)_draws vector;
      4) combine loci draw-by-draw;
      5) average the full-profile LR over MCMC draws as the final step.

    Returns:
      total_log10_LRs: (N,) array
      locus_log10: dict {locus_name: (N,) array of per-locus diagnostic log10LR summaries}

    Important:
      total_log10_LRs is not the sum of locus_log10 summaries.
      The total is:
          log10(mean_m product_locus LR_locus,m)
    """
    sub = df[
        (df[COL_HYP] == hypothesis) &
        (df[COL_CONTR] == contributor)
    ].copy()

    grouped = []

    for locus, g in sub.groupby(COL_MARKER):
        g = g.copy()

        prob_mat = np.vstack([
            parse_vector_cell(x)
            for x in g[COL_POST_DRAWS]
        ])

        lr_mat = np.vstack([
            parse_vector_cell(x)
            for x in g[COL_LR_DRAWS]
        ])

        if prob_mat.shape != lr_mat.shape:
            raise ValueError(
                f"Shape mismatch at locus {locus}: "
                f"prob_mat={prob_mat.shape}, lr_mat={lr_mat.shape}"
            )

        if prob_mat.shape[1] == 0:
            raise ValueError(f"No MCMC draws found at locus {locus}")

        log10_lr_mat = np.vstack([
            lr_draws_to_log10(row)
            for row in lr_mat
        ])

        # rows = genotype rows, columns = MCMC draws
        grouped.append((locus, prob_mat, log10_lr_mat))

    if not grouped:
        raise ValueError(
            f"No genotype rows found for hypothesis={hypothesis}, contributor={contributor}"
        )

    n_draws_set = {prob_mat.shape[1] for _, prob_mat, _ in grouped}
    if len(n_draws_set) != 1:
        raise ValueError(
            f"Not all loci have the same number of MCMC draws: {n_draws_set}"
        )

    n_mcmc_draws = n_draws_set.pop()

    locus_log10 = {locus: np.empty(N, dtype=float) for locus, *_ in grouped}
    total_log10 = np.empty(N, dtype=float)

    for i in range(N):
        # This theta draw is only used to sample a coherent multilocus genotype profile.
        theta_sample_idx = np.random.randint(n_mcmc_draws)

        profile_log10_draws = None

        for locus, prob_mat, log10_lr_mat in grouped:
            w = prob_mat[:, theta_sample_idx].astype(float)
            w_sum = w.sum()

            if not np.isfinite(w_sum) or w_sum <= 0:
                raise ValueError(
                    f"Invalid posterior weights at locus {locus}, "
                    f"MCMC draw {theta_sample_idx}: sum={w_sum}"
                )

            w = w / w_sum

            genotype_idx = np.random.choice(len(w), p=w)

            chosen_locus_log10_draws = log10_lr_mat[genotype_idx, :]

            # Diagnostic per-locus Bayesian LR summary.
            # Do not sum these to obtain the total.
            locus_log10[locus][i] = log10_mean_exp_from_log10(
                chosen_locus_log10_draws
            )

            if profile_log10_draws is None:
                profile_log10_draws = chosen_locus_log10_draws.copy()
            else:
                profile_log10_draws = profile_log10_draws + chosen_locus_log10_draws

        # Correct Bayesian averaging:
        # first combine loci within each MCMC draw,
        # then average the full-profile LR over MCMC draws.
        total_log10[i] = log10_mean_exp_from_log10(profile_log10_draws)

    return total_log10, locus_log10

def safe_wasserstein_distance(x, y):
    """
    Computes the 1D Wasserstein / earth mover distance between two samples.
    Non-finite values are removed.

    Important:
    - If x and y are log10 LR values, the distance is in log10 LR units.
    - This compares the full empirical distributions, not only their means.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return np.nan

    return wasserstein_distance(x, y)

def safe_signed_wasserstein_distance(x_simulated, y_actual):
    """
    Signed Wasserstein-like distance.

    Magnitude:
        1D Wasserstein distance between simulated and actual distributions.

    Sign:
        positive if the actual distribution has to move right to match simulated.
        negative if the actual distribution has to move left to match simulated.

    Assumes x_simulated and y_actual are on the same scale, e.g. log10 LR.
    """
    x = np.asarray(x_simulated, dtype=float)
    y = np.asarray(y_actual, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return np.nan

    w = wasserstein_distance(x, y)

    mean_shift_needed = np.mean(y) - np.mean(x)

    if np.isclose(mean_shift_needed, 0.0):
        return 0.0

    return np.sign(mean_shift_needed) * w

def prediction_interval_metrics(samples, observed, alpha=0.10):
    """
    Coverage and interval score for an empirical predictive distribution.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0 or not np.isfinite(observed):
        return {
            "pi_lower": np.nan,
            "pi_upper": np.nan,
            "pi_width": np.nan,
            "covered": np.nan,
            "interval_score": np.nan,
        }

    lower = float(np.quantile(x, alpha / 2))
    upper = float(np.quantile(x, 1 - alpha / 2))
    observed = float(observed)

    width = upper - lower
    covered = int(lower <= observed <= upper)

    penalty_low = (2 / alpha) * (lower - observed) if observed < lower else 0.0
    penalty_high = (2 / alpha) * (observed - upper) if observed > upper else 0.0

    interval_score = width + penalty_low + penalty_high

    return {
        "pi_lower": lower,
        "pi_upper": upper,
        "pi_width": width,
        "covered": covered,
        "interval_score": interval_score,
    }


# =========================
# TRUE donor LR computation (with Ø fallbacks)
# =========================
def compute_true_donor_LR(results_df, hypothesis, contributor, donor_df,
                          donor_locus_col="Marker", donor_a1_col="Allele1", donor_a2_col="Allele2"):
    """
    Bayesian-vector true donor LR for rework mixtures.

    For the known donor genotype:
      1) find the matching genotype row per locus;
      2) collect its LR(a)_draws vector;
      3) sum log10 LR over loci for each MCMC draw;
      4) average the full-profile LR over MCMC draws only at the final step.
    """
    sub = results_df[
        (results_df[COL_HYP] == hypothesis) &
        (results_df[COL_CONTR] == contributor)
    ].copy()

    sub["canon_pair"] = sub.apply(
        lambda r: canonical_pair(r[COL_A1], r[COL_A2]),
        axis=1
    )

    profile_log10_draws = None
    missing_markers = []

    n_fallback = 0
    n_full_drop = 0
    locus_details = []
    locus_log10_list = []

    for _, row in donor_df.iterrows():
        locus = row[donor_locus_col]
        a1 = canonical_allele(row[donor_a1_col])
        a2 = canonical_allele(row[donor_a2_col])
        donor_pair = canonical_pair(a1, a2)

        locus_rows = sub[sub[COL_MARKER] == locus]

        best = None
        match_type = None

        # 1) Exact genotype match
        exact = locus_rows[locus_rows["canon_pair"] == donor_pair]

        if not exact.empty:
            best = exact.iloc[0]
            match_type = "exact"

        # 2) Fallback: one allele plus Ø
        if best is None:
            fallback_pairs = [
                canonical_pair(a1, "Ø"),
                canonical_pair(a2, "Ø"),
            ]

            fallback = locus_rows[locus_rows["canon_pair"].isin(fallback_pairs)]

            if not fallback.empty:
                fallback = fallback.copy()

                # Only for selecting between possible fallback candidates.
                fallback["_log10_lr_summary"] = fallback[COL_LR_DRAWS].apply(
                    lambda x: log10_mean_exp_from_log10(
                        lr_draws_to_log10(parse_vector_cell(x))
                    )
                )

                best = fallback.sort_values(
                    "_log10_lr_summary",
                    ascending=False
                ).iloc[0]

                match_type = "fallback_oneØ"
                n_fallback += 1

        # 3) Full dropout: Ø,Ø
        if best is None:
            full = locus_rows[
                locus_rows["canon_pair"] == canonical_pair("Ø", "Ø")
            ]

            if not full.empty:
                best = full.iloc[0]
                match_type = "full_dropØØ"
                n_full_drop += 1

        # 4) No match
        if best is None:
            missing_markers.append(locus)
            locus_details.append({
                "marker": locus,
                "donor_pair": donor_pair,
                "used_pair": None,
                "match_type": "no_match",
                "log10_LR": np.nan,
                "LR": np.nan,
            })
            continue

        lr_draws = parse_vector_cell(best[COL_LR_DRAWS])
        locus_log10_draws = lr_draws_to_log10(lr_draws)

        # Diagnostic locus-level Bayesian LR summary.
        # The profile LR is not obtained by summing these summaries.
        locus_log10_summary = log10_mean_exp_from_log10(locus_log10_draws)
        locus_lr_summary = (
            10 ** locus_log10_summary
            if np.isfinite(locus_log10_summary)
            else np.nan
        )

        locus_log10_list.append(locus_log10_summary)

        locus_details.append({
            "marker": locus,
            "donor_pair": donor_pair,
            "used_pair": best["canon_pair"],
            "match_type": match_type,
            "log10_LR": locus_log10_summary,
            "LR": locus_lr_summary,
        })

        if profile_log10_draws is None:
            profile_log10_draws = locus_log10_draws.copy()
        else:
            if len(profile_log10_draws) != len(locus_log10_draws):
                raise ValueError(
                    f"MCMC vector length mismatch at locus {locus}: "
                    f"profile has {len(profile_log10_draws)}, "
                    f"locus has {len(locus_log10_draws)}"
                )

            profile_log10_draws = profile_log10_draws + locus_log10_draws

    if missing_markers:
        print(f"[WARNING] No matching genotype rows for markers: {missing_markers}")

    if profile_log10_draws is None:
        log10_total = np.nan
        LR_total = np.nan
    else:
        # Correct Bayesian averaging:
        # average over MCMC draws only after all loci have been combined.
        log10_total = log10_mean_exp_from_log10(profile_log10_draws)
        LR_total = 10 ** log10_total if np.isfinite(log10_total) else np.nan

    return log10_total, LR_total, n_fallback, n_full_drop, locus_details, locus_log10_list


def load_donor(donor_id: str) -> pd.DataFrame:
    if donor_id not in _donor_cache:
        path = DONOR_DIR / f"{donor_id}.csv"
        df = pd.read_csv(path, sep=";")
        df["Allele1"] = df["Allele1"].apply(canonical_allele)
        df["Allele2"] = df["Allele2"].apply(canonical_allele)
        _donor_cache[donor_id] = df
    return _donor_cache[donor_id]


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

# def ks_D_uniform_01_from_percentiles(percentiles_0_100: np.ndarray) -> float:
#     """
#     Kolmogorov–Smirnov D statistic against Uniform(0,1) for percentile values in [0,100].

#     Returns:
#       D = sup_x |F_n(x) - x|  (with x in [0,1])

#     Notes:
#     - This computes D only (no p-value), and does not require SciPy.
#     - If array is empty -> np.nan
#     """
#     x = np.asarray(percentiles_0_100, dtype=float)
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         return np.nan

#     # convert to [0,1]
#     u = x / 100.0

#     # clamp just in case of tiny numerical overshoots
#     u = np.clip(u, 0.0, 1.0)

#     # empirical CDF at sorted points
#     u_sorted = np.sort(u)
#     n = u_sorted.size
#     i = np.arange(1, n + 1)

#     # KS components
#     D_plus = np.max(i / n - u_sorted)
#     D_minus = np.max(u_sorted - (i - 1) / n)
#     D = float(max(D_plus, D_minus))
#     return D

def ks_D_uniform_01_from_percentiles(percentiles_0_100: np.ndarray, return_location=False):
    """
    Kolmogorov–Smirnov D statistic against Uniform(0,1) for percentile values in [0,100].

    If return_location=True, also returns the x-location in percentile scale [0,100]
    where the maximum KS deviation is reached.
    """
    x = np.asarray(percentiles_0_100, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        if return_location:
            return np.nan, np.nan, None
        return np.nan

    u = np.clip(x / 100.0, 0.0, 1.0)

    u_sorted = np.sort(u)
    n = u_sorted.size
    i = np.arange(1, n + 1)

    D_plus_vals = i / n - u_sorted
    D_minus_vals = u_sorted - (i - 1) / n

    idx_plus = np.argmax(D_plus_vals)
    idx_minus = np.argmax(D_minus_vals)

    D_plus = D_plus_vals[idx_plus]
    D_minus = D_minus_vals[idx_minus]

    if D_plus >= D_minus:
        D = float(D_plus)
        ks_x = float(u_sorted[idx_plus] * 100.0)
        direction = "D+"
    else:
        D = float(D_minus)
        ks_x = float(u_sorted[idx_minus] * 100.0)
        direction = "D-"

    if return_location:
        return D, ks_x, direction

    return D

def plot_histogram(ax, totals_on, mu_mixture_on, true_log10_on,
                   replicate_data, mixture, contr, show_all=True, replicate=None):

    binwidth = 0.5

    bins_real = np.arange(
        totals_on.min(),
        totals_on.max()+binwidth,
        binwidth
    )

    # # Base histogram (rework)
    # ax.hist(totals_on, bins=bins_real, edgecolor="black", alpha=0.35,
    #         label="Samples after rework", color='blue')

    # ax.axvline(mu_mixture_on, linestyle="--", linewidth=2, color="blue",
    #            label=f"Average after rework = {mu_mixture_on:.2f}")

    # True LR line
    ax.axvline(true_log10_on, linestyle="--", linewidth=2, color="red",
               label=f"True donor logLR = {true_log10_on:.2f}")

    # Loop over replicates
    for rep, df in replicate_data.items():
        color = COLOR_MAP.get(rep, 'black')
        mean_sim = df['true_log10_LR'].mean()
        bins_sims = np.arange(
            df['true_log10_LR'].min(),
            df['true_log10_LR'].max() + binwidth,
            binwidth
        )

        ax.hist(df['true_log10_LR'], bins=bins_sims,
                color=color, edgecolor="black", alpha=0.35,
                label=f"Simulations of mixture {rep}_{mixture}")

        # ax.axvline(mean_sim, linestyle="--", linewidth=2, color=color,
        #            label=f"Average of mixture {rep}_{mixture} = {mean_sim:.2f}")

    ax.set_xlabel("log10 LR")
    ax.set_ylabel("Count")
    if replicate is None:
        ax.set_title(f"{mixture} – {contr} (all replicates)")
    else:
        ax.set_title(f"{mixture} – {contr} (replicate {replicate})")
    ax.legend()

# =========================
# PROCESS ONE MIXTURE
# =========================
def process_mixture(mixture_type, dataset_number, OUT_ROOT, N_GLOBAL, FILTERED_DF_BASE, N_CONTRIBUTORS,version,mle_or_dnax):
    
    mixture = f"{dataset_number}{mixture_type}{N_CONTRIBUTORS}"
    output_folder = OUT_ROOT / "mixtures" / mixture
    print(output_folder)
    if not output_folder.is_dir():
        print(f"⚠ Skipping mixture {mixture} — folder not found.")
        return None

    print("\n" + "=" * 60)
    print(f"PROCESSING MIX {mixture}: dataset {dataset_number}")
    print("=" * 60)

    if N_CONTRIBUTORS == 2:
        TRUE_DONORS = {
            1: ["1A", "1B"],#,"1C","1D","1E"],
            2: ["2F", "2G"],#, "2H", "2I", "2J"],
            3: ["3K", "3L"],#,"3M","3N","3O"],
            4: ["4P", "4Q"],#, "4R", "4S", "4T"],
            5: ["5U", "5V"],#, "5W", "5X", "5Y"],
            6: ["6Z", "6AA"]##, "6AB", "6AC", "6AD"],
        }
    elif N_CONTRIBUTORS == 3 and mixture_type in ["A","B","C"]:
        TRUE_DONORS = {
            1: ["1A", "1B","1C"],#,"1D","1E"],
            2: ["2F", "2G","2H"],# "2I", "2J"],
            3: ["3K", "3L","3M"],#,"3N","3O"],
            4: ["4P", "4Q","4R"],# "4S", "4T"],
            5: ["5U", "5V","5W"],# "5X", "5Y"],
            6: ["6Z", "6AA","6AB"]# "6AC", "6AD"],
        }
    elif N_CONTRIBUTORS == 3 and mixture_type in ["D","E"]:
        TRUE_DONORS = {
            1: ["1A", "1C","1B"],#,"1D","1E"],
            2: ["2F", "2H","2G"],# "2I", "2J"],
            3: ["3K", "3M","3L"],#,"3N","3O"],
            4: ["4P", "4R","4Q"],# "4S", "4T"],
            5: ["5U", "5W","5V"],# "5X", "5Y"],
            6: ["6Z", "6AB","6AA"]# "6AC", "6AD"],
        }

    all_donors = TRUE_DONORS[dataset_number]
    donor_ids = all_donors[:N_CONTRIBUTORS]


    # # Load marginal results
    # #results_path = output_folder / "results_marginal_prior_LR.csv"
    # try:
    #     results_path = output_folder / "results_marginal_prior_LR.csv"
    # except:
    #     results_path = output_folder / "results_marginal_deconvolutions_with_prior_and_LR.csv"
    # results_df = pd.read_csv(results_path, sep="\t")
    
    results_path1 = output_folder / "results_marginal_prior_LR.csv"
    results_path2 = output_folder / "results_marginal_deconvolutions_with_prior_and_LR.csv"
    try:
        results_df = pd.read_csv(results_path1)#, sep="\t")
    except:
        results_df = pd.read_csv(results_path2, sep="\t")
    
    required_cols = [
        COL_HYP, COL_CONTR, COL_MARKER, COL_A1, COL_A2,
        COL_POST_DRAWS, COL_LR_DRAWS,
    ]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise KeyError(
            f"Missing required Bayesian-vector columns in {results_path1}: {missing}. "
            "read_csv_traces_corrected2.py expects actual lab rework output with "
            "Marginal_probability_draws and LR(a)_draws, as produced by the MCMC/Bayesian pipeline."
        )

    results_df[COL_A1] = results_df[COL_A1].apply(canonical_allele)
    results_df[COL_A2] = results_df[COL_A2].apply(canonical_allele)


    #mapping = [("Unknown1", true_major, df_major), ("Unknown2", true_minor, df_minor)] #FIX
    mapping = []
    for i, donor_id in enumerate(donor_ids):
        contr = f"Unknown{i+1}"
        donor_df = load_donor(donor_id)
        mapping.append((contr, donor_id, donor_df))
    
    
    
    infos = []
    for contr, donor_id, donor_df in mapping:
        log10LR, LR, n_fb, n_fd, locus_details, locus_log10_list = compute_true_donor_LR(
            results_df, hypothesis=HYPOTHESIS, contributor=contr, donor_df=donor_df
        )
        infos.append({
            "contributor": contr,
            "donor_id": donor_id,
            #"donor_df": donor_df,
            "log10_LR": log10LR,
            "LR": LR,
            "n_fallback": n_fb,
            "n_full_drop": n_fd,
            #"locus_details": locus_details,
            #"locus_log10_list": locus_log10_list,
        })


    percentile_rows = []

    # Process each contributor
    for info in infos: # best["infos"]:
        contr = info["contributor"]
        donor_id = info["donor_id"]
        #donor_df = info["donor_df"]
        true_log10_on = info["log10_LR"]
        true_LR_on = info["LR"]
        n_fallback = info["n_fallback"]
        n_full_drop = info["n_full_drop"]

        donor_index = donor_ids.index(donor_id)
        true_donor_label = f"true_donor{donor_index+1}"

        print(f"\nTrue donor LR ({contr} -> {donor_id} / {true_donor_label}): log10 LR = {true_log10_on:.3f}, LR = {true_LR_on:.3e}")
        print(f"    Fallback loci (a,Ø): {n_fallback}, full-drop loci (Ø,Ø): {n_full_drop}")

        # ---- sample ON once ----
        totals_on, locus_on = sample_N_genotypes_with_locus_log10(
            results_df, hypothesis=HYPOTHESIS, contributor=contr, N=N_GLOBAL
        )

        # ---- ON percentile row ----
        
        deg_on = is_degenerate_sample(totals_on, true_log10_on, atol=DEGENERATE_ATOL, rtol=DEGENERATE_RTOL)
        pct_on = 100.0 * np.mean(totals_on <= true_log10_on)
        mu_mixture_on = np.mean(totals_on)
        sigma_mixture_on = np.std(totals_on, ddof=1)

        version_filtered = version
        
        for replicate_number in replicate_numbers:
            filtered_df_path = FILTERED_DF_BASE / rf"{replicate_number}_{mixture}\results_samples_{mle_or_dnax}_{version_filtered}\filtered_df_{contr}_{version_filtered}.csv"
            if not filtered_df_path.exists():
                print(f"[WARNING] Missing file: {filtered_df_path}")
                #continue
                filtered_df_path = FILTERED_DF_BASE / rf"{replicate_number}_{mixture}\results_samples_{version_filtered}\filtered_df_{contr}_{version_filtered}.csv"
            #Path(rf"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_100simulations\mixtures\{replicate_number}_{mixture}\results_samples_{version_filtered}\filtered_df_{version_filtered}.csv")  
            # filtered_df = pd.read_csv(filtered_df_path)
            # mean_sim = filtered_df['true_log10_LR'].mean()
            # pct_sim = 100.0 * np.mean(filtered_df['true_log10_LR'] <= true_log10_on)
            
            filtered_df = pd.read_csv(filtered_df_path)

            actual_rework_log10 = filtered_df["true_log10_LR"].to_numpy(dtype=float)
            
            mean_sim = np.nanmean(actual_rework_log10)
            pct_sim = 100.0 * np.mean(actual_rework_log10 <= true_log10_on)
            
            # Wasserstein / earth mover distance between:
            # - sampled rework LR distribution: totals_on
            # - actual rework LR distribution: filtered_df["true_log10_LR"]
            wasserstein_log10_LR = safe_wasserstein_distance(totals_on, actual_rework_log10)
            signed_wasserstein_log10_LR = safe_signed_wasserstein_distance(
                totals_on,
                actual_rework_log10
            )
            
            pi95 = prediction_interval_metrics(
                samples=actual_rework_log10,
                observed=true_log10_on,
                alpha=0.05
            )

            percentile_rows.append({
                "mixture": mixture,
                "dataset": dataset_number,
                #"replicate": replicate_number,
                "mixture_type": mixture_type,
                "contributor": contr,
                "true_donor": true_donor_label,
                "donor_id": donor_id,
                "method": "ON",
                "percentile": pct_on,
                "true_log10_LR": true_log10_on,
                "mu_mixture":mu_mixture_on,
                "sigma_mixture":sigma_mixture_on,
                #"n_bad_loci": len(bad_loci),
                "degenerate_sample": deg_on,
                "replicate_number":replicate_number,
                "mean_sim": mean_sim,
                "pct_sim": pct_sim,# NEW,
                "wasserstein_log10_LR": wasserstein_log10_LR,
                "signed_wasserstein_log10_LR": signed_wasserstein_log10_LR,
                "pi95_lower": pi95["pi_lower"],
                "pi95_upper": pi95["pi_upper"],
                "pi95_width": pi95["pi_width"],
                "covered_95": pi95["covered"],
                "interval_score_95": pi95["interval_score"],
            })


        print(f"Percentile (ON) {true_donor_label} ({contr}) in {mixture}: {pct_on:.1f}th")
                    
        
        if SHOW_MIXTURE_HISTOGRAMS:

            # ---- Load all replicate data once ----
            replicate_data = {}
        
            for replicate_number in replicate_numbers:
                path = FILTERED_DF_BASE / rf"{replicate_number}_{mixture}\results_samples_{mle_or_dnax}_{version_filtered}\filtered_df_{contr}_{version_filtered}.csv"
                
                if path.exists():
                    replicate_data[replicate_number] = pd.read_csv(path)
                else:
                    path = FILTERED_DF_BASE / rf"{replicate_number}_{mixture}\results_samples_{version_filtered}\filtered_df_{contr}_{version_filtered}.csv"
                    replicate_data[replicate_number] = pd.read_csv(path)
            # ---- Combined plot ----
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
            plot_histogram(ax, totals_on, mu_mixture_on, true_log10_on,
                           replicate_data, mixture, contr, replicate=None)
        
            plt.show()
        
            # ---- Separate plots per replicate ----
            for rep, df in replicate_data.items():
                plt.figure(figsize=(10, 6))
                ax = plt.gca()
        
                plot_histogram(ax, totals_on, mu_mixture_on, true_log10_on,
                               {rep: df}, mixture, contr, replicate = rep)
                plt.show()
                

    return {
        "mixture": mixture,
        "percentile_rows": percentile_rows,

    }






def compare_with_lab_combined_profiles(
    #FILTERED_DF_BASE,
    FILTERED_DF_ROOT,
    OUT_ROOT,
    CONFIG_YAML,
    SAVE_FILES,
    version,
    N_GLOBAL,
    MIXTURE_TYPES,
    DATASETS,
    mle_or_dnax
):
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    
    N_CONTRIBUTORS = int(cfg['dnastatistx']['N_CONTRIBUTORS'])  # or 3, 4, 5

    FILTERED_DF_BASE = FILTERED_DF_ROOT / "mixtures"
    #start = time.time()
    #RESULTS_DIR = OUT_ROOT / f"results_{version}"
    RESULTS_DIR = FILTERED_DF_ROOT / f"results_{version}"
   
    
    if SAVE_FILES:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    
    
# def main():
    # --- record run params ---
    run_params = {
        # plotting / outputs
        "SHOW_MIXTURE_HISTOGRAMS": SHOW_MIXTURE_HISTOGRAMS,
        #"SHOW_PERCENTILE_HISTS": SHOW_PERCENTILE_HISTS,
        "SAVE_FILES": SAVE_FILES,
        "version": version,

        # core LR sampling
        "N_GLOBAL": N_GLOBAL,
        "HYPOTHESIS": HYPOTHESIS,
        "DELTA_THRESHOLD": DELTA_THRESHOLD,

        # key paths (stringify Paths so json works too)
        "BASE": str(BASE),
        "OUT_ROOT": str(OUT_ROOT),
        "RESULTS_DIR": str(RESULTS_DIR),
        "DONOR_DIR": str(DONOR_DIR),
        #"RAW_MIX_DIR": str(RAW_MIX_DIR),
        
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


    all_percentile_rows = []

    for mixture_type in MIXTURE_TYPES:
        for dataset_number in DATASETS:
            
            #for replicate_number in REPLICATES:
            result = process_mixture(mixture_type, dataset_number, 
                                     OUT_ROOT, N_GLOBAL, FILTERED_DF_BASE, N_CONTRIBUTORS,version,mle_or_dnax)#, replicate_number)
            if result is None:
                continue

            all_percentile_rows.extend(result["percentile_rows"])

    percentile_df = pd.DataFrame(all_percentile_rows)
    
    coverage_by_contributor = (
        percentile_df
        .groupby("contributor")
        .agg(
            n=("covered_95", "size"),
            coverage_95=("covered_95", "mean"),
            mean_interval_score_95=("interval_score_95", "mean"),
            median_interval_score_95=("interval_score_95", "median"),
            mean_pi95_width=("pi95_width", "mean"),
        )
        .reset_index()
    )
    
    print("\nPrediction interval summary by contributor")
    print("-" * 80)
    print(coverage_by_contributor.to_string(index=False))
    
    if SAVE_FILES:
        coverage_by_contributor.to_csv(
            RESULTS_DIR / f"coverage_interval_score_by_contributor_{version}.csv",
            index=False
        )
    
    
    def get_prop(row):
        key = (N_CONTRIBUTORS, row["mixture_type"], row["contributor"])
        return MIXTURE_TYPE_MAP.get(key, np.nan)
    percentile_df["mixture_prop"] = percentile_df.apply(get_prop, axis=1)
    percentile_df["N_CONTRIBUTORS"] = N_CONTRIBUTORS
    
    if SAVE_FILES:
        percentile_df.to_csv(RESULTS_DIR / f'percentile_df_{version}.csv', index=False)
        # outlier_df.to_csv(RESULTS_DIR / f"outlier_df_{version}.csv", index=False)
    
    wrong_mixtures = ["1A2", '2A2', '3A2', '4A2', '6A2', '1C2', '4C2'] 
     
    subset_no_major_raw = percentile_df [
        (percentile_df["mixture_prop"] <= 0.7) &
        (~percentile_df["mixture"].isin(wrong_mixtures))
    ]
    
    title = "All contributors normal phi<=0.7 wrong removed"
    subset_no_major = subset_no_major_raw["pct_sim"].dropna()
    
    plt.figure()

    # histogram
    # plt.hist(subset, bins=10)
    # plt.xlim(0, 100)
    plt.hist(subset_no_major, bins=np.arange(0, 101, 10))

    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    
    plt.xlabel("Predicted-distribution percentile of true minor donor rework LR")
    plt.ylabel("Frequency")
    
    # ---- KS statistic ----
    #ks_D = ks_D_uniform_01_from_percentiles(subset_no_major.values)
    ks_D, ks_x, ks_direction = ks_D_uniform_01_from_percentiles(
        subset_no_major.values,
        return_location=True
    )
    
    # plt.axvline(
    #     ks_x,
    #     color="red",
    #     linestyle="--",
    #     linewidth=2,
    #     label=f"max KS at {ks_x:.1f}% ({ks_direction})"
    # )
    # plt.legend()
    
    # expected KS under uniform ~ 1/sqrt(n)
    n = len(subset_no_major)
    ks_ref = 1.36 / np.sqrt(n) if n > 0 else np.nan
    
    # title with KS
    #plt.title(f"{title} pct_sim\nKS D = {ks_D:.3f}") # (ref ≈ {ks_ref:.3f})")
    plt.title(
        f"{title} pct_sim\n"
        f"KS D = {ks_D:.3f} at {ks_x:.1f}% ({ks_direction})"
    )
    plt.grid()
    if SAVE_FILES:
        plt.savefig(RESULTS_DIR/"percentile_hist.png")
    plt.show()
    
    # plt.figure()

    # plt.scatter(subset_no_major_raw["mixture_prop"].dropna(), subset_no_major_raw["pct_sim"].dropna(), alpha=0.3)
    
    # plt.xlabel("Mixture proportion")
    # plt.ylabel("pct_sim")
    # plt.title("Unknown2 and Unknown3 mixture proportion vs percentile")
    
    # plt.show()

    plt.figure()
    
    plt.scatter(subset_no_major_raw["pct_sim"].dropna(),subset_no_major_raw["mixture_prop"].dropna(), alpha=0.3)
    
    plt.ylabel("Mixture proportion")
    plt.xlabel("pct_sim")
    plt.title(f"{title} mixture proportion vs percentile")
    plt.grid()
    if SAVE_FILES:
        plt.savefig(RESULTS_DIR/"percentile_mixture_prop_scatter.png")
    plt.show()

    
    
    for i in range(N_CONTRIBUTORS):
        contr = f"Unknown{i+1}"
        
        subset_raw = percentile_df[
            percentile_df["contributor"] == contr
        ]
        subset = subset_raw["pct_sim"].dropna()
    
        if subset.empty:
            print(f"[WARNING] No data for {contr}")
            continue
        
        plt.figure()

        # histogram
        # plt.hist(subset, bins=10)
        # plt.xlim(0, 100)
        plt.hist(subset, bins=np.arange(0, 101, 10))
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, 10))
        
        plt.xlabel("Predicted-distribution percentile of true minor donor rework LR")
        plt.ylabel("Frequency")
        
        # ---- KS statistic ----
        #ks_D = ks_D_uniform_01_from_percentiles(subset.values)
        ks_D, ks_x, ks_direction = ks_D_uniform_01_from_percentiles(
            subset.values,
            return_location=True
        )
        # plt.axvline(
        #     ks_x,
        #     color="red",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"max KS at {ks_x:.1f}% ({ks_direction})"
        # )
        # plt.legend()
        # expected KS under uniform ~ 1/sqrt(n)
        n = len(subset)
        ks_ref = 1.36 / np.sqrt(n) if n > 0 else np.nan
        
        # title with KS
        #plt.title(f"{contr} pct_sim\nKS D = {ks_D:.3f}at {ks_x:.1f}% ({ks_direction})") # (ref ≈ {ks_ref:.3f})")
        plt.grid()
#f"KS D = {ks_D:.3f} at {ks_x:.1f}% ({ks_direction})"
        plt.show()
        
        plt.figure()

        plt.scatter( subset_raw["pct_sim"].dropna(),subset_raw["mixture_prop"].dropna(), alpha=0.3)
        
        plt.ylabel("Mixture proportion")
        plt.xlabel("pct_sim")
        plt.grid()
        plt.title(f"{contr} mixture proportion vs percentile")
        
        plt.show()
        
        
    for i in range(N_CONTRIBUTORS):
        contr = f"Unknown{i+1}"        
        for key, value in sorted(MIXTURE_TYPE_MAP.items(), key=lambda x: x[1]):
            n_contr, mixture_type, contributor = key
        
            if key[0] != N_CONTRIBUTORS or key[2]!=contr:
                continue
        
            subset = percentile_df[
                (percentile_df["contributor"] == contributor) &
                (percentile_df["mixture_type"] == mixture_type)
            ]["pct_sim"].dropna()
        
            if subset.empty:
                print(f"[WARNING] No data for {contributor}, {mixture_type}")
                continue
        
            plt.figure()
        
            # histogram
            # plt.hist(subset, bins=10)
            # plt.xlim(0, 100)
            
            plt.hist(subset, bins=np.arange(0, 101, 10))
            plt.xlim(0, 100)
            plt.xticks(np.arange(0, 101, 10))
            
            plt.xlabel("Predicted-distribution percentile of true minor donor rework LR")
            plt.ylabel("Frequency")
        
            # ---- KS statistic ----
            #ks_D = ks_D_uniform_01_from_percentiles(subset.values)
            ks_D, ks_x, ks_direction = ks_D_uniform_01_from_percentiles(
                subset.values,
                return_location=True
            )
            # plt.axvline(
            #     ks_x,
            #     color="red",
            #     linestyle="--",
            #     linewidth=2,
            #     label=f"max KS at {ks_x:.1f}% ({ks_direction})"
            # )
            # plt.legend()
        
            n = len(subset)
            ks_ref = 1.36 / np.sqrt(n) if n > 0 else np.nan
        
            # include dictionary value (mixture proportion)
            plt.title(
                f"N={n_contr}, mixture {mixture_type}, {contributor}\n"
                f"prop={value:.3f}, KS D={ks_D:.3f} at {ks_x:.1f}% ({ks_direction})"
            )
        
            plt.grid()
            plt.show()
            
    
    # for i in range(N_CONTRIBUTORS):
    #     contr = f"Unknown{i+1}"
    
    #     subset = percentile_df[
    #         (percentile_df["contributor"] == contr) &
    #         (~percentile_df["pct_sim"].isna())
    #     ]
    
    #     if subset.empty:
    #         print(f"[WARNING] No data for {contr}")
    #         continue
    
    #     x = subset["percentile"].values
    #     y = subset["pct_sim"].values
    
    #     # correlation
    #     r, p_value = pearsonr(x, y)
    #     print(f"[{contr}] Pearson r: {r}, p-value: {p_value}")
    
    #     # regression
    #     slope, intercept = np.polyfit(x, y, 1)
    #     regression_line = slope * x + intercept
    
    #     # plot
    #     plt.figure()
    #     plt.scatter(x, y)
    #     plt.plot(x, regression_line)
    
    #     plt.xlabel("Percentile")
    #     plt.ylabel("pct_sim")
    #     plt.title(f"Scatterplot of Percentile vs pct_sim ({contr})")
    
    #     plt.show()
        
    return percentile_df
        
        
    

np.random.seed(12345)

# =========================
# CONFIG
# =========================

SHOW_MIXTURE_HISTOGRAMS = True  # per-mixture ON vs corrected totals
#SHOW_PERCENTILE_HISTS = True      # across-mixture percentile histograms


HYPOTHESIS = "H2"
DELTA_THRESHOLD = np.inf




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


replicate_numbers = [1,2,3]

COLOR_MAP = {
    1: 'green',
    2: 'orange',
    3: 'purple'
}

MIXTURE_TYPE_MAP = {
    (2, 'A',"Unknown1") : 300/450,
    (2, 'A',"Unknown2") : 150/450,
    (2, 'B',"Unknown1") : 300/330,
    (2, 'B',"Unknown2") : 30/330,
    (2, 'C',"Unknown1") : 150/300,
    (2, 'C',"Unknown2") : 150/300,
    (2, 'D',"Unknown1") : 150/180,
    (2, 'D',"Unknown2") : 30/180,
    (2, 'E',"Unknown1") : 600/630,
    (2, 'E',"Unknown2") : 30/630,
    (3, 'A',"Unknown1") : 300/600,
    (3, 'A',"Unknown2") : 150/600,
    (3, 'A',"Unknown3") : 150/600,
    (3, 'B',"Unknown1") : 300/360,
    (3, 'B',"Unknown2") : 30/360,
    (3, 'B',"Unknown3") : 30/360,
    (3, 'C',"Unknown1") : 150/360,
    (3, 'C',"Unknown2") : 150/360,
    (3, 'C',"Unknown3") : 60/360,
    (3, 'D',"Unknown1") : 150/240,
    (3, 'D',"Unknown2") : 30/240,
    (3, 'D',"Unknown3") : 60/240,
    (3, 'E',"Unknown1") : 600/690,
    (3, 'E',"Unknown2") : 30/690,
    (3, 'E',"Unknown3") : 60/690,
    }

# -------------------------
# Columns in results file
# -------------------------
COL_HYP = "Hypothesis"
COL_CONTR = "Contributor"
COL_MARKER = "Locus"
COL_A1 = "Allele 1"
COL_A2 = "Allele 2"
# Bayesian vector columns for actual lab rework output
COL_POST_DRAWS = "Marginal_probability_draws"
COL_LR_DRAWS = "LR(a)_draws"

# =========================
# CACHES
# =========================
_raw_mix_cache = {}    # mixture -> raw df
_donor_cache = {}      # donor_id -> donor df

BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
DONOR_DIR = BASE / "Dataset PP6FC Mixtures" / "Donoren"


if __name__ == "__main__":
    
    traces_folder_root = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces")
    FILTERED_DF_ROOT = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_runF_debug21")  
    SCRIPT_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
    RESOURCES_PATH = SCRIPT_PATH / "resources"
    CONFIG_YAML = RESOURCES_PATH / "config_2p_new_dnax.yaml"
    SAVE_FILES_2 = False
    version_2 = 'v1'
    N_GLOBAL_2 = 20
    MIXTURE_TYPES = ["A", "B"] #, "C", "D", "E"]
    DATASETS = [1] #,2,3,4,5,6]

    compare_with_lab_combined_profiles(
        FILTERED_DF_ROOT = FILTERED_DF_ROOT,
        OUT_ROOT=traces_folder_root,
        CONFIG_YAML = CONFIG_YAML,
        SAVE_FILES=SAVE_FILES_2,
        version = version_2,
        N_GLOBAL=N_GLOBAL_2,
        MIXTURE_TYPES = MIXTURE_TYPES,
        DATASETS=DATASETS
    )





