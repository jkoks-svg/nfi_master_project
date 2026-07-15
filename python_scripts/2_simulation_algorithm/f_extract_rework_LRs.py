import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import platform
import sys
import yaml

#mle_or_dnax = 'dnax' #should stay mle when looking at corrected2 results!

np.random.seed(12345)

# =========================
# CONFIG
# =========================
HYPOTHESIS = "H2"

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

BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
DONOR_DIR = BASE / "Dataset PP6FC Mixtures" / "Donoren"

#N_GLOBAL = 100


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

            continue

        # 3) Full dropout (Ø,Ø)
        full = locus_rows[locus_rows["canon_pair"] == canonical_pair("Ø", "Ø")]
        if not full.empty:
            best = full.iloc[0]
            lr_val = float(best[COL_LR])
            log10_val = np.log10(lr_val)
            log10_lr_values.append(log10_val)
            n_full_drop += 1

            continue

        # 4) No match
        missing_markers.append(locus)


    if missing_markers:
        print(f"[WARNING] No matching genotype rows for markers: {missing_markers}")

    log10_total = float(np.nansum(log10_lr_values))
    LR_total = 10 ** log10_total
    #locus_log10_list = [v for v in log10_lr_values if np.isfinite(v)]
    return log10_total, LR_total, n_fallback, n_full_drop#, locus_details, #locus_log10_list


def load_donor(mixture_folder, donor_id: str, mixture, sample) -> pd.DataFrame:
    key = (donor_id, mixture, sample)

    if key not in _donor_cache:
            
        idx = int(donor_id[1:])  # U1 -> 1, U2 -> 2, etc.
        path = mixture_folder / rf"{mixture}\sampled_joint_genotypes\{sample}\Unknown{idx}_genotype.csv"

        df = pd.read_csv(path)
        df["Allele1"] = df["Allele1"].apply(canonical_allele)
        df["Allele2"] = df["Allele2"].apply(canonical_allele)

        _donor_cache[key] = df

    return _donor_cache[key]


# =========================
# PROCESS ONE MIXTURE
# =========================
def process_mixture(mixture_folder, mixture_type, dataset_number, replicate_number, sample, N_CONTRIBUTORS,mle_or_dnax):

    mixture_original = f'{replicate_number}_{dataset_number}{mixture_type}{N_CONTRIBUTORS}'
    OUT_ROOT = mixture_folder / rf"{mixture_original}\sampled_joint_genotypes\{sample}"
    mixture = f'combined_{mle_or_dnax}_{mixture_original}_{sample}' #f"combined_{dataset_number}{mixture_type}2_sample_1"
    output_folder = OUT_ROOT / "mixtures" / mixture
    print(output_folder)
    if not output_folder.is_dir():
        print(f"⚠ Skipping mixture {mixture} — folder not found.")
        return None

    print("\n" + "=" * 60)
    print(f"PROCESSING MIX {mixture}: dataset {dataset_number}")
    print("=" * 60)
    
    #true_donors = ['U1', 'U2']   # ← CHANGE HERE when scaling
    true_donors = [f'U{i}' for i in range(1, N_CONTRIBUTORS + 1)]
    donor_dfs = {
        donor_id: load_donor(mixture_folder, donor_id, mixture_original, sample)
        for donor_id in true_donors
    }
            
    # results_df = pd.read_csv(results_path, sep="\t")
    results_path1 = output_folder / "results_marginal_prior_LR.csv"
    results_path2 = output_folder / "results_marginal_deconvolutions_with_prior_and_LR.csv"
    try:
        if mle_or_dnax == 'dnax':
            results_df = pd.read_csv(results_path1, sep="\t")
        else:
            results_df = pd.read_csv(results_path1)
    except:
        results_df = pd.read_csv(results_path2, sep="\t")
    
    results_df[COL_A1] = results_df[COL_A1].apply(canonical_allele)
    results_df[COL_A2] = results_df[COL_A2].apply(canonical_allele)

    mapping = [
        (f"Unknown{i+1}", donor_id, donor_dfs[donor_id])
        for i, donor_id in enumerate(true_donors)
    ]
    
    
    infos = []
    percentile_rows = []
    
    for contr, donor_id, donor_df in mapping:
        log10LR, LR, n_fb, n_fd = compute_true_donor_LR(
            results_df,
            hypothesis=HYPOTHESIS,
            contributor=contr,
            donor_df=donor_df
        )
    
        infos.append({
            "contributor": contr,
            "donor_id": donor_id,
            "log10_LR": log10LR,
            "LR": LR,
            "n_fallback": n_fb,
            "n_full_drop": n_fd,
        })
    
    # Process each contributor
    for info in infos: #best["infos"]:
        contr = info["contributor"]
        donor_id = info["donor_id"]

        true_log10_on = info["log10_LR"]
        true_LR_on = info["LR"]
        n_fallback = info["n_fallback"]
        n_full_drop = info["n_full_drop"]

        # role label
        #true_donor_label = "true_donor1" if donor_id == 'U1' else "true_donor2"
        true_donor_label = f"true_donor{true_donors.index(donor_id) + 1}"

        print(f"\nTrue donor LR ({contr} -> {donor_id} / {true_donor_label}): log10 LR = {true_log10_on:.3f}, LR = {true_LR_on:.3e}")
        print(f"    Fallback loci (a,Ø): {n_fallback}, full-drop loci (Ø,Ø): {n_full_drop}")


        percentile_rows.append({
            "mixture": mixture,
            "mixture_original":mixture_original,
            "dataset": dataset_number,
            "replicate": replicate_number,
            "mixture_type": mixture_type,
            "contributor": contr,
            "true_donor": true_donor_label,
            "donor_id": donor_id,
            "method": "ON",
            "true_log10_LR": true_log10_on,

        })

    return {
        "mixture": mixture,
        "percentile_rows": percentile_rows,
    }


# =========================
# MAIN
# =========================
def main1(mixture_folder,SAVE_FILES, version, CONFIG_YAML,mle_or_dnax):
    #start = time.time()
    
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    N_CONTRIBUTORS = int(cfg['dnastatistx']['N_CONTRIBUTORS'])

    # --- record run params ---
    run_params = {
        # plotting / outputs
        "SAVE_FILES": SAVE_FILES,
        "version": version,
        "HYPOTHESIS": HYPOTHESIS,
    }

    all_percentile_rows = []


    mixture_list = [item.name for item in mixture_folder.iterdir() if item.is_dir()]
    
    for mixture in mixture_list:
       
        RESULTS_DIR = Path(mixture_folder / rf"{mixture}\results_samples_{mle_or_dnax}_{version}")

        r, rest = mixture.split('_')
        replicate_number, dataset_number, mixture_type = int(r), int(rest[:-2]), rest[-2]        
    
        INPUT_ROOT = Path(mixture_folder / rf"{mixture}\sampled_joint_genotypes" )
        sample_list = [p.name for p in INPUT_ROOT.iterdir() if p.is_dir()]
        for sample in sample_list:
            if SAVE_FILES:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            if SAVE_FILES:
                write_run_parameters_txt(RESULTS_DIR, run_params, filename=f"run_params_read_csv_{version}.txt")
                write_run_parameters_json(RESULTS_DIR, run_params, filename=f"run_params_read_csv_{version}.json")
            
            
            #for replicate_number in REPLICATES:
            result = process_mixture(mixture_folder,mixture_type, dataset_number, replicate_number, sample, N_CONTRIBUTORS,mle_or_dnax)#, replicate_number)
            if result is None:
                continue

            all_percentile_rows.extend(result["percentile_rows"])


    percentile_df = pd.DataFrame(all_percentile_rows)

    
    return percentile_df #, fallback_df


                
                
def save_filtered_dfs(percentile_df, results_base_dir, version, save_files, mle_or_dnax):
    """
    Save filtered DataFrames per mixture and contributor.
    """

    results_base_dir = Path(results_base_dir)

    # Loop over mixtures and contributors directly from dataframe
    grouped = percentile_df.groupby(["mixture_original", "contributor"])

    for (mixture, contr), group in grouped:

        filtered_df = group[
            group["method"] == "ON"
        ].copy()

        if filtered_df.empty:
            print(f"[SKIP] No data for mixture {mixture}, contributor {contr}")
            continue

        if save_files:
            results_dir = (
                results_base_dir
                / "mixtures"
                / mixture
                / f"results_samples_{mle_or_dnax}_{version}"
            )
            results_dir.mkdir(parents=True, exist_ok=True)

            filtered_df.to_csv(
                results_dir / f"filtered_df_{contr}_{version}.csv",
                index=False
            )

        print(f"[OK] Saved {mixture} - {contr}")
                

def create_filtered_dfs(mixture_folder, results_base_dir, version, save_files, CONFIG_YAML, mle_or_dnax):

    percentile_df = main1(
        mixture_folder,
        save_files,
        version,
        CONFIG_YAML,
        mle_or_dnax
    )

    save_filtered_dfs(
        percentile_df,
        results_base_dir,
        version,
        save_files,
        mle_or_dnax,
    )
        
def main():
    SAVE_FILES = True
    version = 'v1'
    mixture_folder = Path(
        r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_runF_debug21\mixtures"
    )

    results_base_dir = Path(
        r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_runF_debug21"
    )
    RESOURCES_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources")
    CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    
    create_filtered_dfs(mixture_folder, results_base_dir, version, SAVE_FILES, CONFIG_YAML)
    

    
if __name__ == "__main__":
    main()
    
    
    
