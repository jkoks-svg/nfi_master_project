# -*- coding: utf-8 -*-
"""
Merged pipeline script:
  1) Run DNAStatistX in batch over MIXTURES_DIR into OUTPUT_ROOT/mixtures/<mix_id>/
  2) Post-process each mixture folder: add genotype priors (raw + fixed) + LR columns
     into results_marginal_deconvolutions_with_prior_and_LR.csv

Run-level logging:
  - Writes ONE run-level .txt and ONE run-level .json in OUTPUT_ROOT that capture:
      * all paths / switches
      * full DNAStatistX command spec (single source of truth)
      * post-processing settings (EPS, RARE_FREQ, freq file path, etc.)
      * integrity hashes of key resource files

No per-mixture parameter files are created.

Requires:
  resources/config.yaml
  resources/reference_file.txt
  resources/<jar_file>
  resources/<frequencies_file>              (for DNAStatistX itself; set in config.yaml)
  resources/NFI_frequencies.csv             (for genotype prior post-processing)
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
from confidence import Configuration


# ----------------------------------------------------------------------
# EDIT THESE PATHS / SWITCHES
# ----------------------------------------------------------------------

MIXTURES_DIR = Path(
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\mixtures_2p"
    #r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p\mixtures"
)

# trace_files = [
#     MIXTURES_DIR / "1_1A2.txt",
#     MIXTURES_DIR / "2_1A2.txt",
#     MIXTURES_DIR / "3_1A2.txt",
# ]

DATASETS = [1, 2, 3, 4, 5, 6]
MIXTURE_TYPES = ["A", "B", "C", "D", "E"]
REPLICATES = [1, 2, 3]
N_CONTRIB = 2

def build_trace_files(dataset: int, mixture: str) -> list[Path]:
    files = []
    for rep in REPLICATES:
        fname = f"{rep}_{dataset}{mixture}{N_CONTRIB}.txt"
        fpath = MIXTURES_DIR / fname
        # if not fpath.exists():
        #     raise FileNotFoundError(f"Missing replicate: {fpath}")
        if not fpath.exists():
            return None
        files.append(fpath)
    return files


OUTPUT_ROOT = Path(
    #r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p5p\output_test_1B_new"
    #r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_combined_Bayes_test"
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_DNASx_rework_frac_thresh_on_degr_on"
)

# Where DNAStatistX outputs will be placed:
#   OUTPUT_ROOT/mixtures/<mix_id>/results.json
#   OUTPUT_ROOT/mixtures/<mix_id>/results_marginal_deconvolutions.csv  (expected)
OUTPUT_MIXTURES_DIR = OUTPUT_ROOT / "mixtures"

SKIP_FIRST_N = 0
MAX_MIXTURES = None

# Post-processing frequency file (for genotype priors)
FREQ_CSV_FOR_PRIORS = Path(
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources\NFI_frequencies.csv"
)

# ---- rare-allele model constants for priors ----
RARE_FREQ = 3*10**(-4) #1e-6
EPS = RARE_FREQ**2 #1e-12
H2_FILTER_THRESHOLD = 10**(-9)


# ----------------------------------------------------------------------
# Derived paths
# ----------------------------------------------------------------------
COND_KNOWNS =  "ABCD1234NL#01"
SCRIPT_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
RESOURCES_PATH = SCRIPT_PATH / "resources"
REFERENCE_FILE = RESOURCES_PATH / "2G_reference_format_ordered.txt" #"reference_file.txt"
CONFIG_YAML = RESOURCES_PATH / "config_2p_Bayes_test.yaml"


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    return str(o)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_params_txt(outdir: Path, params: dict, filename: str):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / filename

    def fmt(v):
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(map(str, v)) + "]"
        if isinstance(v, dict):
            lines = []
            for k in sorted(v.keys()):
                lines.append(f"  {k}: {v[k]}")
            return "\n" + "\n".join(lines)
        return str(v)

    meta = {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
    }

    lines = []
    lines.append("RUN METADATA")
    lines.append("-" * 80)
    for k in sorted(meta.keys()):
        lines.append(f"{k}: {meta[k]}")

    lines.append("\nRUN-LEVEL PARAMETERS (pipeline)")
    lines.append("-" * 80)
    for k in sorted(params.keys()):
        lines.append(f"{k}: {fmt(params[k])}")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Saved run-level params txt -> {p}")
    return p


def write_params_json(outdir: Path, params: dict, filename: str):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / filename

    payload = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "script": Path(__file__).name if "__file__" in globals() else "<interactive>",
        "run_params": params,
    }
    p.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"[INFO] Saved run-level params json -> {p}")
    return p


# ----------------------------------------------------------------------
# DNAStatistX config + robust command spec
# ----------------------------------------------------------------------

def load_dnax_config():
    if not CONFIG_YAML.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_YAML}")
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "dnastatistx" not in cfg:
        raise KeyError(f"'dnastatistx' missing in {CONFIG_YAML}")
    return cfg["dnastatistx"]


def build_command_spec() -> dict:
    """
    Single source of truth for *all* command parameters.
    """
    cfg = load_dnax_config()
    thresholds_cfg = Configuration(cfg["thresholds"])

    jar_file = RESOURCES_PATH / cfg["jar_file"]
    freqs_file = RESOURCES_PATH / cfg["frequencies_file"]

    model = str(cfg["model"])
    kit = str(cfg["kit"])
    threads = int(cfg["threads"])
    method = str(cfg["method"])

    method_flag = {
        "mle": "--method-maximize",
        "maximize": "--method-maximize",
        "integrate": "--method-integrate",
    }.get(method.lower(), "--method-maximize")

    thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    if "DEFAULT" not in thresholds_map:
        thresholds_map["DEFAULT"] = thresholds_cfg.DEFAULT

    return {
        "java_executable": "java",
        "jar_file": str(jar_file),
        "action": "calculate",
        "reference_profile": str(REFERENCE_FILE),

        "method": method,
        "method_flag": method_flag,

        "model": model,
        "model_flag": f"--{model}",

        "H1": {"contributors": 2, "cond_knowns": COND_KNOWNS},
        "H2": {"contributors": 2},

        "kit": kit,
        "frequencies_file": str(freqs_file),
        "threads": threads,
        "coancestry": "0.0",

        "thresholds": thresholds_map,
        "frac_threshold": "*=0.03",
        "dropin_prob" : "1.0E-6",
        "dropin_lambda" : "1.0E-6",
        "random_seed" : 9103477288049225897,
        "deconvolution": True,
        "degradation": (model == "model-euroformix"),
    }


#def render_cmd_from_spec(spec: dict, mixture_file: Path, output_base: Path) -> list[str]:
def render_cmd_from_spec(
    spec: dict,
    trace_files: list[Path],
    output_base: Path
) -> list[str]:    

    thresholds = spec["thresholds"]
    default_thr = thresholds.get("DEFAULT")
    thresholds_args = [f"{k}={v}" for k, v in thresholds.items() if k != "DEFAULT"]

    cmd = [
        spec["java_executable"], "-jar", spec["jar_file"],
        spec["action"],
        #"--trace-profile", str(mixture_file),
        "--trace-profile", *map(str, trace_files),
        "--reference-profile", spec["reference_profile"],
        "--output", str(output_base),

        spec["method_flag"],
        spec["model_flag"],

        "-H1", "--contributors", str(spec["H1"]["contributors"]), "--cond-knowns", spec["H1"]["cond_knowns"],
        "-H2", "--contributors", str(spec["H2"]["contributors"]),

        "--kit", spec["kit"],
        "-P", spec["frequencies_file"],
        "--threads", str(spec["threads"]),
        "--coancestry", str(spec["coancestry"]),

        "--threshold",
        *thresholds_args,
        f"*={default_thr}",
        "--frac-threshold", spec["frac_threshold"],
        "--dropin-prob", spec['dropin_prob'],
        "--dropin-lambda", spec['dropin_lambda'],
        
        "--random-seed" , str(spec["random_seed"])
    ]

    if spec.get("deconvolution", False):
        cmd.append("--deconvolution")
    if spec.get("degradation", False):
        cmd.append("--degradation")

    return cmd


#def run_one_mixture_dnax(mixture_file: Path, output_dir: Path, command_spec: dict) -> bool:
def run_one_mixture_dnax(
    mixture_file: Path,
    output_dir: Path,
    command_spec: dict,
    trace_files: list[Path] | None = None
) -> bool:
    
    if trace_files is None:
        trace_files = [mixture_file]
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "results"

    # Clean previous DNAStatistX outputs
    for ext in (".json", ".txt"):
        f = output_base.with_suffix(ext)
        if f.exists():
            f.unlink()

    #cmd = render_cmd_from_spec(command_spec, mixture_file, output_base)
    cmd = render_cmd_from_spec(command_spec, trace_files, output_base)


    print("\n=== DNAStatistX Running ===")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # keep stdout/stderr for debugging
    (output_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (output_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        print("ERROR: DNAStatistX returned non-zero status")
        print("STDERR:\n", proc.stderr)
        return False

    json_path = output_base.with_suffix(".json")
    if not json_path.exists():
        print("ERROR: No results.json produced")
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
        return False

    print(f"✓ DNAStatistX completed {mixture_file.name}, output → {output_dir}")
    return True


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


def build_rare_per_locus(freq_map):
    rare_per_locus = {}
    for locus, locus_map in freq_map.items():
        positives = [f for f in locus_map.values() if f > 0]
        if positives:
            rare_per_locus[locus] = min(min(positives), RARE_FREQ)
        else:
            rare_per_locus[locus] = RARE_FREQ
    return rare_per_locus


def safe_allele_freq(locus_map, locus, allele_label, rare_per_locus):
    f = locus_map.get(allele_label)
    if f is None or f <= 0 or not np.isfinite(f):
        f = rare_per_locus.get(locus, RARE_FREQ)
    return max(float(f), EPS)


def genotype_prior_fixed(freq_map, observed_alleles, rare_per_locus, locus: str, a1, a2) -> float:
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
        p_obs = safe_allele_freq(locus_map, loc, la2, rare_per_locus)
        return max(2.0 * p_obs * pU, EPS)

    if is_null2 and not is_null1:
        p_obs = safe_allele_freq(locus_map, loc, la1, rare_per_locus)
        return max(2.0 * p_obs * pU, EPS)

    p1 = safe_allele_freq(locus_map, loc, la1, rare_per_locus)
    p2 = safe_allele_freq(locus_map, loc, la2, rare_per_locus)

    gp = (p1 ** 2) if (la1 == la2) else (2.0 * p1 * p2)
    return max(float(gp), EPS)

def filter_small_probabilities(
    input_csv,
    output_csv,
    prob_column="Probability",
    threshold=H2_FILTER_THRESHOLD
):
    # Read CSV (let pandas handle scientific notation)
    df = pd.read_csv(input_csv, sep = '\t')

    # Ensure Probability column is numeric
    df[prob_column] = pd.to_numeric(df[prob_column], errors="coerce")

    # Filter rows
    df_filtered = df[df[prob_column] >= threshold]

    # Write output CSV
    df_filtered.to_csv(output_csv, sep = '\t', index=False)


def postprocess_add_priors(root_mixtures_dir: Path, freq_csv: Path) -> tuple[int, int]:
    """
    For each subfolder in root_mixtures_dir:
      - reads results_marginal_deconvolutions.csv
      - writes results_marginal_deconvolutions_with_prior_and_LR.csv

    Returns: (n_ok, n_skipped_or_failed)
    """
    if not freq_csv.exists():
        raise FileNotFoundError(f"Frequency CSV not found: {freq_csv}")

    freq_df = pd.read_csv(freq_csv, encoding="utf-8")
    freq_df.rename(columns={"PentaD": "Penta D", "PentaE": "Penta E"}, inplace=True)
    freq_map = build_frequency_map(freq_df)
    rare_per_locus = build_rare_per_locus(freq_map)

    mixture_dirs = [d for d in glob.glob(os.path.join(str(root_mixtures_dir), "*")) if os.path.isdir(d)]
    print("\n=== Post-processing priors ===")
    print(f"Found {len(mixture_dirs)} mixture folders under {root_mixtures_dir}")

    n_ok = 0
    n_bad = 0

    for mixdir in mixture_dirs:
        in_file = os.path.join(mixdir, "results_marginal_deconvolutions.csv")
        out_file = os.path.join(mixdir, "results_marginal_deconvolutions_with_prior_and_LR.csv")
        
        # in_file_joint_H2 = 
        # out_file_joint_H2 = 
        filter_small_probabilities(
            input_csv= os.path.join(mixdir, "results_joint_deconvolution_H2.csv"),
            output_csv=os.path.join(mixdir, "results_joint_deconvolution_H2_clean.csv")
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
                gp_fixed = genotype_prior_fixed(freq_map, observed, rare_per_locus, locus, a1, a2)
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
            n_ok += 1
        except Exception as e:
            print(f"  ❌ Failed for {mixdir}: {e}")
            n_bad += 1

    return n_ok, n_bad


# ----------------------------------------------------------------------
# Run-level parameter bundle (both steps)
# ----------------------------------------------------------------------

def build_pipeline_runlevel_params(command_spec: dict, example_cmd: list[str] | None) -> dict:
    jar_path = Path(command_spec["jar_file"])
    freq_path = Path(command_spec["frequencies_file"])
    ref_path = Path(command_spec["reference_profile"])

    params = {
        # pipeline paths / switches
        "MIXTURES_DIR": str(MIXTURES_DIR),
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "OUTPUT_MIXTURES_DIR": str(OUTPUT_MIXTURES_DIR),
        "SKIP_FIRST_N": SKIP_FIRST_N,
        "MAX_MIXTURES": MAX_MIXTURES,
        "COND_KNOWNS": COND_KNOWNS,
        "H2_FILTER_THRESHOLD": H2_FILTER_THRESHOLD,

        "SCRIPT_PATH": str(SCRIPT_PATH),
        "RESOURCES_PATH": str(RESOURCES_PATH),
        "CONFIG_YAML": str(CONFIG_YAML),
        "REFERENCE_FILE": str(REFERENCE_FILE),

        # step 1: DNAStatistX command spec
        "DNASX_COMMAND_SPEC": deepcopy(command_spec),

        # step 2: genotype priors settings
        "PRIORS": {
            "FREQ_CSV_FOR_PRIORS": str(FREQ_CSV_FOR_PRIORS),
            "EPS": EPS,
            "RARE_FREQ": RARE_FREQ,
            "expected_input_filename": "results_marginal_deconvolutions.csv",
            "output_filename": "results_marginal_deconvolutions_with_prior_and_LR.csv",
        },

        # integrity hashes
        "HASHES": {
            "config_yaml_sha256": file_sha256(CONFIG_YAML),
            "reference_file_sha256": file_sha256(ref_path),
            "jar_sha256": file_sha256(jar_path),
            "dnax_frequencies_sha256": file_sha256(freq_path),
            "priors_frequency_csv_sha256": file_sha256(FREQ_CSV_FOR_PRIORS),
        },
    }

    if example_cmd is not None:
        params["EXAMPLE_DNASX_CMD_LIST"] = example_cmd
        params["EXAMPLE_DNASX_CMD_STRING"] = " ".join(example_cmd)

    return params


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # mixtures = sorted(MIXTURES_DIR.glob("*.txt"))
    # if not mixtures:
    #     raise SystemExit(f"No .txt mixtures found in {MIXTURES_DIR}")

    # mixtures_to_run = mixtures[SKIP_FIRST_N:]
    # if MAX_MIXTURES is not None:
    #     mixtures_to_run = mixtures_to_run[:MAX_MIXTURES]

    # print(f"Found {len(mixtures)} mixture .txt files")
    # print(f"Running {len(mixtures_to_run)} mixtures (skipped first {SKIP_FIRST_N})")
    
    # # example cmd for logging
    # example_cmd = None
    # if mixtures_to_run:
    #     example_mix = mixtures_to_run[0]
    #     example_out = OUTPUT_MIXTURES_DIR / example_mix.stem / "results"
    #     example_cmd = render_cmd_from_spec(command_spec, example_mix, example_out)

    # # ---- run-level logging once (covers BOTH steps) ----
    # runlevel_params = build_pipeline_runlevel_params(command_spec, example_cmd)
    # write_params_txt(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.txt")
    # write_params_json(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.json")

    # # ---- step 1: run DNAStatistX ----
    # n_ok = 0
    # n_fail = 0
    # print(mixtures_to_run)
    # print(len(mixtures_to_run))
    # print(type(mixtures_to_run))
    # for mix_file in mixtures_to_run[0:1]:
    #     mix_id = mix_file.stem
    #     mix_output_dir = OUTPUT_MIXTURES_DIR / "trace_1A2" #mix_id
    #     ok = run_one_mixture_dnax(mix_file, mix_output_dir, command_spec)
    #     if ok:
    #         n_ok += 1
    #     else:
    #         n_fail += 1

    # ---- build spec once (single source of truth) ----
    command_spec = build_command_spec()
    
    # ---- example command for logging ----
    example_cmd = None
    try:
        example_trace = build_trace_files(DATASETS[0], MIXTURE_TYPES[0])
        example_out = OUTPUT_MIXTURES_DIR / f"{DATASETS[0]}{MIXTURE_TYPES[0]}{N_CONTRIB}" / "results"
        example_cmd = render_cmd_from_spec(command_spec, example_trace, example_out)
    except Exception:
        pass
    
    # ---- run-level logging once ----
    runlevel_params = build_pipeline_runlevel_params(command_spec, example_cmd)
    write_params_txt(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.txt")
    write_params_json(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.json")

    n_ok = 0
    n_fail = 0
    
    for dataset in DATASETS:
        for mixture in MIXTURE_TYPES:
            trace_id = f"{dataset}{mixture}{N_CONTRIB}"
            print(f"\n=== Running TRACE {trace_id} ===")
    
            # trace_files = build_trace_files(dataset, mixture)
            # output_dir = OUTPUT_MIXTURES_DIR / trace_id
            
            trace_files = build_trace_files(dataset, mixture)
            if trace_files is None:
                print(f"[SKIP] Incomplete trace {trace_id}")
                continue
            
            output_dir = OUTPUT_MIXTURES_DIR / trace_id

    
            # ok = run_one_mixture_dnax(
            #     mixture_file=trace_files[0],  # dummy, kept for structure
            #     output_dir=output_dir,
            #     command_spec=command_spec,
            # )
            ok = run_one_mixture_dnax(
                mixture_file=trace_files[0],
                output_dir=output_dir,
                command_spec=command_spec,
                trace_files=trace_files
            )
                
            if ok:
                n_ok += 1
            else:
                n_fail += 1


    print("\n=== DNAStatistX batch completed ===")
    print(f"Success: {n_ok}, Failed: {n_fail}")

    # ---- step 2: add genotype priors + LR columns ----
    pp_ok, pp_bad = postprocess_add_priors(OUTPUT_MIXTURES_DIR, FREQ_CSV_FOR_PRIORS)

    print("\n=== Priors post-processing completed ===")
    print(f"Success: {pp_ok}, Skipped/Failed: {pp_bad}")

    print("\nPipeline done.")


if __name__ == "__main__":
    main()

