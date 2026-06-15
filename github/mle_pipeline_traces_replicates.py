# -*- coding: utf-8 -*-
"""
Plug-in MLE replacement for mcmc_pipeline_traces_replicates.py

Goal
----
Keep the same input traversal and output folders as the simulated-rework
replicate pipeline, but do NOT run MCMC. Instead, read the plug-in MLE
parameters from a DNAStatistX results.json file and use that single parameter
vector to create, per combined replicate analysis:

    results_joint_deconvolution_H2_clean.csv
    results_marginal_prior_LR.csv
    results_mle.json

Expected input layout, same as your simulated rework pipeline:

    mixture_folder/
        <mixture>/
            sampled_joint_genotypes/
                sample_0001/
                    *Trace*.txt
                    mixtures/
                        combined_mle_<mixture>_sample_0001/       # written by this script

For each sample folder, the three traces are:
    1. original trace: MIXTURES_DIR / f"{mixture}.txt"
    2. first simulated *Trace*.txt in sample folder
    3. second simulated *Trace*.txt in sample folder

The plug-in MLE parameters are read from:

    mle_results_folder/
        <mixture>/
            sampled_joint_genotypes/
                <sample>/
                    mixtures/
                        combined_dnax_<mixture>_<sample>/
                            results.json

By default, mle_results_folder is candidate_mixture_folder if provided,
otherwise mixture_folder.

This script assumes that create_deconv_tables_both.py provides the same helper
functions as in your current code, especially:
    - build_params_from_results_json
    - build_common_inputs_for_mcmc
    - fill_missing_alleles_in_replicate_dfs
    - extract_observed_alleles_and_thresholds
    - create_joint_df
    - build_combined_analysis_df
    - create_marginal_df_combined
    - split_and_sort_genotype
    - load_candidate_genotype_pairs
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ----------------------------------------------------------------------
# Defaults: edit these if you always run from the same project structure
# ----------------------------------------------------------------------
DEFAULT_CREATE_DECONV_TABLES_BOTH_PY = Path(
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\python scripts\Bayes model\create_deconv_tables_both.py"
)


# ----------------------------------------------------------------------
# Import create_deconv_tables_both.py from an arbitrary file path
# ----------------------------------------------------------------------
def import_create_deconv_module(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"create_deconv_tables_both.py not found: {path}")

    # Important: make sibling imports work, e.g. gamma_model_degr_on.py,
    # genotypes.py, mcmc_parameters_degr_on.py.
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("create_deconv_tables_both", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["create_deconv_tables_both"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------
def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
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

    lines.append("\nRUN-LEVEL PARAMETERS (plug-in MLE pipeline)")
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


def build_mle_runlevel_params(
    *,
    create_deconv_tables_both_py: Path,
    resources_path: Path,
    config_yaml: Path,
    mixtures_dir: Path,
    reference_file: Path | None,
    cond_knowns: str | None,
    candidate_mixture_folder: Path | None = None,
    mle_results_folder: Path | None = None,
    mle_config: dict | None = None,
) -> dict:
    with Path(config_yaml).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    freqs_file = Path(resources_path) / cfg["dnastatistx"]["frequencies_file"]
    kit_properties_txt = Path(resources_path) / cfg["kit_properties_txt"]

    return {
        "PIPELINE_MODE": "plug-in MLE replacement for DNAStatistX deconvolution step",
        "CREATE_DECONV_TABLES_BOTH_PY": str(create_deconv_tables_both_py),
        "RESOURCES_PATH": str(resources_path),
        "CONFIG_YAML": str(config_yaml),
        "MIXTURES_DIR": str(mixtures_dir),
        "REFERENCE_FILE_kept_for_same_signature": str(reference_file) if reference_file else None,
        "COND_KNOWNS_kept_for_same_signature": cond_knowns,
        "MLE_CONFIG": deepcopy(mle_config or {}),
        "CANDIDATE_MIXTURE_FOLDER_DNASTATISTX": str(candidate_mixture_folder) if candidate_mixture_folder else None,
        "MLE_RESULTS_FOLDER_DNASTATISTX": str(mle_results_folder) if mle_results_folder else None,
        "MLE_RESULTS_JSON_PATTERN": "<mle_results_folder>/<mixture>/sampled_joint_genotypes/<sample>/mixtures/combined_dnax_<mixture>_<sample>/results.json",
        "CANDIDATE_CSV_PATTERN": "<candidate_mixture_folder>/<mixture>/sampled_joint_genotypes/<sample>/mixtures/combined_dnax_<mixture>_<sample>/results_joint_deconvolution_H2_clean.csv",
        "OUTPUT_FILES_PER_COMBINED_TRACE": [
            "results_joint_deconvolution_H2_clean.csv",
            "results_marginal_prior_LR.csv",
            "results_mle.json",
            "results_dnax.json",
        ],
        "HASHES": {
            "create_deconv_tables_both_sha256": file_sha256(create_deconv_tables_both_py),
            "config_yaml_sha256": file_sha256(config_yaml),
            "frequencies_sha256": file_sha256(freqs_file),
            "kit_properties_sha256": file_sha256(kit_properties_txt),
            "reference_file_sha256": file_sha256(reference_file) if reference_file else None,
        },
    }


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------
def dnax_trace_id(mixture: str, sample: str) -> str:
    return f"combined_dnax_{mixture}_{sample}"


def mle_trace_id(mixture: str, sample: str) -> str:
    return f"combined_mle_{mixture}_{sample}"


def dnax_combined_dir(root_folder: Path, mixture: str, sample: str) -> Path:
    return (
        Path(root_folder)
        / mixture
        / "sampled_joint_genotypes"
        / sample
        / "mixtures"
        / dnax_trace_id(mixture, sample)
    )


def candidate_csv_path_for_sample(candidate_mixture_folder: Path, mixture: str, sample: str) -> Path:
    return dnax_combined_dir(candidate_mixture_folder, mixture, sample) / "results_joint_deconvolution_H2_clean.csv"


#FIX THIS FOR AAC combinations
def dnax_combined_dir2(root_folder: Path, mixture: str, sample: str) -> Path:
    return (
        Path(root_folder)
        / mixture
        # / "sampled_joint_genotypes"
        # / sample
    )

def mle_results_json_path_for_sample(mle_results_folder: Path, mixture: str, sample: str) -> Path:
    return dnax_combined_dir2(mle_results_folder, mixture, sample) / "results_dnax.json"
#"sampled_theta_parameters.json" 
#"sampled_generation_parameters.json"


# ----------------------------------------------------------------------
# Candidate filtering and MLE parameter loading
# ----------------------------------------------------------------------
def default_mle_config() -> dict:
    return {
        "require_candidate_csv": True,
        "candidate_min_probability": 0.0,
        "candidate_max_rows_per_locus": None,
        "candidate_add_contributor_swaps": False,
        "copy_dnax_results_json": True,
    }


def load_candidate_pairs_from_dnax_csv(
    *,
    deconv,
    candidate_csv: Path,
    mle_config: dict,
):
    """Load candidate genotype pairs from an existing DNAStatistX output CSV."""
    candidate_csv = Path(candidate_csv)

    if not candidate_csv.exists():
        msg = f"Candidate CSV from DNAStatistX pipeline not found: {candidate_csv}"
        if mle_config.get("require_candidate_csv", True):
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}; continuing without candidate-pair filtering.")
        return None

    print(f"[INFO] Loading candidate pairs from DNAStatistX CSV: {candidate_csv}")
    candidate_pairs_by_locus = deconv.load_candidate_genotype_pairs(
        candidate_csv,
        min_probability=mle_config.get("candidate_min_probability", 0.0),
        max_rows_per_locus=mle_config.get("candidate_max_rows_per_locus"),
        add_contributor_swaps=mle_config.get("candidate_add_contributor_swaps", False),
    )
    n_loci = len(candidate_pairs_by_locus)
    n_pairs = sum(len(v) for v in candidate_pairs_by_locus.values())
    print(f"[INFO] Loaded {n_pairs} candidate genotype pairs across {n_loci} loci")
    return candidate_pairs_by_locus


def build_rework_params_from_mle_results_json(
    *,
    deconv,
    results_json_path: Path,
    config_yaml: Path,
    common: dict,
):
    """
    Read one DNAStatistX plug-in MLE parameter vector and wrap it in params_lst.

    This mirrors your other script's parameter_method == "MLE" branch:
        params_lst = [build_params_from_results_json(...)]
        mcmc_result = None
    """
    results_json_path = Path(results_json_path)
    if not results_json_path.exists():
        raise FileNotFoundError(f"MLE results.json not found: {results_json_path}")

    params = deconv.build_params_from_results_json(results_json_path, config_yaml)

    if len(params[0]) != common["N_contributors"]:
        raise ValueError(
            "N_contributors not equal to length of phi vector: "
            f"N_contributors={common['N_contributors']}, phi={params[0]}"
        )

    print("Using plug-in MLE parameters")
    print("results_json", results_json_path)
    print("phi", params[0])
    print("mu", params[1])
    print("sigma", params[2])
    print("C", params[3])
    print("lambda", params[4])
    print("beta", params[5])

    return [params], params


# ----------------------------------------------------------------------
# Deconvolution helpers
# ----------------------------------------------------------------------
def load_and_fill_trace_dfs(*, deconv, trace_files: list[Path]) -> list[pd.DataFrame]:
    """
    Read the original trace plus the two simulated trace profiles,
    then fill missing alleles across the triplicate in the same way as
    create_deconv_tables_both.py.
    """
    trace_dfs = [
        pd.read_csv(trace_file, sep="\t", dtype=str).fillna("")
        for trace_file in trace_files
    ]

    return deconv.fill_missing_alleles_in_replicate_dfs(trace_dfs)


def deconvolve_one_replicate_full_df(
    *,
    deconv,
    mixture_df: pd.DataFrame,
    replicate_number: int,
    combined_trace_id: str,
    params_lst: list,
    common: dict,
    candidate_pairs_by_locus=None,
) -> pd.DataFrame:
    """Create the full joint posterior table for one filled replicate profile."""
    mixture_df = mixture_df.copy()
    observed_alleles, thresholds = deconv.extract_observed_alleles_and_thresholds(
        mixture_df,
        common["thresholds_map"],
        common["frac_threshold"],
        mode="rework",
    )

    joint_df = deconv.create_joint_df(
        params_lst,
        observed_alleles,
        common["all_alleles"],
        thresholds,
        common["freq_dict"],
        common["RARE_FREQ"],
        common["N_contributors"],
        common["kit_properties_df"],
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )

    # These columns are required by build_combined_analysis_df().
    joint_df["mixture"] = f"{replicate_number}_{combined_trace_id}"
    joint_df["replicate"] = replicate_number
    joint_df["mixture_combined"] = combined_trace_id

    return joint_df


def combined_observed_alleles_for_trace_dfs(
    *,
    deconv,
    trace_dfs: list[pd.DataFrame],
    common: dict,
) -> dict:
    """Union observed alleles across the three filled replicates, keeping the maximum height."""
    combined_observed: dict = {}

    for mixture_df in trace_dfs:
        observed_alleles, _ = deconv.extract_observed_alleles_and_thresholds(
            mixture_df.copy(),
            common["thresholds_map"],
            common["frac_threshold"],
            mode="rework",
        )
        for locus, allele_dict in observed_alleles.items():
            combined_observed.setdefault(locus, {})
            for allele, height in allele_dict.items():
                old_height = combined_observed[locus].get(allele, 0)
                combined_observed[locus][allele] = max(old_height, height)

    return combined_observed


def export_combined_rework_csvs(
    *,
    deconv,
    replicate_joint_dfs: list[pd.DataFrame],
    trace_dfs: list[pd.DataFrame],
    output_dir: Path,
    common: dict,
):
    """
    Combine the three replicate joint tables and export the same two CSVs as DNAStatistX:
        - results_joint_deconvolution_H2_clean.csv
        - results_marginal_prior_LR.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    H2_FILTER_THRESHOLD = float(common["cfg"]["H2_FILTER_THRESHOLD"])

    df_single_mixtures_rework = pd.concat(replicate_joint_dfs, ignore_index=True)
    df_rework_posteriors = deconv.build_combined_analysis_df(df_single_mixtures_rework)

    # 1) results_joint_deconvolution_H2_clean.csv
    joint_df_clean = df_rework_posteriors.copy()
    joint_df_clean[["Unknown1-Allele1", "Unknown1-Allele2"]] = (
        joint_df_clean["Unknown1"].apply(deconv.split_and_sort_genotype)
    )
    joint_df_clean[["Unknown2-Allele1", "Unknown2-Allele2"]] = (
        joint_df_clean["Unknown2"].apply(deconv.split_and_sort_genotype)
    )

    joint_df_clean = (
        joint_df_clean.loc[
            joint_df_clean["Probability"] >= H2_FILTER_THRESHOLD,
            [
                "Locus",
                "Unknown1-Allele1",
                "Unknown1-Allele2",
                "Unknown2-Allele1",
                "Unknown2-Allele2",
                "Probability",
            ],
        ]
        .copy()
        .sort_values(by=["Locus", "Probability"], ascending=[True, False])
    )

    joint_csv = output_dir / "results_joint_deconvolution_H2_clean.csv"
    joint_df_clean.to_csv(joint_csv, index=False)
    print(f"[OK] Wrote {joint_csv}")

    # 2) results_marginal_prior_LR.csv
    observed_alleles = combined_observed_alleles_for_trace_dfs(
        deconv=deconv,
        trace_dfs=trace_dfs,
        common=common,
    )

    marginal_df = deconv.create_marginal_df_combined(
        df_rework_posteriors,
        observed_alleles,
        common["freq_dict"],
        common["RARE_FREQ"],
    )

    marginal_df.sort_values(
        by=["Locus", "Contributor", "Marginal_probability"],
        ascending=[True, True, False],
        inplace=True,
    )
    marginal_df[["Allele 1", "Allele 2"]] = marginal_df["Genotype"].apply(
        deconv.split_and_sort_genotype
    )
    marginal_df["Hypothesis"] = "H2"
    marginal_df["Probability"] = marginal_df["Marginal_probability"]

    marginal_df = marginal_df.loc[
        :,
        [
            "Hypothesis",
            "Contributor",
            "Locus",
            "Allele 1",
            "Allele 2",
            "Probability",
            "LR(a)_1",
        ],
    ].copy()

    marginal_csv = output_dir / "results_marginal_prior_LR.csv"
    marginal_df.to_csv(marginal_csv, index=False)
    print(f"[OK] Wrote {marginal_csv}")

    return joint_csv, marginal_csv


def write_mle_results_json(
    *,
    output_dir: Path,
    source_results_json: Path,
    trace_files: list[Path],
    mle_params: tuple,
    mle_config: dict,
):
    """
    Lightweight metadata file for this plug-in MLE run.
    The source DNAStatistX results.json is optionally copied separately as results_dnax.json.
    """
    phi, mu, sigma, C, lam, beta = mle_params
    payload = {
        "producer": "mle_pipeline_traces_replicates.py",
        "note": "This is not a DNAStatistX results.json. The deconvolution CSVs were produced with the plug-in MLE parameter vector read from source_results_json.",
        "source_results_json": str(source_results_json),
        "trace_files": [str(p) for p in trace_files],
        "mle_config": mle_config,
        "mle_parameters": {
            "phi": list(phi),
            "mu": float(mu),
            "sigma": float(sigma),
            "C": float(C),
            "lambda": float(lam),
            "beta": float(beta),
        },
    }
    out = Path(output_dir) / "results_mle.json"
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"[OK] Wrote {out}")
    return out


def copy_dnax_results_json(*, source_results_json: Path, output_dir: Path):
    source_results_json = Path(source_results_json)
    if not source_results_json.exists():
        raise FileNotFoundError(f"DNAStatistX results.json not found: {source_results_json}")
    out = Path(output_dir) / "results_dnax.json"
    shutil.copy2(source_results_json, out)
    print(f"[OK] Copied DNAStatistX results.json -> {out}")
    return out


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def mle_three_mixtures_combined(
    *,
    mixture_folder,
    RESOURCES_PATH,
    CONFIG_YAML,
    MIXTURES_DIR,
    REFERENCE_FILE=None,
    COND_KNOWNS=None,
    create_deconv_tables_both_py=DEFAULT_CREATE_DECONV_TABLES_BOTH_PY,
    candidate_mixture_folder=None,
    mle_results_folder=None,
    mle_config_dict=None,
    overwrite=False,
):
    """
    Drop-in replacement for mcmc_three_mixtures_combined(...), except the deconvolution
    CSVs are created with plug-in MLE parameters instead of posterior MCMC draws.

    candidate_mixture_folder points to the old DNAStatistX pipeline output used to read
    candidate genotype pairs. If candidate_mixture_folder is None, mixture_folder is used.

    mle_results_folder points to the DNAStatistX pipeline output used to read results.json.
    If mle_results_folder is None, candidate_mixture_folder is used; if that is None,
    mixture_folder is used.

    REFERENCE_FILE and COND_KNOWNS are accepted to keep the same call shape, but are
    not used by the H2 deconvolution output generation.
    """
    mixture_folder = Path(mixture_folder)
    #candidate_mixture_folder = Path(candidate_mixture_folder) if candidate_mixture_folder is not None else mixture_folder
    #mle_results_folder = Path(mle_results_folder) if mle_results_folder is not None else candidate_mixture_folder
    
    candidate_mixture_folder = (
        Path(candidate_mixture_folder)
        if candidate_mixture_folder is not None
        else None
    )
    
    mle_results_folder = (
        Path(mle_results_folder)
        if mle_results_folder is not None
        else mixture_folder
    )
    
    resources_path = Path(RESOURCES_PATH)
    config_yaml = Path(CONFIG_YAML)
    mixtures_dir = Path(MIXTURES_DIR)
    reference_file = Path(REFERENCE_FILE) if REFERENCE_FILE is not None else None
    create_deconv_tables_both_py = Path(create_deconv_tables_both_py)

    deconv = import_create_deconv_module(create_deconv_tables_both_py)

    mle_config = default_mle_config()
    if mle_config_dict:
        mle_config.update(mle_config_dict)

    common = deconv.build_common_inputs_for_mcmc(
        RESOURCES_PATH=resources_path,
        CONFIG_YAML=config_yaml,
    )
    if common["N_contributors"] != 2:
        raise NotImplementedError("Current deconvolution code is written for 2 contributors.")

    mixture_list = sorted([item.name for item in mixture_folder.iterdir() if item.is_dir()])

    for mixture in mixture_list:
        input_root = mixture_folder / mixture / "sampled_joint_genotypes"
        if not input_root.exists():
            print(f"[SKIP] No sampled_joint_genotypes folder: {input_root}")
            continue

        sample_list = sorted([p.name for p in input_root.iterdir() if p.is_dir()])

        for sample in sample_list:
            output_root = input_root / sample
            output_mixtures_dir = output_root / "mixtures"
            trace_id = mle_trace_id(mixture, sample)
            output_dir = output_mixtures_dir / trace_id

            results_marginal_csv = output_dir / "results_marginal_prior_LR.csv"
            print(results_marginal_csv)
            if results_marginal_csv.exists() and not overwrite:
                print("Skipping:", output_root)
                continue

            output_root.mkdir(parents=True, exist_ok=True)
            output_mixtures_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            runlevel_params = build_mle_runlevel_params(
                create_deconv_tables_both_py=create_deconv_tables_both_py,
                resources_path=resources_path,
                config_yaml=config_yaml,
                mixtures_dir=mixtures_dir,
                reference_file=reference_file,
                cond_knowns=COND_KNOWNS,
                candidate_mixture_folder=candidate_mixture_folder,
                mle_results_folder=mle_results_folder,
                mle_config=mle_config,
            )
            write_params_txt(output_root, runlevel_params, "run_parameters_PIPELINE.txt")
            write_params_json(output_root, runlevel_params, "run_parameters_PIPELINE.json")

            print(f"\n=== Running plug-in MLE TRACE {trace_id} ===")

            sim_paths = sorted(output_root.glob("*Trace*.txt"))
            if len(sim_paths) < 2:
                print("[SKIP] Not enough simulated traces")
                continue

            path1_org = mixtures_dir / f"{mixture}.txt"
            path2_sim = sim_paths[0]
            path3_sim = sim_paths[1]
            trace_files = [path1_org, path2_sim, path3_sim]

            missing = [p for p in trace_files if not p.exists()]
            if missing:
                print("[SKIP] Missing trace file(s):")
                for p in missing:
                    print("   ", p)
                continue

            trace_dfs = load_and_fill_trace_dfs(
                deconv=deconv,
                trace_files=trace_files,
            )

            # candidate_csv = candidate_csv_path_for_sample(
            #     candidate_mixture_folder=candidate_mixture_folder,
            #     mixture=mixture,
            #     sample=sample,
            # )
            # candidate_pairs_by_locus = load_candidate_pairs_from_dnax_csv(
            #     deconv=deconv,
            #     candidate_csv=candidate_csv,
            #     mle_config=mle_config,
            # )
            candidate_pairs_by_locus = None
            
            if mle_config.get("use_candidate_pairs", False):
                if candidate_mixture_folder is None:
                    raise ValueError(
                        "use_candidate_pairs=True requires candidate_mixture_folder."
                    )
            
                candidate_csv = candidate_csv_path_for_sample(
                    candidate_mixture_folder=candidate_mixture_folder,
                    mixture=mixture,
                    sample=sample,
                )
            
                candidate_pairs_by_locus = load_candidate_pairs_from_dnax_csv(
                    deconv=deconv,
                    candidate_csv=candidate_csv,
                    mle_config=mle_config,
                )
            else:
                print("[INFO] Candidate-pair filtering disabled.")
                
            source_results_json = mle_results_json_path_for_sample(
                mle_results_folder=mle_results_folder,
                mixture=mixture,
                sample=sample,
            )
            params_lst, mle_params = build_rework_params_from_mle_results_json(
                deconv=deconv,
                results_json_path=source_results_json,
                config_yaml=config_yaml,
                common=common,
            )

            replicate_joint_dfs = []
            for replicate_number, (trace_file, mixture_df) in enumerate(
                zip(trace_files, trace_dfs),
                start=1,
            ):
                print(f"Creating plug-in MLE deconvolution for replicate {replicate_number}: {trace_file}")
                replicate_joint_dfs.append(
                    deconvolve_one_replicate_full_df(
                        deconv=deconv,
                        mixture_df=mixture_df,
                        replicate_number=replicate_number,
                        combined_trace_id=trace_id,
                        params_lst=params_lst,
                        common=common,
                        candidate_pairs_by_locus=candidate_pairs_by_locus,
                    )
                )

            export_combined_rework_csvs(
                deconv=deconv,
                replicate_joint_dfs=replicate_joint_dfs,
                trace_dfs=trace_dfs,
                output_dir=output_dir,
                common=common,
            )

            write_mle_results_json(
                output_dir=output_dir,
                source_results_json=source_results_json,
                trace_files=trace_files,
                mle_params=mle_params,
                mle_config=mle_config,
            )

            if mle_config.get("copy_dnax_results_json", True):
                copy_dnax_results_json(
                    source_results_json=source_results_json,
                    output_dir=output_dir,
                )

            print("\nPipeline done for", trace_id)


# Backwards-compatible-ish aliases.
dnax_three_mixtures_combined = mle_three_mixtures_combined
plugin_mle_three_mixtures_combined = mle_three_mixtures_combined


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the simulated-rework folder traversal, but create deconvolution CSVs with plug-in MLE parameters instead of MCMC."
    )
    parser.add_argument("--mixture-folder", required=True, help="The .../output.../mixtures folder containing mixture subfolders.")
    parser.add_argument("--resources-path", required=True, help="Path to resources folder.")
    parser.add_argument("--config-yaml", required=True, help="Path to config.yaml.")
    parser.add_argument("--mixtures-dir", required=True, help="Folder containing original mixture txt files, e.g. mixtures_debug_2p.")
    parser.add_argument("--reference-file", default=None, help="Kept for same signature/logging; not used for H2 MLE deconvolution CSVs.")
    parser.add_argument("--cond-knowns", default=None, help="Kept for same signature/logging; not used for H2 MLE deconvolution CSVs.")
    parser.add_argument("--create-deconv-tables-both-py", default=str(DEFAULT_CREATE_DECONV_TABLES_BOTH_PY))
    parser.add_argument("--candidate-mixture-folder", default=None, help="Old DNAStatistX .../mixtures folder used to read candidate CSVs. Defaults to --mixture-folder.")
    parser.add_argument("--mle-results-folder", default=None, help="Old DNAStatistX .../mixtures folder used to read results.json. Defaults to --candidate-mixture-folder, then --mixture-folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results_marginal_prior_LR.csv outputs.")

    parser.add_argument("--candidate-min-probability", type=float, default=None)
    parser.add_argument("--candidate-max-rows-per-locus", type=int, default=None)
    parser.add_argument("--allow-missing-candidate-csv", action="store_true", help="Continue without candidate-pair filtering if the DNAStatistX candidate CSV is missing.")
    parser.add_argument("--candidate-add-contributor-swaps", action="store_true", help="Also include swapped contributor order for candidate genotype pairs.")
    parser.add_argument("--no-copy-dnax-results-json", action="store_true", help="Do not copy source results.json to output as results_dnax.json.")
    return parser.parse_args()


def main():
    args = parse_args()

    mle_config = {}
    cli_to_key = {
        "candidate_min_probability": "candidate_min_probability",
        "candidate_max_rows_per_locus": "candidate_max_rows_per_locus",
    }
    for attr, key in cli_to_key.items():
        value = getattr(args, attr)
        if value is not None:
            mle_config[key] = value

    if args.allow_missing_candidate_csv:
        mle_config["require_candidate_csv"] = False
    if args.candidate_add_contributor_swaps:
        mle_config["candidate_add_contributor_swaps"] = True
    if args.no_copy_dnax_results_json:
        mle_config["copy_dnax_results_json"] = False

    mle_three_mixtures_combined(
        mixture_folder=Path(args.mixture_folder),
        RESOURCES_PATH=Path(args.resources_path),
        CONFIG_YAML=Path(args.config_yaml),
        MIXTURES_DIR=Path(args.mixtures_dir),
        REFERENCE_FILE=Path(args.reference_file) if args.reference_file else None,
        COND_KNOWNS=args.cond_knowns,
        create_deconv_tables_both_py=Path(args.create_deconv_tables_both_py),
        candidate_mixture_folder=Path(args.candidate_mixture_folder) if args.candidate_mixture_folder else None,
        mle_results_folder=Path(args.mle_results_folder) if args.mle_results_folder else None,
        mle_config_dict=mle_config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
