# -*- coding: utf-8 -*-
"""
MCMC replacement for dnax_pipeline_traces_replicates.py

Goal
----
Keep the same input traversal and output folders as dnax_pipeline_traces_replicates.py,
but do NOT call DNAStatistX. Instead, use the MCMC/deconvolution code from
create_deconv_tables_both.py to create, per combined replicate analysis:

    results_joint_deconvolution_H2_clean.csv
    results_marginal_prior_LR.csv

Expected folder layout, same as the DNAStatistX replicate pipeline:

    mixture_folder/
        <mixture>/
            sampled_joint_genotypes/
                sample_0001/
                    *Trace*.txt
                    mixtures/
                        combined_<mixture>_sample_0001/
                            results_joint_deconvolution_H2_clean.csv
                            results_marginal_prior_LR.csv

For each sample folder, the three traces are:
    1. the original trace: MIXTURES_DIR / f"{mixture}.txt"
    2. the first simulated *Trace*.txt in sample folder
    3. the second simulated *Trace*.txt in sample folder

The MCMC target is p(theta | trace_1, trace_2, trace_3). The posterior genotype
probabilities are then obtained by evaluating each replicate with the same
posterior parameter draws and combining the replicate log-likelihoods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
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
# Logging helpers copied/adapted from dnax_pipeline_traces_replicates.py
# ----------------------------------------------------------------------
def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def file_sha256(path: Path) -> str | None:
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

    lines.append("\nRUN-LEVEL PARAMETERS (MCMC pipeline)")
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


def build_mcmc_runlevel_params(
    *,
    create_deconv_tables_both_py: Path,
    resources_path: Path,
    config_yaml: Path,
    mixtures_dir: Path,
    reference_file: Path | None,
    cond_knowns: str | None,
    mcmc_config: dict,
    candidate_mixture_folder: Path | None = None,
) -> dict:
    with Path(config_yaml).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    freqs_file = Path(resources_path) / cfg["dnastatistx"]["frequencies_file"]
    kit_properties_txt = Path(resources_path) / cfg["kit_properties_txt"]

    return {
        "PIPELINE_MODE": "MCMC replacement for DNAStatistX deconvolution step",
        "CREATE_DECONV_TABLES_BOTH_PY": str(create_deconv_tables_both_py),
        "RESOURCES_PATH": str(resources_path),
        "CONFIG_YAML": str(config_yaml),
        "MIXTURES_DIR": str(mixtures_dir),
        "REFERENCE_FILE_kept_for_same_signature": str(reference_file) if reference_file else None,
        "COND_KNOWNS_kept_for_same_signature": cond_knowns,
        "MCMC_CONFIG": deepcopy(mcmc_config),
        "CANDIDATE_MIXTURE_FOLDER_DNASTATISTX": str(candidate_mixture_folder) if candidate_mixture_folder else None,
        "CANDIDATE_CSV_PATTERN": "<candidate_mixture_folder>/<mixture>/sampled_joint_genotypes/<sample>/mixtures/combined_<mixture>_<sample>/results_joint_deconvolution_H2_clean.csv",
        "OUTPUT_FILES_PER_COMBINED_TRACE": [
            "results_joint_deconvolution_H2_clean.csv",
            "results_marginal_prior_LR.csv",
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
# MCMC replacement functions
# ----------------------------------------------------------------------
def default_mcmc_config() -> dict:
    return {
        "n_iter": 10000,
        "burnin": 2000,
        "thin": 1,
        "proposal_scales": (0.12, 0.10, 0.10, 0.10),
        "seed": 12345,
        "max_posterior_draws": 3,
        # These replace the DNAStatistX results.json starting values.
        "init_mu": 1500.0,
        "init_sigma": 0.15,
        "init_phi1": 0.70,
        "init_beta": 1.0,
        # Prior bounds/shape parameters as used by your MCMC code.
        "mu_1": 40000,
        "sigma_1": 1,
        # Candidate-pair filtering from the old DNAStatistX pipeline output.
        "require_candidate_csv": True,
        "candidate_min_probability": 0.0,
        "candidate_max_rows_per_locus": None,
        "candidate_add_contributor_swaps": False,
    }


def candidate_csv_path_for_sample(candidate_mixture_folder: Path, mixture: str, sample: str) -> Path:
    """Return the DNAStatistX candidate CSV for one mixture/sample."""
    trace_id = f"combined_dnax_{mixture}_{sample}"
    return (
        Path(candidate_mixture_folder)
        / mixture
        / "sampled_joint_genotypes"
        / sample
        / "mixtures"
        / trace_id
        / "results_joint_deconvolution_H2_clean.csv"
    )


def load_candidate_pairs_from_dnax_csv(
    *,
    deconv,
    candidate_csv: Path,
    mcmc_config: dict,
):
    """Load candidate genotype pairs from an existing DNAStatistX output CSV."""
    candidate_csv = Path(candidate_csv)

    if not candidate_csv.exists():
        msg = f"Candidate CSV from DNAStatistX pipeline not found: {candidate_csv}"
        if mcmc_config.get("require_candidate_csv", True):
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}; continuing without candidate-pair filtering.")
        return None

    print(f"[INFO] Loading candidate pairs from DNAStatistX CSV: {candidate_csv}")
    candidate_pairs_by_locus = deconv.load_candidate_genotype_pairs(
        candidate_csv,
        min_probability=mcmc_config.get("candidate_min_probability", 0.0),
        max_rows_per_locus=mcmc_config.get("candidate_max_rows_per_locus"),
        add_contributor_swaps=mcmc_config.get("candidate_add_contributor_swaps", False),
    )
    n_loci = len(candidate_pairs_by_locus)
    n_pairs = sum(len(v) for v in candidate_pairs_by_locus.values())
    print(f"[INFO] Loaded {n_pairs} candidate genotype pairs across {n_loci} loci")
    return candidate_pairs_by_locus

def load_and_fill_trace_dfs(*, deconv, trace_files: list[Path]) -> list[pd.DataFrame]:
    """
    Read the original trace plus the two simulated trace profiles,
    then fill missing alleles across the triplicate in the same way as
    create_deconv_tables_both.py.

    The output list has the same order as trace_files:
        1. original profile
        2. first trace profile
        3. second trace profile
    """
    trace_dfs = [
        pd.read_csv(trace_file, sep="\t", dtype=str).fillna("")
        for trace_file in trace_files
    ]

    return deconv.fill_missing_alleles_in_replicate_dfs(trace_dfs)

def build_rework_params_from_mcmc_trace_dfs(
    *,
    deconv,
    trace_dfs: list[pd.DataFrame], #trace_files: list[Path],
    resources_path: Path,
    config_yaml: Path,
    mcmc_config: dict,
    candidate_pairs_by_locus=None,
):
    """
    Same idea as create_deconv_tables_both.build_rework_params_from_mcmc(),
    but it takes explicit trace file paths and does not need a DNAStatistX results.json
    for initial values.
    """
    common = deconv.build_common_inputs_for_mcmc(
        RESOURCES_PATH=resources_path,
        CONFIG_YAML=config_yaml,
    )

    if common["N_contributors"] != 2:
        raise NotImplementedError("Current MCMC implementation is written for 2 contributors.")

    observed_alleles_list = []
    thresholds_list = []

    # for trace_file in trace_files:
    #     mixture_df = pd.read_csv(trace_file, sep="\t")
    #     observed_alleles, thresholds = deconv.extract_observed_alleles_and_thresholds(
    #         mixture_df,
    #         common["thresholds_map"],
    #         common["frac_threshold"],
    #         mode="rework",
    #     )
    #     observed_alleles_list.append(observed_alleles)
    #     thresholds_list.append(thresholds)
    for mixture_df in trace_dfs:
        observed_alleles, thresholds = deconv.extract_observed_alleles_and_thresholds(
            mixture_df.copy(),
            common["thresholds_map"],
            common["frac_threshold"],
            mode="rework",
        )
        observed_alleles_list.append(observed_alleles)
        thresholds_list.append(thresholds)
        
    print("Initial MCMC values")
    print("init_mu", mcmc_config["init_mu"])
    print("init_sigma", mcmc_config["init_sigma"])
    print("init_phi1", mcmc_config["init_phi1"])
    print("init_beta", mcmc_config["init_beta"])

    print("Building posterior parameter samples with joint replicate MCMC...")
    mcmc_result = deconv.run_mcmc_for_rework_parameters(
        observed_alleles_list=observed_alleles_list,
        all_alleles_by_locus=common["all_alleles"],
        thresholds_list=thresholds_list,
        freq_dict=common["freq_dict"],
        RARE_FREQ=common["RARE_FREQ"],
        C=common["C"],
        lam=common["lam"],
        N_contributors=common["N_contributors"],
        n_iter=mcmc_config["n_iter"],
        burnin=mcmc_config["burnin"],
        thin=mcmc_config["thin"],
        proposal_scales=mcmc_config["proposal_scales"],
        seed=mcmc_config["seed"],
        init_mu=mcmc_config["init_mu"],
        init_sigma=mcmc_config["init_sigma"],
        init_phi1=mcmc_config["init_phi1"],
        init_beta=mcmc_config["init_beta"],
        mu_1=mcmc_config["mu_1"],
        sigma_1=mcmc_config["sigma_1"],
        kit_properties_df=common["kit_properties_df"],
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )

    print(
        "MCMC finished: "
        f"acc={mcmc_result['acceptance_rate']:.3f}, "
        f"mu={mcmc_result['posterior_mean']['mu']:.2f}, "
        f"sigma={mcmc_result['posterior_mean']['sigma']:.4f}, "
        f"phi={mcmc_result['posterior_mean']['phi']}, "
        f"beta={mcmc_result['posterior_mean']['beta']:.4f}"
    )

    samples = mcmc_result["samples"]
    params_lst = [
        ([float(phi1), float(phi2)], float(mu), float(sigma), common["C"], common["lam"], float(beta))
        for mu, sigma, phi1, phi2, beta in zip(
            samples["mu"],
            samples["sigma"],
            samples["phi1"],
            samples["phi2"],
            samples["beta"],
        )
    ]

    max_posterior_draws = mcmc_config.get("max_posterior_draws")
    if max_posterior_draws is not None and len(params_lst) > max_posterior_draws:
        rng = np.random.default_rng(mcmc_config["seed"])
        idx = rng.choice(len(params_lst), size=max_posterior_draws, replace=False)
        params_lst = [params_lst[i] for i in idx]

    return params_lst, mcmc_result, common


# def deconvolve_one_replicate_full_df(
#     *,
#     deconv,
#     trace_file: Path,
#     replicate_number: int,
#     combined_trace_id: str,
#     params_lst: list,
#     common: dict,
#     candidate_pairs_by_locus=None,
# ) -> pd.DataFrame:
#     """Create the full joint posterior table for one replicate trace."""
#     mixture_df = pd.read_csv(trace_file, sep="\t")
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

    # These columns are required by create_deconv_tables_both.build_combined_analysis_df().
    joint_df["mixture"] = f"{replicate_number}_{combined_trace_id}"
    joint_df["replicate"] = replicate_number
    joint_df["mixture_combined"] = combined_trace_id

    return joint_df


# def combined_observed_alleles_for_trace_files(*, deconv, trace_files: list[Path], common: dict) -> dict:
#     """Union observed alleles across the three replicates, keeping the maximum height."""
#     combined_observed: dict = {}

#     for trace_file in trace_files:
#         mixture_df = pd.read_csv(trace_file, sep="\t")
#         observed_alleles, _ = deconv.extract_observed_alleles_and_thresholds(
#             mixture_df,
#             common["thresholds_map"],
#             common["frac_threshold"],
#             mode="rework",
#         )
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


# def export_combined_rework_csvs(
#     *,
#     deconv,
#     replicate_joint_dfs: list[pd.DataFrame],
#     trace_files: list[Path],
#     output_dir: Path,
#     common: dict,
# ):
def export_combined_rework_csvs(
    *,
    deconv,
    replicate_joint_dfs: list[pd.DataFrame],
    trace_dfs: list[pd.DataFrame],
    output_dir: Path,
    common: dict,
):
    """Write the exact two output CSVs that the downstream pipeline expects."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df_rework_input = pd.concat(replicate_joint_dfs, ignore_index=True)
    df_rework_posteriors = deconv.build_combined_analysis_df(df_rework_input)

    h2_filter_threshold = float(common["cfg"]["H2_FILTER_THRESHOLD"])

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
            joint_df_clean["Probability"] >= h2_filter_threshold,
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
    # observed_alleles = combined_observed_alleles_for_trace_files(
    #     deconv=deconv,
    #     trace_files=trace_files,
    #     common=common,
    # )
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


def write_mcmc_results_json(output_dir: Path, mcmc_result: dict, trace_files: list[Path], mcmc_config: dict):
    """
    Optional lightweight results.json replacement.
    This is not a DNAStatistX JSON; it records MCMC diagnostics so every combined folder is self-contained.
    """
    payload = {
        "producer": "mcmc_pipeline_traces_replicates.py",
        "note": "This is not a DNAStatistX results.json. The deconvolution CSVs were produced by MCMC.",
        "trace_files": [str(p) for p in trace_files],
        "mcmc_config": mcmc_config,
        "acceptance_rate": mcmc_result.get("acceptance_rate"),
        "posterior_mean": mcmc_result.get("posterior_mean"),
        "posterior_ci95": mcmc_result.get("posterior_ci95"),
    }
    out = output_dir / "results_mcmc.json"
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"[OK] Wrote {out}")


def mcmc_three_mixtures_combined(
    *,
    mixture_folder,
    RESOURCES_PATH,
    CONFIG_YAML,
    MIXTURES_DIR,
    REFERENCE_FILE=None,
    COND_KNOWNS=None,
    create_deconv_tables_both_py=DEFAULT_CREATE_DECONV_TABLES_BOTH_PY,
    candidate_mixture_folder=None,
    mcmc_config_dict=None,
    overwrite=False,
):
    """
    Drop-in replacement for dnax_three_mixtures_combined(...), except the deconvolution
    CSVs are created by MCMC instead of DNAStatistX.

    candidate_mixture_folder points to the OLD DNAStatistX pipeline output. The script
    reads candidate genotype pairs from that folder and writes new MCMC outputs under
    mixture_folder. If candidate_mixture_folder is None, mixture_folder is used.

    REFERENCE_FILE and COND_KNOWNS are accepted to keep the same call shape, but are
    not used by the H2 deconvolution output generation.
    """
    mixture_folder = Path(mixture_folder)
    candidate_mixture_folder = Path(candidate_mixture_folder) if candidate_mixture_folder is not None else mixture_folder
    resources_path = Path(RESOURCES_PATH)
    config_yaml = Path(CONFIG_YAML)
    mixtures_dir = Path(MIXTURES_DIR)
    reference_file = Path(REFERENCE_FILE) if REFERENCE_FILE is not None else None
    create_deconv_tables_both_py = Path(create_deconv_tables_both_py)

    deconv = import_create_deconv_module(create_deconv_tables_both_py)

    mcmc_config = default_mcmc_config()
    if mcmc_config_dict:
        mcmc_config.update(mcmc_config_dict)

    # Ensure proposal_scales is a tuple if read from JSON/CLI as list.
    if isinstance(mcmc_config.get("proposal_scales"), list):
        mcmc_config["proposal_scales"] = tuple(mcmc_config["proposal_scales"])

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
            trace_id = f"combined_mcmc_{mixture}_{sample}"
            output_dir = output_mixtures_dir / trace_id

            results_marginal_csv = output_dir / "results_marginal_prior_LR.csv"
            print(results_marginal_csv)
            if results_marginal_csv.exists() and not overwrite:
                print("Skipping:", output_root)
                continue

            output_root.mkdir(parents=True, exist_ok=True)
            output_mixtures_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            runlevel_params = build_mcmc_runlevel_params(
                create_deconv_tables_both_py=create_deconv_tables_both_py,
                resources_path=resources_path,
                config_yaml=config_yaml,
                mixtures_dir=mixtures_dir,
                reference_file=reference_file,
                cond_knowns=COND_KNOWNS,
                mcmc_config=mcmc_config,
                candidate_mixture_folder=candidate_mixture_folder,
            )
            write_params_txt(output_root, runlevel_params, "run_parameters_PIPELINE.txt")
            write_params_json(output_root, runlevel_params, "run_parameters_PIPELINE.json")

            print(f"\n=== Running MCMC TRACE {trace_id} ===")

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
            candidate_csv = candidate_csv_path_for_sample(
                candidate_mixture_folder=candidate_mixture_folder,
                mixture=mixture,
                sample=sample,
            )
            candidate_pairs_by_locus = load_candidate_pairs_from_dnax_csv(
                deconv=deconv,
                candidate_csv=candidate_csv,
                mcmc_config=mcmc_config,
            )

            # params_lst, mcmc_result, common = build_rework_params_from_mcmc_trace_files(
            #     deconv=deconv,
            #     trace_files=trace_files,
            #     resources_path=resources_path,
            #     config_yaml=config_yaml,
            #     mcmc_config=mcmc_config,
            #     candidate_pairs_by_locus=candidate_pairs_by_locus,
            # )
            params_lst, mcmc_result, common = build_rework_params_from_mcmc_trace_dfs(
                deconv=deconv,
                trace_dfs=trace_dfs,
                resources_path=resources_path,
                config_yaml=config_yaml,
                mcmc_config=mcmc_config,
                candidate_pairs_by_locus=candidate_pairs_by_locus,
            )

            # replicate_joint_dfs = []
            # for replicate_number, trace_file in enumerate(trace_files, start=1):
            #     print(f"Creating MCMC deconvolution for replicate {replicate_number}: {trace_file}")
            #     replicate_joint_dfs.append(
            #         deconvolve_one_replicate_full_df(
            #             deconv=deconv,
            #             trace_file=trace_file,
            #             replicate_number=replicate_number,
            #             combined_trace_id=trace_id,
            #             params_lst=params_lst,
            #             common=common,
            #             candidate_pairs_by_locus=candidate_pairs_by_locus,
            #         )
            #     )

            replicate_joint_dfs = []
            for replicate_number, (trace_file, mixture_df) in enumerate(
                zip(trace_files, trace_dfs),
                start=1,
            ):
                print(f"Creating MCMC deconvolution for replicate {replicate_number}: {trace_file}")
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





            # export_combined_rework_csvs(
            #     deconv=deconv,
            #     replicate_joint_dfs=replicate_joint_dfs,
            #     trace_files=trace_files,
            #     output_dir=output_dir,
            #     common=common,
            # )
            
            export_combined_rework_csvs(
                deconv=deconv,
                replicate_joint_dfs=replicate_joint_dfs,
                trace_dfs=trace_dfs,
                output_dir=output_dir,
                common=common,
            )

            write_mcmc_results_json(output_dir, mcmc_result, trace_files, mcmc_config)

            print("\nPipeline done for", trace_id)

# Backwards-compatible name: lets you keep your old call style.
dnax_three_mixtures_combined = mcmc_three_mixtures_combined

# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the dnax_pipeline_traces_replicates folder traversal, but create deconvolution CSVs with MCMC."
    )
    parser.add_argument("--mixture-folder", required=True, help="The .../output.../mixtures folder containing mixture subfolders.")
    parser.add_argument("--resources-path", required=True, help="Path to resources folder.")
    parser.add_argument("--config-yaml", required=True, help="Path to config.yaml.")
    parser.add_argument("--mixtures-dir", required=True, help="Folder containing original mixture txt files, e.g. mixtures_debug_2p.")
    parser.add_argument("--reference-file", default=None, help="Kept for same signature/logging; not used for H2 MCMC deconvolution CSVs.")
    parser.add_argument("--cond-knowns", default=None, help="Kept for same signature/logging; not used for H2 MCMC deconvolution CSVs.")
    parser.add_argument("--create-deconv-tables-both-py", default=str(DEFAULT_CREATE_DECONV_TABLES_BOTH_PY))
    parser.add_argument("--candidate-mixture-folder", default=None, help="Old DNAStatistX .../mixtures folder used to read candidate CSVs. Defaults to --mixture-folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results_marginal_prior_LR.csv outputs.")

    # Common MCMC knobs
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    parser.add_argument("--max-posterior-draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-mu", type=float, default=None)
    parser.add_argument("--init-sigma", type=float, default=None)
    parser.add_argument("--init-phi1", type=float, default=None)
    parser.add_argument("--init-beta", type=float, default=None)
    parser.add_argument("--candidate-min-probability", type=float, default=None)
    parser.add_argument("--candidate-max-rows-per-locus", type=int, default=None)
    parser.add_argument("--allow-missing-candidate-csv", action="store_true", help="Continue without candidate-pair filtering if the DNAStatistX candidate CSV is missing.")
    parser.add_argument("--candidate-add-contributor-swaps", action="store_true", help="Also include swapped contributor order for candidate genotype pairs.")
    return parser.parse_args()


def main():
    args = parse_args()

    mcmc_config = {}
    cli_to_key = {
        "n_iter": "n_iter",
        "burnin": "burnin",
        "thin": "thin",
        "max_posterior_draws": "max_posterior_draws",
        "seed": "seed",
        "init_mu": "init_mu",
        "init_sigma": "init_sigma",
        "init_phi1": "init_phi1",
        "init_beta": "init_beta",
        "candidate_min_probability": "candidate_min_probability",
        "candidate_max_rows_per_locus": "candidate_max_rows_per_locus",
    }
    for attr, key in cli_to_key.items():
        value = getattr(args, attr)
        if value is not None:
            mcmc_config[key] = value
    if args.allow_missing_candidate_csv:
        mcmc_config["require_candidate_csv"] = False
    if args.candidate_add_contributor_swaps:
        mcmc_config["candidate_add_contributor_swaps"] = True

    mcmc_three_mixtures_combined(
        mixture_folder=Path(args.mixture_folder),
        RESOURCES_PATH=Path(args.resources_path),
        CONFIG_YAML=Path(args.config_yaml),
        MIXTURES_DIR=Path(args.mixtures_dir),
        REFERENCE_FILE=Path(args.reference_file) if args.reference_file else None,
        COND_KNOWNS=args.cond_knowns,
        create_deconv_tables_both_py=Path(args.create_deconv_tables_both_py),
        candidate_mixture_folder=Path(args.candidate_mixture_folder) if args.candidate_mixture_folder else None,
        mcmc_config_dict=mcmc_config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
