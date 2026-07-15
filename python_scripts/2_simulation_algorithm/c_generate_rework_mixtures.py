# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:09:24 2026

@author: jortk
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

#import json
import subprocess
#from pathlib import Path
#import yaml
#import pandas as pd
#from confidence import Configuration
#import random
#import numpy as np

def build_generate_command_spec(genotype_files, fixed_params, RESOURCES_PATH, CONFIG_YAML):
    #jar_file = RESOURCES_PATH / "dnastatistx-cli-2.5.0-DES-700-202601150926.jar"
    #CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    #cfg = load_dnax_config1(CONFIG_YAML)
    
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    #thresholds_cfg = Configuration(cfg["dnastatistx"]["thresholds"])
    jar_file = RESOURCES_PATH / cfg["dnastatistx"]["jar_file"]
    freqs_file = RESOURCES_PATH / cfg["dnastatistx"]["frequencies_file"]

    model = "--"+str(cfg["dnastatistx"]["model"])
    kit = str(cfg["dnastatistx"]["kit"])
    #thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    #thresholds_map.setdefault("DEFAULT", thresholds_cfg.DEFAULT)
    thresholds_map = dict(cfg["dnastatistx"]["thresholds"])

    if "DEFAULT" not in thresholds_map:
        raise KeyError(
            "config.yaml must define dnastatistx.thresholds.DEFAULT"
        )
    
    # if int(cfg['dnastatistx']['N_CONTRIBUTORS']) == 2:
    #     cond_knowns = ["Unknown1", "Unknown2"]
    #     reference_profiles = [str(U1_file), str(U2_file)]
        
    contributors = int(cfg['dnastatistx']['N_CONTRIBUTORS'])
    cond_knowns = [f"Unknown{i}" for i in range(1, contributors+1)]
    reference_profiles = [str(p) for p in genotype_files]
    # elif int(cfg['dnastatistx']['N_CONTRIBUTORS']) == 3:
    #     cond_knowns = ["Unknown1", "Unknown2"]

    return {
        "java_executable": "java",
        "jar_file": str(jar_file),
        "action": "calculate",
        "generate_flag": "--generate",
        "hypothesis": "-H0",
        "contributors": int(cfg['dnastatistx']['N_CONTRIBUTORS']),
        "cond_knowns": cond_knowns,
        "reference_profiles": reference_profiles,
        "model": model,
        "method": "--method-fixed",
        "fixed_params": fixed_params, #"EPH=8878.8 PHV=0.17728 DEG=0.93224 MX1=0.71412 MX2=0.28588", #1_1A2
        "degradation": True,
        "validation": True,
        "kit": kit,
        "thresholds": thresholds_map,
        "frac_threshold": str(cfg['dnastatistx']["frac_threshold"]), #"*=0.03",
        "population_file": str(freqs_file),
        "coancestry": str(cfg['dnastatistx']["coancestry"]), #"0.0",
        "dropin_prob": str(cfg['dnastatistx']["dropin_prob"]), #"1.0E-6",
        "dropin_lambda": str(cfg['dnastatistx']["dropin_lambda"]), #"1.0E-6",
        #"random_seed": 9103477288049225897,
    }

def render_generate_cmd(spec: dict, output_dir: Path) -> list[str]:
    thresholds_args = [f"{k}={v}" for k, v in spec["thresholds"].items()]
    cmd = [spec["java_executable"], "-jar", spec["jar_file"], spec["action"], spec["generate_flag"], spec["hypothesis"], "--contributors", str(spec["contributors"]), "--cond-knowns", *spec["cond_knowns"], "--reference-profile", *spec["reference_profiles"], spec["model"], spec["method"], "--fixed-parameters", *spec["fixed_params"].split()]
    if spec.get("degradation"): cmd.append("--degradation")
    if spec.get("validation"): cmd.append("--validation")
    cmd += ["--kit", spec["kit"], "--threshold", *thresholds_args, "--frac-threshold", spec["frac_threshold"], "--population", spec["population_file"], "--output", str(output_dir), "--coancestry", str(spec["coancestry"]),  "--dropin-prob", spec['dropin_prob'], "--dropin-lambda", spec['dropin_lambda']]#, "--random-seed", str(spec["random_seed"])]
    return cmd

def run_generate_profile(output_dir: Path, command_spec: dict) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = render_generate_cmd(command_spec, output_dir)
    print("\n=== DNAStatistX GENERATE ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (output_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (output_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        print("ERROR: Generation failed")
        print(proc.stderr)
        return False
    print(f"✓ Generated profiles saved in {output_dir}")
    return True

def parse2(s):
    rep, rest = s.split('_')
    return int(rep), rest

# ---------------------------------------------------------------------
# DNAStatistX / MLE loading helpers
# ---------------------------------------------------------------------

def load_h2_model_parameters(results_json_path):
    """
    Load H2 model parameters from a DNAStatistX results JSON file.

    Expected structure:
        results_json.loc["H2", "hypothesesResults"]["modelParameters"]
    """
    results_json_path = Path(results_json_path)

    if not results_json_path.exists():
        raise FileNotFoundError(f"Missing DNAStatistX results file: {results_json_path}")

    results_json = pd.read_json(results_json_path)
    h2_results = results_json.loc["H2", "hypothesesResults"]

    return h2_results["modelParameters"]


def normalize_mixture_proportions(mx):
    """
    Make mixture proportions positive and normalize them to sum to 1.
    """
    mx = np.asarray(mx, dtype=float)

    if np.any(~np.isfinite(mx)):
        raise ValueError(f"Mixture proportions contain non-finite values: {mx}")

    mx = np.clip(mx, 1e-12, None)
    mx = mx / mx.sum()

    return mx.tolist()


def mle_params_from_model_parameters(model_params):
    """
    Convert DNAStatistX modelParameters to the fixed-parameter names
    used by the DNAStatistX profile generator.
    """
    return {
        "EPH": float(model_params["expectedPeakHeight"]),
        "PHV": float(model_params["peakHeightVariance"]),
        "DEG": float(model_params["degradationSlope"]),
        "MX": normalize_mixture_proportions(model_params["mixtureProportions"]),
    }


def format_fixed_params(params):
    """
    Convert parameter dictionary to DNAStatistX fixed-parameter string.

    Example:
        EPH=8000 PHV=0.15 DEG=0.94 MX1=0.91 MX2=0.09
    """
    mx_params = " ".join(
        f"MX{i + 1}={value}"
        for i, value in enumerate(params["MX"])
    )

    return (
        f"EPH={params['EPH']} "
        f"PHV={params['PHV']} "
        f"DEG={params['DEG']} "
        f"{mx_params}"
    )


# ---------------------------------------------------------------------
# MCMC helpers for test4
# ---------------------------------------------------------------------

def get_phi_keys(samples):
    """
    Return phi keys sorted numerically: phi1, phi2, phi3, ...
    """
    return sorted(
        [
            key for key in samples.keys()
            if re.fullmatch(r"phi\d+", key)
        ],
        key=lambda key: int(key.replace("phi", "")),
    )


def load_mcmc_samples(results_mcmc_path):
    """
    Load the samples block from results_mcmc.json.

    Expected structure:
        {
            "samples": {
                "mu": [...],
                "sigma": [...],
                "beta": [...],
                "phi1": [...],
                "phi2": [...]
            }
        }
    """
    results_mcmc_path = Path(results_mcmc_path)

    if not results_mcmc_path.exists():
        raise FileNotFoundError(f"Missing MCMC results file: {results_mcmc_path}")

    with results_mcmc_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "samples" not in payload:
        raise KeyError(f"No 'samples' block found in {results_mcmc_path}")

    samples = payload["samples"]

    required_keys = ["mu", "sigma", "beta"]
    for key in required_keys:
        if key not in samples:
            raise KeyError(f"Missing samples['{key}'] in {results_mcmc_path}")

    phi_keys = get_phi_keys(samples)

    if not phi_keys:
        raise KeyError(
            f"No phi keys found in {results_mcmc_path}. "
            "Expected samples['phi1'], samples['phi2'], ..."
        )

    keys_to_check = required_keys + phi_keys
    sample_lengths = {key: len(samples[key]) for key in keys_to_check}

    if len(set(sample_lengths.values())) != 1:
        raise ValueError(
            f"Not all MCMC sample arrays have the same length in {results_mcmc_path}: "
            f"{sample_lengths}"
        )

    return samples


#def draw_all_parameters_from_mcmc(results_mcmc_path):
def draw_all_parameters_from_mcmc(results_mcmc_path, draw_index=None):
    """
    Draw EPH, PHV, DEG and mixture proportions from one shared MCMC row.

    Mapping:
        mu    -> EPH
        sigma -> PHV
        beta  -> DEG
        phi1, phi2, ... -> MX1, MX2, ...
    """
    results_mcmc_path = Path(results_mcmc_path)
    samples = load_mcmc_samples(results_mcmc_path)

    #n_samples = len(samples["mu"])
    #draw_index = int(np.random.randint(0, n_samples))
    n_samples = len(samples["mu"])

    if draw_index is None:
        draw_index = int(np.random.randint(0, n_samples))
    else:
        draw_index = int(draw_index)
        if draw_index < 0 or draw_index >= n_samples:
            raise ValueError(
                f"draw_index={draw_index} is outside MCMC range 0..{n_samples - 1}"
            )

    phi_keys = get_phi_keys(samples)

    mx = [
        float(samples[phi_key][draw_index])
        for phi_key in phi_keys
    ]

    params = {
        "EPH": float(samples["mu"][draw_index]),
        "PHV": float(samples["sigma"][draw_index]),
        "DEG": float(samples["beta"][draw_index]),
        "MX": normalize_mixture_proportions(mx),
    }

    metadata = {
        "draw_index": draw_index,
        "mcmc_results_path": str(results_mcmc_path),
        "mcmc_parameter_names": {
            "EPH": "mu",
            "PHV": "sigma",
            "DEG": "beta",
            "MX": phi_keys,
        },
    }

    return params, metadata

def load_sampled_theta_index(sample_dir):
    """
    Load the theta index saved by sample_from_joint.
    """
    path = Path(sample_dir) / "sampled_theta_index.json"

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("theta_draw_index")


# def save_sampled_generation_parameters(
#     *,
#     output_path,
#     params,
#     metadata,
#     mixture,
#     sample_name,
#     test_mode,
# ):
#     """
#     Save the sampled generation parameters for one sample folder.

#     Saved location for test4:
#         OUTPUT_ROOT / "mixtures" / mixture /
#         "sampled_joint_genotypes" / sample /
#         "sampled_generation_parameters.json"
#     """
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     payload = {
#         "mixture": mixture,
#         "sample": sample_name,
#         "test_mode": test_mode,
#         "mcmc_results_path": metadata.get("mcmc_results_path"),
#         "draw_index": metadata.get("draw_index"),
#         "dnastatistx_fixed_parameters": {
#             "EPH": float(params["EPH"]),
#             "PHV": float(params["PHV"]),
#             "DEG": float(params["DEG"]),
#             "MX": [float(x) for x in params["MX"]],
#         },
#         "mcmc_parameter_names": metadata.get("mcmc_parameter_names"),
#     }

#     output_path.write_text(
#         json.dumps(payload, indent=2),
#         encoding="utf-8",
#     )

#     print(f"[INFO] Saved sampled parameters -> {output_path}")

def save_sampled_generation_parameters(
    *,
    output_path,
    params,
    metadata,
    mixture,
    sample_name,
    test_mode,
):
    """
    Save sampled parameters in a DNAStatistX-like results.json format.

    This allows later loading with:

        results_json = pd.read_json(path)
        modelParameters = results_json.loc["H2", "hypothesesResults"]["modelParameters"]

    Only modelParameters is saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "hypothesesResults": {
            "H2": {
                "modelParameters": {
                    "mixtureProportions": [float(x) for x in params["MX"]],
                    "expectedPeakHeight": float(params["EPH"]),
                    "peakHeightVariance": float(params["PHV"]),
                    "degradationSlope": float(params["DEG"]),
                    "backwardStutterProportion": 0.0,
                    "forwardStutterProportion": 0.0,
                }
            }
        },
        "metadata": {
            "mixture": mixture,
            "sample": sample_name,
            "test_mode": test_mode,
            "mcmc_results_path": metadata.get("mcmc_results_path"),
            "draw_index": metadata.get("draw_index"),
            "mcmc_parameter_names": metadata.get("mcmc_parameter_names"),
        },
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] Saved sampled parameters -> {output_path}")
    
    
    
def save_sampled_theta_for_later(
    *,
    sample_dir,
    mixture,
    mcmc_mixture_folder,
    test_mode,
):
    """
    Save the sampled theta that belongs to this sampled genotype set.

    This is useful for test0: generation still uses the MLE parameters,
    but the theta draw used when sampling the genotype can be saved for
    later scripts.
    """
    sample_dir = Path(sample_dir)

    theta_draw_index = load_sampled_theta_index(sample_dir)

    if theta_draw_index is None:
        print(f"[WARNING] No sampled_theta_index.json found in {sample_dir}.")
        return

    if mcmc_mixture_folder is None:
        print(
            f"[WARNING] Cannot save sampled theta parameters for {sample_dir}, "
            "because mcmc_mixture_folder is None."
        )
        return

    results_mcmc_path = (
        Path(mcmc_mixture_folder)
        / mixture
        / "results_mcmc.json"
    )

    theta_params, theta_metadata = draw_all_parameters_from_mcmc(
        results_mcmc_path,
        draw_index=theta_draw_index,
    )

    save_sampled_generation_parameters(
        output_path=sample_dir / "sampled_theta_parameters.json",
        params=theta_params,
        metadata=theta_metadata,
        mixture=mixture,
        sample_name=sample_dir.name,
        test_mode=f"{test_mode}_sampled_theta_saved_for_later",
    )
# ---------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------

# def choose_generation_parameters(
#     *,
#     test_mode,
#     mixture,
#     rest,
#     mixture_folder,
#     traces_folder,
#     model_params,
#     mcmc_mixture_folder=None,
# ):
def choose_generation_parameters(
    *,
    test_mode,
    mixture,
    rest,
    mixture_folder,
    traces_folder,
    model_params,
    mcmc_mixture_folder=None,
    forced_draw_index=None,
):
    """
    Choose parameters for profile generation.

    test0:
        Use all MLE parameters from the current mixture.

    test1:
        Use EPH/PHV/DEG from the current mixture.
        Randomly choose mixture proportions from replicate 1, 2 or 3.

    test2:
        Use EPH/PHV/DEG from the current mixture.
        Use mixture proportions from the combined trace result.

    test3:
        Use EPH/PHV/DEG from the current mixture.
        Sample MX1 from a normal distribution around the current MLE.
        Only works for two-person mixtures.

    test4:
        Sample EPH, PHV, DEG and mixture proportions from results_mcmc.json.

    fallback:
        Use all MLE parameters from the current mixture.
    """
    mixture_folder = Path(mixture_folder)
    traces_folder = Path(traces_folder)

    params = mle_params_from_model_parameters(model_params)
    metadata = {}

    if test_mode == "test0":
        return params, metadata

    if test_mode == "test1":
        replicate_mx_values = []

        for replicate in [1, 2, 3]:
            replicate_results_path = (
                mixture_folder
                / f"{replicate}_{rest}"
                / "results.json"
            )

            replicate_model_params = load_h2_model_parameters(replicate_results_path)
            replicate_mx_values.append(replicate_model_params["mixtureProportions"])

        params["MX"] = normalize_mixture_proportions(
            random.choice(replicate_mx_values)
        )

        return params, metadata

    if test_mode == "test2":
        combined_results_path = traces_folder / rest / "results.json"
        combined_model_params = load_h2_model_parameters(combined_results_path)

        params["MX"] = normalize_mixture_proportions(
            combined_model_params["mixtureProportions"]
        )

        return params, metadata

    if test_mode == "test3":
        current_mx = model_params["mixtureProportions"]

        if len(current_mx) != 2:
            raise ValueError(
                "test3 is only implemented for two-person mixtures, "
                "because it samples MX1 and sets MX2 = 1 - MX1."
            )

        mx1 = np.random.normal(
            loc=float(current_mx[0]),
            scale=0.03,
        )

        params["MX"] = normalize_mixture_proportions([mx1, 1.0 - mx1])

        return params, metadata

    if test_mode == "test4":
        if mcmc_mixture_folder is None:
            raise ValueError("test_mode='test4' requires mcmc_mixture_folder.")

        results_mcmc_path = (
            Path(mcmc_mixture_folder)
            / mixture
            / "results_mcmc.json"
        )

        #return draw_all_parameters_from_mcmc(results_mcmc_path)
        return draw_all_parameters_from_mcmc(
            results_mcmc_path,
            draw_index=forced_draw_index,
        )

    return params, metadata


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def generate_mixtures(
    mixture_folder,
    traces_folder,
    RESOURCES_PATH,
    CONFIG_YAML,
    N_REPLICATES_PER_SAMPLE,
    test_mode,
    mcmc_mixture_folder=None,
):
    """
    Generate synthetic mixtures using DNAStatistX.

    For test4, this samples all parameters from the MCMC chain:
        mu    -> EPH
        sigma -> PHV
        beta  -> DEG
        phi1, phi2, ... -> MX1, MX2, ...

    It also saves the sampled parameters to:
        mixture_folder / mixture /
        "sampled_joint_genotypes" / sample /
        "sampled_generation_parameters.json"

    Parameters
    ----------
    mixture_folder : str or Path
        Folder containing mixture directories.

    traces_folder : str or Path
        Folder containing combined-trace results.
        Used only for test2.

    RESOURCES_PATH : str or Path
        DNAStatistX resources path.

    CONFIG_YAML : str or Path
        DNAStatistX config YAML.

    N_REPLICATES_PER_SAMPLE : int
        Number of synthetic replicate profiles to generate per sampled genotype set.

    test_mode : str
        One of:
            test0, test1, test2, test3, test4

    mcmc_mixture_folder : str or Path, optional
        Folder containing MCMC result folders.
        Required for test4.

        Expected path:
            mcmc_mixture_folder / mixture / "results_mcmc.json"
    """
    mixture_folder = Path(mixture_folder)
    traces_folder = Path(traces_folder)
    RESOURCES_PATH = Path(RESOURCES_PATH)
    CONFIG_YAML = Path(CONFIG_YAML)

    if mcmc_mixture_folder is not None:
        mcmc_mixture_folder = Path(mcmc_mixture_folder)

    if test_mode == "test4" and mcmc_mixture_folder is None:
        raise ValueError("test_mode='test4' requires mcmc_mixture_folder.")

    if not mixture_folder.exists():
        raise FileNotFoundError(f"mixture_folder does not exist: {mixture_folder}")

    if not CONFIG_YAML.exists():
        raise FileNotFoundError(f"CONFIG_YAML does not exist: {CONFIG_YAML}")

    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mixture_list = sorted(
        item.name
        for item in mixture_folder.iterdir()
        if item.is_dir()
    )

    print("[MODE] Generating new simulated mixtures with DNAStatistX...")
    print(f"[MODE] test_mode = {test_mode}")
    print(f"[INFO] Found {len(mixture_list)} mixture folders.")

    for mixture in mixture_list:
        rep, rest = parse2(mixture)

        results_dnax_path = mixture_folder / mixture / "results_dnax.json"

        if not results_dnax_path.exists():
            print(f"[WARNING] Missing {results_dnax_path}. Skipping {mixture}.")
            continue

        model_params = load_h2_model_parameters(results_dnax_path)

        joint_path = (
            mixture_folder
            / mixture
            / "sampled_joint_genotypes"
        )

        if not joint_path.exists():
            print(f"[WARNING] Missing {joint_path}. Skipping {mixture}.")
            continue

        sample_dirs = sorted(
            sample_dir
            for sample_dir in joint_path.glob("sample_*")
            if sample_dir.is_dir()
        )

        if not sample_dirs:
            print(f"[WARNING] No sample_* folders found in {joint_path}.")
            continue

        for sample_dir in sample_dirs:
            sample_number = sample_dir.name.split("_")[-1]

            genotype_files = sorted(sample_dir.glob("Unknown*_genotype.csv"))

            if not genotype_files:
                print(f"[WARNING] No genotype files found in {sample_dir}. Skipping.")
                continue
            
            # forced_draw_index = None

            # if test_mode == "test4":
            #     forced_draw_index = load_sampled_theta_index(sample_dir)
                
            forced_draw_index = None
            
            if test_mode == "test4":
                forced_draw_index = load_sampled_theta_index(sample_dir)
            
            if test_mode == "test0":
                save_sampled_theta_for_later(
                    sample_dir=sample_dir,
                    mixture=mixture,
                    mcmc_mixture_folder=mcmc_mixture_folder,
                    test_mode=test_mode,
                )
                
    
            
            params, metadata = choose_generation_parameters(
                test_mode=test_mode,
                mixture=mixture,
                rest=rest,
                mixture_folder=mixture_folder,
                traces_folder=traces_folder,
                model_params=model_params,
                mcmc_mixture_folder=mcmc_mixture_folder,
                forced_draw_index=forced_draw_index,
            )

            # params, metadata = choose_generation_parameters(
            #     test_mode=test_mode,
            #     mixture=mixture,
            #     rest=rest,
            #     mixture_folder=mixture_folder,
            #     traces_folder=traces_folder,
            #     model_params=model_params,
            #     mcmc_mixture_folder=mcmc_mixture_folder,
            # )

            fixed_params = format_fixed_params(params)

            if test_mode == "test4":
                sampled_params_json = sample_dir / "sampled_generation_parameters.json"

                save_sampled_generation_parameters(
                    output_path=sampled_params_json,
                    params=params,
                    metadata=metadata,
                    mixture=mixture,
                    sample_name=sample_dir.name,
                    test_mode=test_mode,
                )

            output_root = sample_dir / f"sample_{sample_number}"
            output_root.mkdir(parents=True, exist_ok=True)

            print(f"\n[INFO] Mixture: {mixture}")
            print(f"[INFO] Sample: {sample_dir.name}")
            print(f"[INFO] Output: {output_root}")

            if metadata.get("draw_index") is not None:
                print(f"[INFO] MCMC draw index: {metadata['draw_index']}")

            print(f"[INFO] Fixed parameters: {fixed_params}")

            for replicate_index in range(N_REPLICATES_PER_SAMPLE):
                print(
                    f"  [STEP] Generating replicate {replicate_index + 1} "
                    f"of {N_REPLICATES_PER_SAMPLE}"
                )

                cmd_spec = build_generate_command_spec(
                    genotype_files,
                    fixed_params,
                    RESOURCES_PATH,
                    CONFIG_YAML,
                )

                run_generate_profile(output_root, cmd_spec)

    print("\n✅ Generation complete.")