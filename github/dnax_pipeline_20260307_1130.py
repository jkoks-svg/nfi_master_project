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
# import os
# import glob

import yaml
# import numpy as np
# import pandas as pd
from confidence import Configuration
from postprocess_add_priors import postprocess_add_priors_helper

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

# def load_dnax_config(CONFIG_YAML):
#     if not CONFIG_YAML.exists():
#         raise FileNotFoundError(f"Missing config: {CONFIG_YAML}")
#     with CONFIG_YAML.open("r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)
#     if "dnastatistx" not in cfg:
#         raise KeyError(f"'dnastatistx' missing in {CONFIG_YAML}")
#     return cfg["dnastatistx"]


def build_command_spec(RESOURCES_PATH, CONFIG_YAML, REFERENCE_FILE,COND_KNOWNS) -> dict:
    """
    Single source of truth for *all* command parameters.
    """
    #CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    #cfg = load_dnax_config(CONFIG_YAML)
    thresholds_cfg = Configuration(cfg['dnastatistx']["thresholds"])

    jar_file = RESOURCES_PATH / cfg['dnastatistx']["jar_file"]
    freqs_file = RESOURCES_PATH / cfg['dnastatistx']["frequencies_file"]

    model = str(cfg['dnastatistx']["model"])
    #kit = str(cfg["kit"])
    #threads = int(cfg["threads"])
    method = str(cfg['dnastatistx']["method"])

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

        "method": str(cfg['dnastatistx']["method"]),
        "method_flag": method_flag,

        "model": str(cfg['dnastatistx']["model"]),
        "model_flag": f"--{model}",

        "H1": {"contributors": int(cfg['dnastatistx']['N_CONTRIBUTORS']), "cond_knowns": COND_KNOWNS},
        "H2": {"contributors": int(cfg['dnastatistx']['N_CONTRIBUTORS'])},

        "kit": str(cfg['dnastatistx']["kit"]),
        "frequencies_file": str(freqs_file),
        "threads": int(cfg['dnastatistx']["threads"]),
        
        "thresholds": thresholds_map,
        
        "coancestry": str(cfg['dnastatistx']["coancestry"]),
        "frac_threshold": str(cfg['dnastatistx']["frac_threshold"]),
        "dropin_prob" : str(cfg['dnastatistx']["dropin_prob"]), 
        "dropin_lambda" : str(cfg['dnastatistx']["dropin_lambda"]), 
        "random_seed" : str(cfg['dnastatistx']["random_seed"]),
        "deconvolution": True,
        "degradation": cfg['dnastatistx']["degradation"] #False # (model == "model-euroformix"), SWITCHED_OFF
    }


def render_cmd_from_spec(spec: dict, mixture_file: Path, output_base: Path) -> list[str]:
    thresholds = spec["thresholds"]
    default_thr = thresholds.get("DEFAULT")
    thresholds_args = [f"{k}={v}" for k, v in thresholds.items() if k != "DEFAULT"]

    cmd = [
        spec["java_executable"], "-jar", spec["jar_file"],
        spec["action"],
        "--trace-profile", str(mixture_file),
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


def run_one_mixture_dnax(mixture_file: Path, output_dir: Path, command_spec: dict) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "results"

    # Clean previous DNAStatistX outputs
    for ext in (".json", ".txt"):
        f = output_base.with_suffix(ext)
        if f.exists():
            f.unlink()

    cmd = render_cmd_from_spec(command_spec, mixture_file, output_base)

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
# Run-level parameter bundle (both steps)
# ----------------------------------------------------------------------

def build_pipeline_runlevel_params(command_spec: dict, example_cmd: list[str] | None) -> dict:
    jar_path = Path(command_spec["jar_file"])
    freq_path = Path(command_spec["frequencies_file"])
    ref_path = Path(command_spec["reference_profile"])

    params = {
        # pipeline paths / switches
        #"MIXTURES_DIR": str(MIXTURES_DIR),
        #"OUTPUT_ROOT": str(OUTPUT_ROOT),
        #"OUTPUT_MIXTURES_DIR": str(OUTPUT_MIXTURES_DIR),
        #"SKIP_FIRST_N": SKIP_FIRST_N,
        #"MAX_MIXTURES": MAX_MIXTURES,
        #"COND_KNOWNS": COND_KNOWNS,
        #"H2_FILTER_THRESHOLD": H2_FILTER_THRESHOLD,

        #"SCRIPT_PATH": str(SCRIPT_PATH),
        #"RESOURCES_PATH": str(RESOURCES_PATH),
        #"CONFIG_YAML": str(CONFIG_YAML),
        #"REFERENCE_FILE": str(REFERENCE_FILE),

        # step 1: DNAStatistX command spec
        "DNASX_COMMAND_SPEC": deepcopy(command_spec),

        # step 2: genotype priors settings
        "PRIORS": {
      #      "FREQ_CSV_FOR_PRIORS": str(FREQ_CSV_FOR_PRIORS),
            #"EPS": EPS,
            #"RARE_FREQ": RARE_FREQ,
            "expected_input_filename": "results_marginal_deconvolutions.csv",
            "output_filename": "results_marginal_deconvolutions_with_prior_and_LR.csv",
        },

        # integrity hashes
        "HASHES": {
        #    "config_yaml_sha256": file_sha256(CONFIG_YAML),
            "reference_file_sha256": file_sha256(ref_path),
            "jar_sha256": file_sha256(jar_path),
            "dnax_frequencies_sha256": file_sha256(freq_path),
       #     "priors_frequency_csv_sha256": file_sha256(FREQ_CSV_FOR_PRIORS),
        },
    }

    if example_cmd is not None:
        params["EXAMPLE_DNASX_CMD_LIST"] = example_cmd
        params["EXAMPLE_DNASX_CMD_STRING"] = " ".join(example_cmd)

    return params




# ----------------------------------------------------------------------
# Pipeline function (callable from other scripts)
# ----------------------------------------------------------------------

def dnax_single_mixtures(
    mixtures_dir,
    output_root,
    RESOURCES_PATH,
    CONFIG_YAML,
    REFERENCE_FILE,
    COND_KNOWNS,
    skip_first_n=0,
    max_mixtures=None,
):

    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    RARE_FREQ = float(cfg['RARE_FREQ'])
    H2_FILTER_THRESHOLD = float(cfg['H2_FILTER_THRESHOLD'])
    DELETE_INTERMEDIATE_FILES = cfg['DELETE_INTERMEDIATE_FILES']
    freqs_file = RESOURCES_PATH / cfg['dnastatistx']["frequencies_file"]
    
    mixtures_dir = Path(mixtures_dir)
    output_root = Path(output_root)
    output_mixtures_dir = output_root / "mixtures"

    output_root.mkdir(parents=True, exist_ok=True)
    output_mixtures_dir.mkdir(parents=True, exist_ok=True)

    mixtures = sorted(mixtures_dir.glob("*.txt"))
    if not mixtures:
        raise SystemExit(f"No .txt mixtures found in {mixtures_dir}")

    mixtures_to_run = mixtures[skip_first_n:]
    if max_mixtures is not None:
        mixtures_to_run = mixtures_to_run[:max_mixtures]

    print(f"Found {len(mixtures)} mixture .txt files")
    print(f"Running {len(mixtures_to_run)} mixtures (skipped first {skip_first_n})")

    # ---- build spec once ----
    command_spec = build_command_spec(RESOURCES_PATH, CONFIG_YAML, REFERENCE_FILE, COND_KNOWNS)

    example_cmd = None
    if mixtures_to_run:
        example_mix = mixtures_to_run[0]
        example_out = output_mixtures_dir / example_mix.stem / "results"
        example_cmd = render_cmd_from_spec(command_spec, example_mix, example_out)

    # ---- run-level logging ----
    runlevel_params = build_pipeline_runlevel_params(command_spec, example_cmd)
    write_params_txt(output_root, runlevel_params, "run_parameters_PIPELINE.txt")
    write_params_json(output_root, runlevel_params, "run_parameters_PIPELINE.json")

    # ---- step 1: run DNAStatistX ----
    n_ok = 0
    n_fail = 0

    for mix_file in mixtures_to_run:
        mix_id = mix_file.stem
        mix_output_dir = output_mixtures_dir / mix_id
        ok = run_one_mixture_dnax(mix_file, mix_output_dir, command_spec)

        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print("\n=== DNAStatistX batch completed ===")
    print(f"Success: {n_ok}, Failed: {n_fail}")

    # ---- step 2: postprocess ----
    #pp_ok, pp_bad = postprocess_add_priors(output_mixtures_dir, freqs_file, RARE_FREQ, H2_FILTER_THRESHOLD)#freq_csv_for_priors)
    pp_ok, pp_bad = postprocess_add_priors_helper(output_mixtures_dir, freqs_file, RARE_FREQ, H2_FILTER_THRESHOLD, DELETE_INTERMEDIATE_FILES)#freq_csv_for_priors)

    print("\n=== Priors post-processing completed ===")
    print(f"Success: {pp_ok}, Skipped/Failed: {pp_bad}")

    print("\nPipeline done.")

    return {
        "dnax_success": n_ok,
        "dnax_failed": n_fail,
        "postprocess_success": pp_ok,
        "postprocess_failed": pp_bad,
    }


# ----------------------------------------------------------------------
# CLI wrapper (optional)
# ----------------------------------------------------------------------

def main():

    MIXTURES_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\mixtures_debug_1B2_filled")
    OUTPUT_ROOT = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_1B2_filled")    
    RESOURCES_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources")
    
    REFERENCE_FILE = RESOURCES_PATH / "2G_reference_format_ordered.txt" #ADJUST WHEN USING wiskundig correcte LR
    COND_KNOWNS =  "ABCD1234NL#01" #ADJUST WHEN USING wiskundig correcte LR
    CONFIG_YAML = RESOURCES_PATH / "config_2p_Bayes_test.yaml"
    
    dnax_single_mixtures(
        mixtures_dir=MIXTURES_DIR,
        output_root=OUTPUT_ROOT,
        RESOURCES_PATH=RESOURCES_PATH,
        CONFIG_YAML= CONFIG_YAML,
        REFERENCE_FILE = REFERENCE_FILE,
        COND_KNOWNS=COND_KNOWNS,
    )


if __name__ == "__main__":
    main()

