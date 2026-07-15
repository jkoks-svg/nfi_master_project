# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:40:55 2026

@author: jortk
"""


#dnax_pipeline_traces_replicates


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


PYTHON_SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(PYTHON_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SCRIPTS_DIR))

import yaml
import numpy as np
import pandas as pd
#from confidence import Configuration
#from postprocess_add_priors2 import postprocess_add_priors_helper2
from helpers.postprocess_add_priors import postprocess_add_priors_helper



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



def build_command_spec(RESOURCES_PATH, CONFIG_YAML, REFERENCE_FILE,COND_KNOWNS) -> dict:
    """
    Single source of truth for *all* command parameters.
    """
    #CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    #cfg = load_dnax_config(CONFIG_YAML)
    #thresholds_cfg = Configuration(cfg['dnastatistx']["thresholds"])

    jar_file = RESOURCES_PATH / cfg["dnastatistx"]["jar_file"]
    freqs_file = RESOURCES_PATH / cfg["dnastatistx"]["frequencies_file"]

    model = str(cfg["dnastatistx"]["model"])
    kit = str(cfg["dnastatistx"]["kit"])
    threads = int(cfg["dnastatistx"]["threads"])
    method = str(cfg["dnastatistx"]["method"])

    method_flag = {
        "mle": "--method-maximize",
        "maximize": "--method-maximize",
        "integrate": "--method-integrate",
    }.get(method.lower(), "--method-maximize")

    # thresholds_map = {k: thresholds_cfg[k] for k in thresholds_cfg.keys()}
    # if "DEFAULT" not in thresholds_map:
    #     thresholds_map["DEFAULT"] = thresholds_cfg.DEFAULT
        
    thresholds_map = dict(cfg["dnastatistx"]["thresholds"])
    
    if "DEFAULT" not in thresholds_map:
        raise KeyError(
            "config.yaml must define dnastatistx.thresholds.DEFAULT"
        )

    return {
        "java_executable": "java",
        "jar_file": str(jar_file),
        "action": "calculate",
        "reference_profile": str(REFERENCE_FILE),

        "method": method,
        "method_flag": method_flag,

        "model": model,
        "model_flag": f"--{model}",

        "H1": {"contributors": int(cfg['dnastatistx']['N_CONTRIBUTORS']), "cond_knowns": COND_KNOWNS},
        "H2": {"contributors": int(cfg['dnastatistx']['N_CONTRIBUTORS'])},

        "kit": kit,
        "frequencies_file": str(freqs_file),
        "threads": threads,
        "coancestry": str(cfg['dnastatistx']["coancestry"]),

        "thresholds": thresholds_map,
        "frac_threshold": str(cfg['dnastatistx']["frac_threshold"]),
        "dropin_prob" : str(cfg['dnastatistx']["dropin_prob"]),
        "dropin_lambda" : str(cfg['dnastatistx']["dropin_lambda"]),
        "random_seed" : str(cfg['dnastatistx']["random_seed"]),
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

    print(f"✓ DNAStatistX completed, output → {output_dir}")
    return True





# ----------------------------------------------------------------------
# Run-level parameter bundle (both steps)
# ----------------------------------------------------------------------

def build_pipeline_runlevel_params(command_spec: dict, example_cmd: list[str] | None) -> dict:
    jar_path = Path(command_spec["jar_file"])
    freq_path = Path(command_spec["frequencies_file"])
    ref_path = Path(command_spec["reference_profile"])

    params = {


        # step 1: DNAStatistX command spec
        "DNASX_COMMAND_SPEC": deepcopy(command_spec),

        # step 2: genotype priors settings
        "PRIORS": {
         #   "FREQ_CSV_FOR_PRIORS": str(FREQ_CSV_FOR_PRIORS),
         #   "EPS": EPS,
         #   "RARE_FREQ": RARE_FREQ,
            "expected_input_filename": "results_marginal_deconvolutions.csv",
            "output_filename": "results_marginal_deconvolutions_with_prior_and_LR.csv",
        },

        # integrity hashes
        "HASHES": {
        #    "config_yaml_sha256": file_sha256(CONFIG_YAML),
            "reference_file_sha256": file_sha256(ref_path),
            "jar_sha256": file_sha256(jar_path),
            "dnax_frequencies_sha256": file_sha256(freq_path),
            #"priors_frequency_csv_sha256": file_sha256(FREQ_CSV_FOR_PRIORS),
        },
    }

    if example_cmd is not None:
        params["EXAMPLE_DNASX_CMD_LIST"] = example_cmd
        params["EXAMPLE_DNASX_CMD_STRING"] = " ".join(example_cmd)

    return params

# ----------------------------------------------------------------------
# Pipeline function (callable)
# ----------------------------------------------------------------------

def dnax_three_mixtures_combined(
    mixture_folder,
    RESOURCES_PATH,
    CONFIG_YAML,
    MIXTURES_DIR,
    REFERENCE_FILE,
    COND_KNOWNS
):
    
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    RARE_FREQ = float(cfg['RARE_FREQ'])
    H2_FILTER_THRESHOLD = float(cfg['H2_FILTER_THRESHOLD'])
    DELETE_INTERMEDIATE_FILES = cfg['DELETE_INTERMEDIATE_FILES']
    freq_csv_for_priors = RESOURCES_PATH / cfg['dnastatistx']["frequencies_file"]

    mixture_folder = Path(mixture_folder)
    mixture_list = [item.name for item in mixture_folder.iterdir() if item.is_dir()]

    for mixture in mixture_list:

        INPUT_ROOT = mixture_folder / mixture / "sampled_joint_genotypes"

        sample_list = [p.name for p in INPUT_ROOT.iterdir() if p.is_dir()]
        #print(sample_list)

        for sample in sample_list:

            OUTPUT_ROOT = INPUT_ROOT / sample
            OUTPUT_MIXTURES_DIR = OUTPUT_ROOT / "mixtures"
            
            #\combined_1_6B3_sample_0052\results_marginal_prior_LR.csv"
            RESULTS_MARGINAL_CSV = OUTPUT_MIXTURES_DIR / f"combined_dnax_{mixture}_{sample}" / "results_marginal_prior_LR.csv"
            print(RESULTS_MARGINAL_CSV)
            if RESULTS_MARGINAL_CSV.exists():
                print("Skipping: ", OUTPUT_ROOT)
                continue

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            OUTPUT_MIXTURES_DIR.mkdir(parents=True, exist_ok=True)

            # ---- build command spec ----
            command_spec = build_command_spec(RESOURCES_PATH, CONFIG_YAML, REFERENCE_FILE,COND_KNOWNS) #build_command_spec()

            # ---- run-level logging ----
            runlevel_params = build_pipeline_runlevel_params(command_spec, None)

            write_params_txt(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.txt")
            write_params_json(OUTPUT_ROOT, runlevel_params, "run_parameters_PIPELINE.json")

            n_ok = 0
            n_fail = 0

            trace_id = f"combined_dnax_{mixture}_{sample}"
            print(f"\n=== Running TRACE {trace_id} ===")

            sim_paths = list(OUTPUT_ROOT.glob("*Trace*.txt"))

            if len(sim_paths) < 2:
                print("[SKIP] Not enough simulated traces")
                continue

            #path1_org = SCRIPT_PATH / f"inputs/input_HT_2p_nodropin/mixtures/{mixture}.txt"
            path1_org = MIXTURES_DIR / f"{mixture}.txt"
            path2_sim = sim_paths[0]
            path3_sim = sim_paths[1]

            trace_files = [path1_org, path2_sim, path3_sim]

            output_dir = OUTPUT_MIXTURES_DIR / trace_id

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


            pp_ok, pp_bad = postprocess_add_priors_helper(
                output_dir,  #OUTPUT_MIXTURES_DIR,
                freq_csv_for_priors,
                RARE_FREQ, 
                H2_FILTER_THRESHOLD,
                DELETE_INTERMEDIATE_FILES
            )

            print("\n=== Priors post-processing completed ===")
            print(f"Success: {pp_ok}, Skipped/Failed: {pp_bad}")

            print("\nPipeline done.")



def main():
    mixture_folder = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_simulations\mixtures")
    RESOURCES_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources")
    #SCRIPT_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
    MIXTURES_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\mixtures_debug_2p")
    REFERENCE_FILE = RESOURCES_PATH / "reference_file.txt" #ADJUST WHEN USING wiskundig correcte LR
    COND_KNOWNS =  "ABCD1234NL#01" #ADJUST WHEN USING wiskundig correcte LR
    
    CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    
    dnax_three_mixtures_combined(
        mixture_folder=mixture_folder,
        MIXTURES_DIR = MIXTURES_DIR,
        RESOURCES_PATH = RESOURCES_PATH,
        CONFIG_YAML = CONFIG_YAML,
        REFERENCE_FILE=REFERENCE_FILE,
        COND_KNOWNS=COND_KNOWNS,
    )


if __name__ == "__main__":
    main()

