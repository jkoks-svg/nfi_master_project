"""Central paths and shared settings for the four-step analysis pipeline.

Keep this file in the same folder as:
    a_remove_dropins.py
    b_dnastatistx_single_mixtures.py
    c_dnastatistx_rework_mixtures.py
    a_create_deconv_tables_draws.py

When the project is moved, the main folder to update is DOCUMENTS_DIR.
"""

from pathlib import Path


# ----------------------------------------------------------------------
# Main folders
# ----------------------------------------------------------------------
INPUT_DOCUMENTS_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Github")
RAW_DATASET_DIR = INPUT_DOCUMENTS_DIR / "dataset_example"
RAW_MIXTURE_DIR = RAW_DATASET_DIR / "HT_2p"
DONOR_DIR = RAW_DATASET_DIR / "Donoren"
RESOURCES_PATH = INPUT_DOCUMENTS_DIR / "resources"

OUTPUT_DOCUMENTS_DIR = Path(r"C:\Users\jortk\Documents")
FINAL_DIR = OUTPUT_DOCUMENTS_DIR / "github"
ANALYSIS_DIR = FINAL_DIR / "input_HT_2p_nodropin"


# ----------------------------------------------------------------------
# Step 1: remove drop-ins
# ----------------------------------------------------------------------

DROPIN_OUTPUT_MIXTURES_DIR = ANALYSIS_DIR / "mixtures"
DROPIN_OUTPUT_LOG_DIR = ANALYSIS_DIR / "mixtures_log"


# ----------------------------------------------------------------------
# Steps 2-4: mixture input
# ----------------------------------------------------------------------


ANALYSIS_MIXTURES_DIR = DROPIN_OUTPUT_MIXTURES_DIR
#ANALYSIS_MIXTURES_DIR = ANALYSIS_DIR / "mixtures_debug_2p"


# ----------------------------------------------------------------------
# DNAStatistX resources and shared hypothesis setting
# ----------------------------------------------------------------------

REFERENCE_FILE = RESOURCES_PATH / "reference_file.txt"
COND_KNOWNS = "ABCD1234NL#01"

DNAX_CONFIG_YAML = RESOURCES_PATH / "config.yaml" #"_2p_Bayes_test_no_delete.yaml"
MCMC_CONFIG_YAML = RESOURCES_PATH / "config.yaml" #_2p_Bayes_test_yes_delete.yaml"


# ----------------------------------------------------------------------
# Output folders
# ----------------------------------------------------------------------

DNAX_SINGLE_OUTPUT_ROOT = ANALYSIS_DIR / "output_2p_dnax_raw"
DNAX_REWORK_OUTPUT_ROOT = ANALYSIS_DIR / "output_2p_dnax_raw_combined"

MCMC_SINGLE_OUTPUT_ROOT = ANALYSIS_DIR / "output_2p_mcmc"
MCMC_REWORK_OUTPUT_ROOT = ANALYSIS_DIR / "output_2p_mcmc_combined"


# ----------------------------------------------------------------------
# DNAStatistX result-file paths used by d_create_deconv_tables_draws.py
# ----------------------------------------------------------------------

def _result_file(output_root: Path, mixture: str, filename: str) -> Path:
    return output_root / "mixtures" / mixture / filename


def single_params_json(mixture: str) -> Path:
    return _result_file(DNAX_SINGLE_OUTPUT_ROOT, mixture, "results.json")


def combined_params_json(rework_mixture: str) -> Path:
    return _result_file(DNAX_REWORK_OUTPUT_ROOT, rework_mixture, "results.json")


def reference_csv(mixture: str) -> Path:
    return _result_file(
        DNAX_SINGLE_OUTPUT_ROOT,
        mixture,
        "results_joint_deconvolution_H2.csv",
    )


def combined_reference_csv(rework_mixture: str) -> Path:
    return _result_file(
        DNAX_REWORK_OUTPUT_ROOT,
        rework_mixture,
        "results_joint_deconvolution_H2.csv",
    )


def single_candidate_csv(mixture: str) -> Path:
    return _result_file(
        DNAX_SINGLE_OUTPUT_ROOT,
        mixture,
        "results_joint_deconvolution_H2_clean.csv",
    )


def rework_candidate_csv(rework_mixture: str) -> Path:
    return _result_file(
        DNAX_REWORK_OUTPUT_ROOT,
        rework_mixture,
        "results_joint_deconvolution_H2_clean.csv",
    )
