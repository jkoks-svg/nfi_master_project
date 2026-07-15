# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:08:13 2026

@author: jortk
"""

from pathlib import Path
import time
import yaml

from b_sample_from_joint_deconv import run_joint_sampling
from c_generate_rework_mixtures import generate_mixtures
from d_dnastatistx_rework_mixtures import dnax_three_mixtures_combined
from e_single_draw_rework_mixtures import mle_three_mixtures_combined
from f_extract_rework_LRs import create_filtered_dfs
from g_plot_rework_LRs import compare_with_lab_combined_profiles

start = time.perf_counter()

INPUT_DOCUMENTS_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Github")
RESOURCES_PATH = INPUT_DOCUMENTS_DIR / "resources"
REFERENCE_FILE = RESOURCES_PATH / "reference_file.txt" #ADJUST WHEN USING wiskundig correcte LR
COND_KNOWNS =  "ABCD1234NL#01" #ADJUST WHEN USING wiskundig correcte LR
N_SAMPLES = 2

OUTPUT_DOCUMENTS_DIR = Path(r"C:\Users\jortk\Documents")
FINAL_DIR = OUTPUT_DOCUMENTS_DIR / "github"
ANALYSIS_DIR = FINAL_DIR / "input_HT_2p_nodropin"

N_CONTRIBUTORS_RUN = 2
if N_CONTRIBUTORS_RUN == 2:
    CONFIG_YAML = RESOURCES_PATH / "config.yaml"
    MIXTURES_DIR = ANALYSIS_DIR / "mixtures" 
    OUTPUT_ROOT = ANALYSIS_DIR / "output_2p_mcmc" 
    traces_folder_root =  ANALYSIS_DIR / "output_2p_mcmc_combined" 
    MIXTURE_TYPES = ["A", "B"]#,"C", "D", 'E']
    

#deleted dnax_pipeline part
# You Should already have created output files for single and rework mixtures

DEBUG = False
run_joint_sampling(
    mixture_folder=OUTPUT_ROOT / "mixtures",
    RESOURCES_PATH = RESOURCES_PATH,
    N_samples=N_SAMPLES,
    DEBUG = DEBUG
)


N_REPLICATES_PER_SAMPLE = 2 #DO NOT CHANGE
test = 'test0'
#MCMC_MIXTURE_FOLDER = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p")

generate_mixtures(
    mixture_folder=OUTPUT_ROOT / "mixtures",
    traces_folder='placeholder_folder', #traces_folder,
    RESOURCES_PATH=RESOURCES_PATH,
    CONFIG_YAML=CONFIG_YAML,
    N_REPLICATES_PER_SAMPLE=N_REPLICATES_PER_SAMPLE,
    test_mode=test,
    mcmc_mixture_folder=OUTPUT_ROOT / 'mixtures' #MCMC_MIXTURE_FOLDER / 'mixtures',
)


dnax_three_mixtures_combined(
    mixture_folder=OUTPUT_ROOT / "mixtures",
    #SCRIPT_PATH = SCRIPT_PATH,
    MIXTURES_DIR = MIXTURES_DIR,
    RESOURCES_PATH = RESOURCES_PATH,
    CONFIG_YAML=CONFIG_YAML,
    REFERENCE_FILE=REFERENCE_FILE,
    COND_KNOWNS=COND_KNOWNS,
)


mle_three_mixtures_combined(
    mixture_folder=OUTPUT_ROOT / "mixtures",          # new MLE output location
    MIXTURES_DIR=MIXTURES_DIR,
    RESOURCES_PATH=RESOURCES_PATH,
    CONFIG_YAML=CONFIG_YAML,
    REFERENCE_FILE=REFERENCE_FILE,
    COND_KNOWNS=COND_KNOWNS,

    mle_config_dict={
        # Candidate-pair filtering from the old DNAStatistX pipeline output.
        "use_candidate_pairs": False,
        "require_candidate_csv": False,
        "candidate_min_probability": 0.0,
        "candidate_max_rows_per_locus": None,
        "candidate_add_contributor_swaps": False,
        "copy_dnax_results_json": True,
    },

    overwrite=True,

    # Script with the shared deconvolution functions
    #create_deconv_tables_both_py='placeholder',
    create_deconv_tables_both_py=Path(__file__).with_name(
        "create_deconv_tables_scalar.py"
    ),

    # Old DNAStatistX output location used for candidate CSVs
    candidate_mixture_folder=None, #DNASTATISTX_OUTPUT_ROOT / "mixtures", #fix later

    # Location where the MLE results.json files are found.
    # In your case this is probably the same DNAStatistX output folder.
    mle_results_folder=OUTPUT_ROOT / "mixtures",
)



SAVE_FILES = True
version = 'v1'
mle_or_dnax = 'mle'

create_filtered_dfs(mixture_folder = OUTPUT_ROOT / "mixtures", 
              results_base_dir = OUTPUT_ROOT, 
              version = version, 
              save_files= SAVE_FILES,
              CONFIG_YAML=CONFIG_YAML,
              mle_or_dnax=mle_or_dnax)

SAVE_FILES_2 = False
version_2 = 'v1'
mle_or_dnax = 'mle'

DATASETS = [1]#,2,3,4,5,6]


percentile_df = compare_with_lab_combined_profiles(
    FILTERED_DF_ROOT = OUTPUT_ROOT, #/ "mixtures",
    OUT_ROOT=traces_folder_root,
    CONFIG_YAML = CONFIG_YAML,
    SAVE_FILES=SAVE_FILES_2,
    version = version_2,
    N_GLOBAL=N_SAMPLES,
    MIXTURE_TYPES = MIXTURE_TYPES,
    DATASETS=DATASETS,
    mle_or_dnax = mle_or_dnax
)


runtime = time.perf_counter() - start
print(f"\nTotal runtime: {runtime/60:.2f} minutes")



