# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:08:13 2026

@author: jortk
"""

from pathlib import Path
import time
import yaml

from dnax_pipeline_20260307_1130 import dnax_single_mixtures
from sample_from_joint_20260304_1350 import run_joint_sampling
#from dnax_pipeline_generate_20260307_1232 import generate_mixtures
from dnax_pipeline_generate_new import generate_mixtures
from dnax_pipeline_traces_replicates_20260307_1440 import dnax_three_mixtures_combined
from mle_pipeline_traces_replicates import mle_three_mixtures_combined
from mcmc_pipeline_traces_replicates_candidate_dnax import mcmc_three_mixtures_combined
from read_csv_traces_replicates_20260324_1800 import create_filtered_dfs
from read_csv_traces_20260318_1315 import compare_with_lab_combined_profiles
from wasserstein import analyse_wasserstein_column

start = time.perf_counter()

SCRIPT_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
RESOURCES_PATH = SCRIPT_PATH / "resources"
REFERENCE_FILE = RESOURCES_PATH / "reference_file.txt" #ADJUST WHEN USING wiskundig correcte LR
COND_KNOWNS =  "ABCD1234NL#01" #ADJUST WHEN USING wiskundig correcte LR
N_SAMPLES = 100


N_CONTRIBUTORS_RUN = 2
if N_CONTRIBUTORS_RUN == 2:

    MIXTURES_DIR = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\mixtures_2p")
    OUTPUT_ROOT = Path(r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_runF_Qallele_fixed_100samples")  
    #r"C:\Users\jortk\Documents\input_HT_2p_nodropin\output_runF_test3"
    #r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_runF_Qallele_fixed_100samples") 
    #r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_1B2_v7")
    #r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p"
    #r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\mixtures_2p"
    CONFIG_YAML = RESOURCES_PATH / "config_2p_Bayes_test.yaml"
    traces_folder_root =  Path(r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces") 
    # r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_2p"
    # r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces"
    #r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_1B2_v7")
    MIXTURE_TYPES = ["A", "B","C", "D", 'E']
    
# if N_CONTRIBUTORS_RUN == 3:
#     MIXTURES_DIR = Path(r"C:\Users\jortk\Documents\inputs\input_HT_3p_nodropin\mixtures_3p_BDE")
#     OUTPUT_ROOT = Path(r"C:\Users\jortk\Documents\input_HT_3p_nodropin\output_runF_BDE")
#     CONFIG_YAML = RESOURCES_PATH / "config_3p_new_dnax.yaml"
#     traces_folder_root =  Path(r"C:\Users\jortk\Documents\inputs\input_HT_3p_nodropin\output_runF_3p_BDE_traces")
#     MIXTURE_TYPES = ["B","D", 'E']

#deleted dnax_pipeline part
# You Should already have created output files for single and rework mixtures

# DEBUG = False
# run_joint_sampling(
#     mixture_folder=OUTPUT_ROOT / "mixtures",
#     RESOURCES_PATH = RESOURCES_PATH,
#     N_samples=N_SAMPLES,
#     DEBUG = DEBUG
# )


# N_REPLICATES_PER_SAMPLE = 2 #DO NOT CHANGE
# test = 'test4'
# MCMC_MIXTURE_FOLDER = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p")

# generate_mixtures(
#     mixture_folder=OUTPUT_ROOT / "mixtures",
#     traces_folder='placeholder_folder', #traces_folder,
#     RESOURCES_PATH=RESOURCES_PATH,
#     CONFIG_YAML=CONFIG_YAML,
#     N_REPLICATES_PER_SAMPLE=N_REPLICATES_PER_SAMPLE,
#     test_mode=test,
#     mcmc_mixture_folder=MCMC_MIXTURE_FOLDER / 'mixtures',
# )


# dnax_three_mixtures_combined(
#     mixture_folder=OUTPUT_ROOT / "mixtures",
#     #SCRIPT_PATH = SCRIPT_PATH,
#     MIXTURES_DIR = MIXTURES_DIR,
#     RESOURCES_PATH = RESOURCES_PATH,
#     CONFIG_YAML=CONFIG_YAML,
#     REFERENCE_FILE=REFERENCE_FILE,
#     COND_KNOWNS=COND_KNOWNS,
# )

# DEFAULT_CREATE_DECONV_TABLES_BOTH_PY = Path(
#     r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\python scripts\Bayes model\create_deconv_tables_both.py"
# )

# mle_three_mixtures_combined(
#     mixture_folder=OUTPUT_ROOT / "mixtures",          # new MLE output location
#     MIXTURES_DIR=MIXTURES_DIR,
#     RESOURCES_PATH=RESOURCES_PATH,
#     CONFIG_YAML=CONFIG_YAML,
#     REFERENCE_FILE=REFERENCE_FILE,
#     COND_KNOWNS=COND_KNOWNS,

#     mle_config_dict={
#         # Candidate-pair filtering from the old DNAStatistX pipeline output.
#         "use_candidate_pairs": False,
#         "require_candidate_csv": False,
#         "candidate_min_probability": 0.0,
#         "candidate_max_rows_per_locus": None,
#         "candidate_add_contributor_swaps": False,
#     },

#     overwrite=False,

#     # Script with the shared deconvolution functions
#     create_deconv_tables_both_py=DEFAULT_CREATE_DECONV_TABLES_BOTH_PY,

#     # Old DNAStatistX output location used for candidate CSVs
#     candidate_mixture_folder=None, #DNASTATISTX_OUTPUT_ROOT / "mixtures", #fix later

#     # Location where the MLE results.json files are found.
#     # In your case this is probably the same DNAStatistX output folder.
#     mle_results_folder=OUTPUT_ROOT / "mixtures",
# )

# DNASTATISTX_OUTPUT_ROOT = OUTPUT_ROOT #Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_B2")

# mcmc_three_mixtures_combined(
#     mixture_folder=OUTPUT_ROOT / "mixtures",          # new MCMC output location
#     MIXTURES_DIR=MIXTURES_DIR,
#     RESOURCES_PATH=RESOURCES_PATH,
#     CONFIG_YAML=CONFIG_YAML,
#     REFERENCE_FILE=REFERENCE_FILE,
#     COND_KNOWNS=COND_KNOWNS,
#     mcmc_config_dict={
#         "n_iter": 2000,
#         "burnin": 500,
#         "thin": 10,
#         "proposal_scales": (0.12, 0.10, 0.10, 0.10),
#         "seed": 12345,
#         "max_posterior_draws": 100,
#         # These replace the DNAStatistX results.json starting values.
#         "init_mu": 10000.0,
#         "init_sigma": 0.20,
#         "init_phi1": 0.51,
#         "init_beta": 0.90,
#         # Prior bounds/shape parameters as used by your MCMC code.
#         "mu_1": 40000,
#         "sigma_1": 1,
#         # Candidate-pair filtering from the old DNAStatistX pipeline output.
#         "require_candidate_csv": True,
#         "candidate_min_probability": 0.0,
#         "candidate_max_rows_per_locus": None,
#         "candidate_add_contributor_swaps": False,
#     },
#     overwrite=False,
#     # old DNAStatistX output location used for candidate CSVs
#     create_deconv_tables_both_py=DEFAULT_CREATE_DECONV_TABLES_BOTH_PY,
#     candidate_mixture_folder=DNASTATISTX_OUTPUT_ROOT / "mixtures", #put candidate_pairs_by_locus to none
# )

# SAVE_FILES = True
# version = 'v1'

# create_filtered_dfs(mixture_folder = OUTPUT_ROOT / "mixtures", 
#               results_base_dir = OUTPUT_ROOT, 
#               version = version, 
#               save_files= SAVE_FILES,
#               CONFIG_YAML=CONFIG_YAML)

SAVE_FILES_2 = True
version_2 = 'v1'

DATASETS = [1,2,3,4,5,6]


percentile_df = compare_with_lab_combined_profiles(
    FILTERED_DF_ROOT = OUTPUT_ROOT, #/ "mixtures",
    OUT_ROOT=traces_folder_root,
    CONFIG_YAML = CONFIG_YAML,
    SAVE_FILES=SAVE_FILES_2,
    version = version_2,
    N_GLOBAL=N_SAMPLES,
    MIXTURE_TYPES = MIXTURE_TYPES,
    DATASETS=DATASETS
)

# wasserstein_summary_df = analyse_wasserstein_column(
#     percentile_df["signed_wasserstein_log10_LR"],
#     output_dir=None,
#     filename=None,
#     bins=30,
# )

runtime = time.perf_counter() - start
print(f"\nTotal runtime: {runtime/60:.2f} minutes")



# generate_mixtures(
#     mixture_folder= OUTPUT_ROOT / "mixtures", #mixture_folder,
#     traces_folder = "placeholder_folder", #traces_folder_root / "mixtures",
#     RESOURCES_PATH = RESOURCES_PATH,
#     CONFIG_YAML = CONFIG_YAML,
#     N_REPLICATES_PER_SAMPLE = N_REPLICATES_PER_SAMPLE,
#     test_mode=test
# )
