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
from dnax_pipeline_generate_20260307_1232 import generate_mixtures
from dnax_pipeline_traces_replicates_20260307_1440 import dnax_three_mixtures_combined
from read_csv_traces_replicates_20260324_1800 import create_filtered_dfs
from read_csv_traces_20260318_1315 import compare_with_lab_combined_profiles

start = time.perf_counter()

SCRIPT_PATH = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")
RESOURCES_PATH = SCRIPT_PATH / "resources"
REFERENCE_FILE = RESOURCES_PATH / "reference_file.txt" #ADJUST WHEN USING wiskundig correcte LR
COND_KNOWNS =  "ABCD1234NL#01" #ADJUST WHEN USING wiskundig correcte LR
N_SAMPLES = 10


N_CONTRIBUTORS_RUN = 2
if N_CONTRIBUTORS_RUN == 2:

    #MIXTURES_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\mixtures")
    MIXTURES_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\mixtures_debug_2p")

    OUTPUT_ROOT = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\output_simulations_debug_2p")
    CONFIG_YAML = RESOURCES_PATH / "config_2p_new_dnax.yaml"
    #traces_folder_root =  Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces")
    traces_folder_root =  Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces")
    MIXTURE_TYPES = ["A", "B","C", "D", 'E']
    
if N_CONTRIBUTORS_RUN == 3:
    MIXTURES_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_3p_nodropin\mixtures_3p_BDE")
    OUTPUT_ROOT = Path(r"C:\Users\jortk\Documents\input_HT_3p_nodropin\output_runF_BDE")
    CONFIG_YAML = RESOURCES_PATH / "config_3p_new_dnax.yaml"
    traces_folder_root =  Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_3p_nodropin\output_runF_3p_BDE_traces")
    MIXTURE_TYPES = ["B","D", 'E']



# dnax_single_mixtures(
#     mixtures_dir=MIXTURES_DIR,
#     output_root=OUTPUT_ROOT,
#     RESOURCES_PATH=RESOURCES_PATH,
#     CONFIG_YAML = CONFIG_YAML,
#     REFERENCE_FILE = REFERENCE_FILE,
#     COND_KNOWNS=COND_KNOWNS,
# )

#add combined profiles creation

# DEBUG = False


# run_joint_sampling(
#     mixture_folder=OUTPUT_ROOT / "mixtures",
#     RESOURCES_PATH = RESOURCES_PATH,
#     N_samples=N_SAMPLES,
#     DEBUG = DEBUG
# )


# N_REPLICATES_PER_SAMPLE = 2 #DO NOT CHANGE
# test = 'test3'

# generate_mixtures(
#     mixture_folder= OUTPUT_ROOT / "mixtures", #mixture_folder,
#     traces_folder = "placeholder_folder", #traces_folder_root / "mixtures",
#     RESOURCES_PATH = RESOURCES_PATH,
#     CONFIG_YAML = CONFIG_YAML,
#     N_REPLICATES_PER_SAMPLE = N_REPLICATES_PER_SAMPLE,
#     test_mode=test
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


# SAVE_FILES = True
# version = 'v1'

# create_filtered_dfs(mixture_folder = OUTPUT_ROOT / "mixtures", 
#               results_base_dir = OUTPUT_ROOT, 
#               version = version, 
#               save_files= SAVE_FILES,
#               CONFIG_YAML=CONFIG_YAML)

SAVE_FILES_2 = False
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

runtime = time.perf_counter() - start
print(f"\nTotal runtime: {runtime/60:.2f} minutes")

