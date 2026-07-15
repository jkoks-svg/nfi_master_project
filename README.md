# Statistical analysis of replicate measurements of DNA mixtures

This repository contains the Python code used for the MSc thesis **Statistical analysis of replicate measurements of DNA mixtures** by Jort Koks (Delft University of Technology and the Netherlands Forensic Institute, 2026).

The research investigates whether the likelihood ratio (LR) after laboratory rework can be predicted from an original DNA mixture profile. The code covers DNAStatistX-based deconvolution, a Bayesian MCMC implementation of the EuroForMix/DNAStatistX peak-height model, simulation of rework profiles, and validation and diagnostic analyses.

This is research code and a mathematical proof of concept. It is not an operational NFI tool and must not be used for forensic casework without further validation.

## Research documents

- [MSc thesis](MSc_Thesis_Jort_Koks.pdf) - full background, methods, results, limitations, and conclusions.
- [Thesis presentation](Presentatie%202026-06-24%20final.pdf) - presentation defended on 24 June 2026.
- [PPF6C dataset and mixture-generation overview](2p-5p%20PPF6C%20dataset_2024_Mixture%20generation.docx) - mixture types, donor combinations, replicate structure, and filename convention for the full research dataset.

## Repository structure

```text
dataset_example/
    Donoren/                     Example donor profiles
    HT_2p/                       Example two-person mixture profiles

python_scripts/
    1_mixture2deconvolution/     Cleaning, DNAStatistX deconvolution,
                                 and MCMC analysis
    2_simulation_algorithm/      Rework simulation and LR extraction
    3_validation_and_diagnostics/
                                 Validation and additional analyses
    helpers/                     Shared model, genotype, MCMC, and
                                 post-processing functions

resources/
    config.yaml                  DNAStatistX and model configuration
    NFI_frequencies.csv          Allele frequencies
    kit_properties_txt.txt       PPF6C kit properties
    reference_file.txt           Reference profile

MSc_Thesis_Jort_Koks.pdf
Presentatie 2026-06-24 final.pdf
2p-5p PPF6C dataset_2024_Mixture generation.docx
README.md
```

The included `dataset_example` is a small two-person example containing dataset 1, mixture types A and B, and three replicates. It is not the complete research dataset.

## NFI-owned materials not distributed through GitHub

### DNAStatistX

This research used **DNAStatistX CLI 2.5.0-DES-700, build 202601191653**. The code and supplied configuration expect the following filename:

```text
dnastatistx-cli-2.5.0-DES-700-202601191653.jar
```

The DNAStatistX JAR is owned by the Netherlands Forensic Institute (NFI). Access can be requested from the NFI.

After obtaining the JAR, place it in `resources/` under the filename above. If the supplied filename is different, update `dnastatistx.jar_file` in `resources/config.yaml`.

### Full research dataset

The complete 2p-5p PPF6C research dataset is also owned by the NFI and is not included in this repository. Access to the full dataset can be requested from the NFI and remains subject to the NFI's conditions.

The small profiles under `dataset_example/`, `NFI_frequencies.csv`, and the example donor profiles are included with this research repository.

## Requirements

- Python 3.10 or newer.
- Java available through the `java` command.
- The Python packages used by the scripts:

```powershell
python -m pip install numpy pandas scipy matplotlib seaborn statsmodels pyyaml
```

Check that Python and Java are available:

```powershell
python --version
java -version
```

The code was organized around Windows paths, although most file operations use `pathlib`.

## Input naming convention

Mixture profiles use:

```text
<replicate>_<dataset><mixture type><number of contributors>.txt
```

For example, `1_3A2.txt` is replicate 1 of dataset 3, mixture type A, with two contributors. The included `1_1A2.txt` is replicate 1 of dataset 1, mixture type A, with two contributors.

Donor files are CSV files named by dataset and donor, for example `1A.csv` and `1B.csv`. See the dataset-generation overview for the full naming convention and donor combinations.

## 1. Configure the mixture-to-deconvolution pipeline

Open:

```text
python_scripts/1_mixture2deconvolution/pipeline_config.py
```

At minimum, change:

```python
INPUT_DOCUMENTS_DIR = Path(r"C:\path\to\Github")
OUTPUT_DOCUMENTS_DIR = Path(r"C:\path\to\output_folder")
```

`INPUT_DOCUMENTS_DIR` must point to the repository root containing `dataset_example`, `python_scripts`, and `resources`.

With the supplied configuration, output is written below:

```text
<OUTPUT_DOCUMENTS_DIR>/github/input_HT_2p_nodropin/
```

Also check:

- `COND_KNOWNS` and `REFERENCE_FILE` for the intended reference profile;
- `ANALYSIS_MIXTURES_DIR` for the cleaned mixture input; and
- the four DNAStatistX and MCMC output directories.

Model, kit, threshold, frequency, degradation, and DNAStatistX settings are stored in `resources/config.yaml`.

## 2. Run mixture cleaning and deconvolution

Run these scripts in alphabetical order. Start in their directory so the local imports resolve correctly:

```powershell
cd python_scripts\1_mixture2deconvolution

python a_remove_dropins.py
python b_dnastatistx_single_mixtures.py
python c_dnastatistx_rework_mixtures.py
python d_create_deconv_tables_draws.py
```

The scripts perform the following steps:

1. `a_remove_dropins.py` removes alleles not explained by the known research donors and writes cleaned profiles and removal logs.
2. `b_dnastatistx_single_mixtures.py` runs DNAStatistX for each single replicate and post-processes the joint deconvolution output.
3. `c_dnastatistx_rework_mixtures.py` runs DNAStatistX on the combined three-replicate profiles and post-processes the output.
4. `d_create_deconv_tables_draws.py` performs the MLE or MCMC calculations and writes single- and combined-profile deconvolution results.

For the included example data, use the following settings near the top of `a_remove_dropins.py`:

```python
MIXTURE_TYPES = ["A", "B"]
DATASETS = [1]
REPLICATES = [1, 2, 3]
N_CONTRIBUTORS_LIST = [2]
```

The default full-dataset loop skips files that are absent, but restricting these settings makes an example run easier to follow.

Before running `d_create_deconv_tables_draws.py`, check:

```python
parameter_method = "MCMC"
run_single_analysis = True
run_rework_analysis = True
```

Its MCMC iteration count, burn-in, thinning, proposal scales, seed, and retained posterior draws are defined in `mcmc_config_dict` near the bottom of the script.

## Shared helpers

Functions used by both main workflows are stored once under `python_scripts/helpers/`:

- `euroformix_gamma_model.py` implements the peak-height likelihood;
- `genotypes.py` constructs genotypes and genotype priors;
- `mcmc_parameters.py` implements MCMC parameter estimation; and
- `postprocess_add_priors.py` post-processes DNAStatistX deconvolution results.

The analysis scripts add `python_scripts` to the import path and import these modules through `helpers`. Keep the helper folder in its current location.

## 3. Run the rework simulation algorithm

The simulation driver is:

```text
python_scripts/2_simulation_algorithm/a_main_simulation_algorithm.py
```

It runs the following sequence:

1. sample plausible contributor genotypes from the original-profile deconvolution;
2. generate simulated rework replicates;
3. analyse the combined simulated profiles with DNAStatistX;
4. calculate LR values for the simulated profiles;
5. extract the simulated rework LRs; and
6. compare the predicted distributions with the observed laboratory rework LRs.

Edit the path and run-settings block at the top of `a_main_simulation_algorithm.py`. Check:

- `INPUT_DOCUMENTS_DIR`, `OUTPUT_DOCUMENTS_DIR`, `FINAL_DIR`, and `ANALYSIS_DIR`;
- `N_SAMPLES`;
- `N_CONTRIBUTORS_RUN`;
- `MIXTURE_TYPES` and `DATASETS`;
- `test`, which selects the generation mode; and
- `SAVE_FILES`, `version`, and `mle_or_dnax`.

The simulation expects the single-profile DNAStatistX/MCMC output produced by the previous workflow. The supplied `N_SAMPLES = 2` is a small test setting, not the final research run size.

Run the complete simulation from its directory:

```powershell
cd ..\2_simulation_algorithm
python a_main_simulation_algorithm.py
```

The generation modes in `c_generate_rework_mixtures.py` are:

| Mode | Parameters used to generate a simulated profile |
| --- | --- |
| `test0` | MLE parameters of the current single profile |
| `test1` | Current EPH/PHV/DEG and mixture proportions from a randomly selected replicate |
| `test2` | Current EPH/PHV/DEG and mixture proportions from the combined trace |
| `test3` | Current EPH/PHV/DEG and a sampled two-person mixture proportion |
| `test4` | EPH, PHV, DEG, and mixture proportions sampled from `results_mcmc.json` |

Increasing the number of simulations or MCMC iterations can make this workflow computationally expensive.

## 4. Run validation and diagnostics

The two main validation scripts under `python_scripts/3_validation_and_diagnostics/` are:

- `validate_single_LR_coverage.py` for single-profile Bayesian-draw validation; and
- `validate_rework_LR_coverage.py` for combined/rework-profile Bayesian-draw validation.

Both are configured by editing their settings near the top of the file. Check:

- `BASE`, `OUT_ROOT`, `RAW_MIX_DIR` where present, and `RESULTS_DIR`;
- `RUN_MODE`, `MIXTURE_TYPES`, `DATASETS`, and `REPLICATES` where present;
- `BAD_LOCI_MODE` (`"zero"`, `"impute"`, or `"both"`); and
- `SAVE_FILES` and `version`.

The rework validation script contains additional `filtered_df_path` templates inside its dataset loop. Update these if the simulation output uses a different directory layout.

Run the validation scripts from their directory:

```powershell
cd ..\3_validation_and_diagnostics

python validate_single_LR_coverage.py
python validate_rework_LR_coverage.py
```

The other scripts in this folder are stand-alone analyses used for particular thesis tables, plots, or diagnostics. They are not required in one fixed order and contain their own input/output path settings.

## Expected workflow

```text
Raw mixture and donor profiles
        |
        v
Remove drop-ins
        |
        v
DNAStatistX single and combined-profile analyses
        |
        v
MLE/MCMC deconvolution output
        |
        v
Sample genotypes and simulate rework profiles
        |
        v
Calculate and extract simulated rework LRs
        |
        v
Validation, calibration, and diagnostic analyses
```

Run-level parameter files, DNAStatistX standard output/error logs, intermediate CSV/JSON files, MCMC diagnostics, filtered LR tables, and plots are written below the configured output directories.

## Research limitations

The thesis found that including Bayesian parameter uncertainty improved prediction-interval calibration, but the predicted rework-LR distributions remained insufficiently calibrated for casework use. See the thesis for the complete assumptions, evaluation results, and limitations.

Reproducing a particular thesis result requires the corresponding NFI dataset and the matching experiment settings, random seeds, and output-directory choices described in the scripts and thesis.

## Citation

Koks, J. (2026). *Statistical analysis of replicate measurements of DNA mixtures*. MSc thesis, Delft University of Technology, in collaboration with the Netherlands Forensic Institute.
