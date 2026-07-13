# NFI code handover audit

**Project:** Statistical analysis of replicate measurements of DNA mixtures  
**Researcher:** Jort Koks  
**Audit date:** 13 July 2026  
**Scope:** `github.zip`, `resources.zip`, the dataset-generation document, the thesis, and example profile `1_1A2.txt`

## Executive summary

The archive contains 37 Python scripts with 32,165 lines of code. All scripts compile successfully, so the source archive is syntactically intact. The principal handover risks are reproducibility, duplicated implementations, hard-coded machine-specific configuration, missing dependencies, and the absence of automated tests.

The final NFI package should preserve the original scripts unchanged in a `legacy/` directory, while exposing a smaller documented package with reusable model code and a limited set of command-line workflows. Scientific refactoring should be done incrementally and checked against saved thesis outputs.

## Immediate findings

| Finding | Evidence | Consequence | Recommended action |
|---|---:|---|---|
| Personal absolute paths | 29 of 37 scripts contain Windows paths under `C:\Users\jortk\...` | Most scripts cannot run on another computer | Move all paths and run settings into YAML configuration and CLI arguments |
| No reproducible environment | No `pyproject.toml`, `requirements.txt`, lockfile, or environment file | Package versions used for the thesis are unknown | Recover the original Python version and `pip freeze`; create a locked environment |
| Missing configuration dependency | Ten scripts import `confidence.Configuration`, but `confidence` is not included and is not installed in the audit environment | DNAStatistX wrappers cannot be imported | Supply the exact package/version or replace it with an internal typed YAML loader |
| Missing module | `main_simulations.py` imports `dnax_pipeline_generate_20260307_1232`, which is absent | The frequentist driver fails immediately | Replace with the confirmed final generation module after regression testing |
| Duplicated implementations | Several script pairs are 80-98% identical; `mixture_proportions.py` and `mixture_proportions_Bayes.py` differ only in comments/blank lines | Fixes can be applied to one copy but not another | Consolidate into parameterized functions and archive superseded copies |
| Import-time execution | Many analysis scripts read files, plot, create folders, or run analyses at module level | Importing code can trigger unintended work | Put entry points behind `main()` and `if __name__ == "__main__":` |
| Mixed random-number handling | Global `np.random.seed`, `np.random.*`, `default_rng`, and unseeded `random.choice` are all used | Reproducing simulations may depend on execution order; generation is not fully deterministic | Use one explicitly passed `numpy.random.Generator`; record root and derived seeds |
| Broad exception handling | Multiple bare `except:` blocks and broad `except Exception` blocks | Failed loci/files may be silently skipped | Catch expected exceptions explicitly and log mixture, locus, file, and traceback |
| No tests | No test directory or test configuration | Scientific equivalence cannot be demonstrated after refactoring | Add unit, integration, and regression tests before consolidating algorithms |
| Ambiguous resources | Two current DNAStatistX 2.5.0 JAR builds and several old JARs are present | It is unclear which executable generated each result | Confirm the canonical JAR and record SHA-256 in the run manifest |
| Invalid/poorly named frequency file | `NFI_frequencies_fucked.csv` is a malformed one-column, doubly quoted version of the valid CSV | It can be selected accidentally and its filename is unsuitable for handover | Exclude it from runtime resources; retain only in a clearly labelled legacy archive if needed |

## Research workflow represented by the code

1. Prepare mixture and reference profiles.
2. Remove known drop-in peaks for the cleaned proof-of-concept dataset.
3. Run DNAStatistX on single profiles and add genotype priors.
4. Construct frequentist or Bayesian deconvolution tables.
5. Sample complete contributor genotypes.
6. Generate two additional simulated replicate profiles.
7. Analyse the three-profile combination using DNAStatistX, plug-in MLE code, or MCMC code.
8. Construct sampled-donor LR distributions and compare them with the laboratory rework result.
9. Calculate calibration percentiles, coverage, interval score, Wasserstein distance, mixture-proportion diagnostics, and thesis figures.

The dataset document confirms three replicates per sample and the filename convention represented by `1_1A2`: replicate 1, donor-combination dataset 1, mixture type A, two contributors. The full sample name additionally contains the AMEL-X height.

## Provisional inventory and disposition

The dispositions below are provisional until the original run environment and selected thesis outputs are available for regression testing.

### Model, deconvolution, and genotype sampling

| Current script | Role | Thesis area | Proposed disposition |
|---|---|---|---|
| `gamma_model_degr_on.py` | Gamma peak-height, degradation, drop-in likelihood | Chapters 3 and 9 | Retain as core; refactor to `src/nfi_rework/model/peak_height.py` |
| `genotypes.py` | Genotype generation, canonicalization, and priors | Chapters 3, 7, 9, 11 | Retain as core; refactor to `src/nfi_rework/model/genotypes.py` |
| `mcmc_parameters_degr_on.py` | Parameter transformations, priors, posterior, single/rework MCMC | Chapters 9-11 | Retain as core; split into `bayesian/parameters.py`, `priors.py`, and `sampler.py` |
| `create_deconv_tables_both.py` | Earlier scalar/averaged Bayesian deconvolution and diagnostics | Chapters 9-11 | Preserve for provenance; extract shared functions, then archive if regression confirms supersession |
| `create_deconv_tables_corrected2.py` | Draw-wise posterior deconvolution, marginal LR draws, and MCMC orchestration | Chapters 9-12 | Provisional final Bayesian implementation; split into small modules |
| `sample_from_joint_20260304_1350.py` | Scalar joint-genotype sampling | Chapters 5-8 and earlier Bayesian work | Retain frequentist scalar sampler if required; otherwise archive after tests |
| `sample_from_joint_corrected2.py` | Selects one posterior parameter draw and samples conditional genotypes across loci | Chapters 11-12 | Provisional final Bayesian sampler; retain as core |
| `postprocess_add_priors.py` | Adds raw/fixed genotype priors to DNAStatistX output | Chapters 3, 5-8 | Consolidate with genotype-prior utilities; keep one implementation |

### DNAStatistX and simulation pipelines

| Current script | Role | Proposed disposition |
|---|---|---|
| `dnax_pipeline_20260307_1130.py` | Single-profile DNAStatistX batch runner | Retain workflow; rename to `run_single_deconvolution.py` and remove hard-coded examples |
| `dnax_pipeline_generate_new.py` | Earlier simulated-profile generator | Archive after comparison with corrected implementation |
| `dnax_pipeline_generate_corrected2.py` | Generation with saved posterior-draw index and algorithm-choice modes | Provisional final generator; replace `test0`-`test4` with named configuration choices |
| `dnax_pipeline_traces_replicates_20260307_1440.py` | Analyses simulated replicate combinations with DNAStatistX | Retain workflow; share one DNAStatistX runner with the single-profile command |
| `dnax_pipeline_traces_20260322_1600.py` | Analyses laboratory replicate combinations | Retain distinct lab-rework workflow; share implementation with other DNAStatistX runners |
| `dnax_pipeline_traces.py` | Later monolithic trace runner with duplicated prior post-processing | Extract any newer behaviour, then archive the duplicate runner |
| `mle_pipeline_traces_replicates.py` | Plug-in MLE analysis of simulated rework | Retain as a separate analysis engine with a common interface |
| `mcmc_pipeline_traces_replicates_candidate_dnax.py` | MCMC analysis of simulated rework using candidate genotype pairs | Retain as a separate analysis engine with a common interface |
| `main_simulations.py` | Frequentist notebook-like driver | Replace with a command-line workflow; current version has a missing import |
| `main_simulations_Bayes.py` | Earlier Bayesian notebook-like driver | Archive after final-pipeline confirmation |
| `main_simulations_Bayes_corrected2.py` | Latest Bayesian notebook-like driver | Use as the provisional workflow specification, not as final production code |

### Input preparation

| Current script | Role | Proposed disposition |
|---|---|---|
| `csv2txt_reference_file.py` | Converts donor/reference CSV data to DNAStatistX text format | Retain as `prepare-reference`; parameterize paths and add validation |
| `remove_dropins_def3.py` | Removes peaks not explained by known donors in the cleaned research setting | Retain as an explicitly research-only preprocessing command; document that it is not casework-compatible |
| `preprocess_mixture_dfs.py` | Fills allele rows across triplicate profiles | Consolidate with the equivalent functions duplicated in deconvolution scripts |

### LR-distribution and calibration evaluation

| Current script | Role | Proposed disposition |
|---|---|---|
| `read_csv_batch_def_new3.py` | Earlier single-profile scalar evaluation | Archive after regression |
| `read_csv_batch_def_new4.py` | Later single-profile scalar/frequentist evaluation | Provisional frequentist implementation; consolidate into one evaluator |
| `read_csv_batch_def_new3_traces.py` | Rework-profile scalar evaluation | Consolidate through a `profile_mode=rework` option |
| `read_csv_batch_def_new3_traces_draws.py` | Rework-profile draw-wise evaluation | Consolidate through `posterior_mode=drawwise` |
| `read_csv_batch_def_new4_draws.py` | Later single-profile draw-wise/Bayesian evaluation | Provisional Bayesian implementation; consolidate into one evaluator |
| `read_csv_traces_20260318_1315.py` | Earlier predicted-versus-laboratory-rework comparison | Archive after comparison with corrected implementation |
| `read_csv_traces_corrected2.py` | Draw-wise predicted-versus-laboratory-rework comparison with coverage/interval score | Provisional final Bayesian comparison; retain logic |
| `read_csv_traces_replicates_20260324_1800.py` | Creates filtered LR distributions for simulated rework profiles | Retain workflow; extract shared LR and sampling utilities |

The five `read_csv_batch...` scripts should become one module with explicit options such as `profile_mode={single,rework}`, `posterior_mode={scalar,drawwise}`, and `bad_loci_mode={on,zero,impute}`. This preserves scientific distinctions without maintaining five near-copies.

### Diagnostics and thesis analyses

| Current script | Role | Proposed disposition |
|---|---|---|
| `percentile_analysis.py` | Frequentist single-versus-rework summary plots | Convert the final required figures/tables into parameterized reporting functions; retain original in legacy |
| `percentile_analysis_Bayes.py` | Large history of Bayesian/frequentist comparisons; final active block creates single-versus-rework analyses | Extract the final active analysis; retain the 2,838-line original as provenance |
| `mixture_proportions.py` | Mixture-proportion diagnostics | Keep one copy and parameterize inputs |
| `mixture_proportions_Bayes.py` | Executable code is identical to `mixture_proportions.py` | Remove from active package; preserve in legacy |
| `outlier_analysis_def.py` | Locus-level artefact/outlier plots and counts | Retain as a research diagnostic command |
| `wasserstein.py` | Wasserstein-distance summary utility | Move to `evaluation/distances.py` |
| `OLS_results.py` | Chapter 12 algorithm-choice effects, Shapley values, and contrasts | Preserve as a thesis-result script; convert embedded data to a documented CSV if reused |

## Recommended target structure

```text
nfi-rework-lr/
├── README.md
├── pyproject.toml
├── environment.lock.yml
├── CHANGELOG.md
├── CITATION.cff
├── configs/
│   ├── frequentist.yaml
│   ├── bayesian.yaml
│   └── thresholds_ppf6c.yaml
├── resources/
│   ├── NFI_frequencies.csv
│   ├── kit_properties.txt
│   └── README.md
├── src/nfi_rework/
│   ├── config.py
│   ├── identifiers.py
│   ├── io/
│   │   ├── profiles.py
│   │   ├── references.py
│   │   └── results.py
│   ├── model/
│   │   ├── peak_height.py
│   │   ├── genotypes.py
│   │   ├── likelihood.py
│   │   └── artefacts.py
│   ├── frequentist/
│   │   ├── deconvolution.py
│   │   └── lr.py
│   ├── bayesian/
│   │   ├── parameters.py
│   │   ├── priors.py
│   │   ├── sampler.py
│   │   └── deconvolution.py
│   ├── simulation/
│   │   ├── genotype_sampling.py
│   │   ├── profile_generation.py
│   │   └── rework.py
│   ├── dnastatistx/
│   │   ├── command.py
│   │   └── runner.py
│   └── evaluation/
│       ├── calibration.py
│       ├── interval_score.py
│       ├── distances.py
│       └── reporting.py
├── scripts/
│   ├── prepare_profiles.py
│   ├── run_single_analysis.py
│   ├── run_rework_simulation.py
│   ├── run_rework_validation.py
│   └── make_thesis_results.py
├── examples/
│   ├── 1_1A2.txt
│   └── README.md
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── docs/
│   ├── scientific_scope.md
│   ├── data_format.md
│   ├── algorithm_choices.md
│   └── reproduction.md
└── legacy/
    ├── scripts_original/
    ├── resources_original/
    └── CHECKSUMS.sha256
```

The DNAStatistX JAR should be handled according to NFI redistribution rules. If it may be included, keep only the confirmed build in a clearly documented internal resource location. Otherwise, document where an NFI user must obtain it and verify its checksum at startup.

## Test strategy before scientific consolidation

### Unit tests

- Parse `1_1A2`, `3_6E5`, and full sample names consistently.
- Normalize integer, decimal, and missing alleles consistently (`19`, `19.0`, microvariants, and `Ø`).
- Verify genotype prior probabilities and rare-allele handling.
- Check gamma log-density/log-CDF stability for realistic and boundary parameter values.
- Confirm that posterior probabilities sum to one for every locus and every posterior draw.
- Confirm that marginal posterior probabilities sum to one per locus/contributor/draw.
- Verify prediction interval coverage and interval-score formulas on hand-calculated examples.
- Verify deterministic sampling from a supplied RNG seed without modifying global RNG state.

### Integration tests

- Convert a small donor profile and validate DNAStatistX input columns.
- Run preprocessing on a three-replicate synthetic example and verify allele-row alignment.
- Run a one-mixture DNAStatistX smoke test with the canonical JAR, if permitted.
- Run a short MCMC chain on synthetic data and validate output schema, not convergence.
- Execute one simulation under each of the four frequentist/Bayesian algorithm-choice combinations.

### Regression tests

Before refactoring scientific functions, save a small set of known-good output files from the thesis environment. The refactored code should reproduce them within explicit numerical tolerances. At minimum, retain examples supporting:

- frequentist single-profile deconvolution validation;
- observed frequentist rework validation;
- the final frequentist simulation result;
- one MCMC posterior and draw-wise deconvolution result;
- the final Bayesian simulation result;
- the reported coverage values of 69.0% and 81.6%; and
- the reported mean 95% interval scores of 50.5 and 21.6.

Exact stochastic files may require seed-stream preservation rather than only matching aggregate summaries.

## Resource decisions

| Resource | Recommendation |
|---|---|
| `dnastatistx-cli-2.5.0-DES-700-202601191653.jar` | Configuration files point to this build; provisionally canonical. Confirm with NFI and record checksum `0251460948667839752c807fedaae7f27b2075cfa1387f62e370dae9319193c4`. |
| `dnastatistx-cli-2.5.0-DES-700-202601150926.jar` | Treat as superseded until confirmed otherwise; checksum `cd3647f287eca09e5a5eb41aa0c6203569a46a89e0379547c953b764bae3ded4`. |
| `resources/old/` JARs | Exclude from the active runtime; retain only if historical reproduction is required. |
| `NFI_frequencies.csv` | Active allele-frequency table; checksum `a6cf63b7b5232d04db26086dee180e9bf6cd51c3559b52d8c8f6165752d77f85`. |
| `NFI_frequencies_fucked.csv` | Malformed CSV; exclude from active and example resources. |
| `config_2p_Bayes_test.yaml` | Parses correctly and includes degradation plus kit-properties configuration; rename to a meaningful non-test name after confirming it produced the thesis results. |
| `config_*_old_dnax.yaml` | Move to legacy configuration with a short explanation of the difference. |
| `.dnastatistx/dnaStatistx.log` | Do not include as an active resource; retain only if needed for provenance and free of sensitive paths/data. |
| `__pycache__/` | Exclude. |

## Information still needed for a verified handover

1. The exact original Python version and exported package environment.
2. The source and version of `confidence.Configuration`.
3. Confirmation that the `202601191653` DNAStatistX JAR is the final build used for thesis results.
4. The donor genotype/reference inputs needed for `1_1A2` and the cleaned drop-in workflow, preferably anonymized research data rather than casework.
5. A small set of known-good intermediate and final output folders for regression testing.
6. Confirmation of which `corrected2`, `new4`, and `draws` scripts generated the final thesis tables and figures.
7. NFI guidance on whether JAR files, allele frequencies, and example reference profiles should be included in the handover repository.

## Proposed migration sequence

1. **Freeze provenance:** preserve the two supplied ZIP files, create checksums, and copy all original scripts/resources to `legacy/` without edits.
2. **Make execution portable:** add configuration, paths, environment metadata, logging, and safe command-line entry points while retaining existing scientific functions.
3. **Add regression fixtures:** obtain known-good outputs and make the current final scripts reproducible on one or two mixtures.
4. **Extract core modules:** move genotype, likelihood, MCMC, calibration, and DNAStatistX wrapper code into import-safe modules.
5. **Consolidate duplicate families:** unify the batch evaluators, trace evaluators, generation scripts, and pipeline runners behind explicit scientific options.
6. **Reproduce thesis checkpoints:** compare key files and reported aggregate metrics.
7. **Write NFI documentation:** installation, data formats, end-to-end examples, algorithm-choice table, limitations, and troubleshooting.
8. **Package release:** create a versioned internal handover archive with checksums and a clear statement that this is a proof of concept and not validated for casework use.

