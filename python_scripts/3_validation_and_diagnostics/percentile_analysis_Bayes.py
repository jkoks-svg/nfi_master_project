# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 27 08:47:28 2026

# @author: jortk
# """

MIXTURE_TYPE_MARKERS = {
    "A": "o",   # circle
    "B": "X",   # cross
    "C": "s",   # square
    "D": "P",   # plus-filled
    "E": "D",   # diamond
}
import seaborn as sns

# from pathlib import Path
# #from decimal import Decimal, InvalidOperation

# #import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ----------------------------
# # PATHS
# # ----------------------------
# #BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin")
# #version_org = 'v1'
# #version_traces = 'v1'
# CSV_PATH_ORG = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv")
# #BASE / "output_HT_2p_nodropin_runF" /f"results_{version_org}" / f"percentile_df_{version_org}.csv" 
# CSV_PATH_TRACES = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p_corrected_v2\results_v1\percentile_df_v1.csv")
# #Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_2p_corrected_v2\results_v1\percentile_df_v1.csv")
# #BASE / "output_HT_2p_nodropin_runF_traces" /f"results_{version_traces}" / f"percentile_df_{version_traces}.csv" 

# percentile_df_org = pd.read_csv(CSV_PATH_ORG)
# percentile_df_traces = pd.read_csv(CSV_PATH_TRACES)

# trace_params = (
#     percentile_df_traces[
#         [
#             "dataset",
#             "mixture_type",
#             "contributor",
#             "method",
#             "true_donor",
#             'donor_id',
#             "mu_mixture",
#             "sigma_mixture",
#             "true_log10_LR"
#         ]
#     ]
#     .drop_duplicates()
#     .rename(columns={
#         "mu_mixture": "mu_trace",
#         "sigma_mixture": "sigma_trace",
#         "true_log10_LR":"true_log10_LR_trace"
#     })
# )

# percentile_df_org = percentile_df_org.merge(
#     trace_params,
#     on=["dataset", "mixture_type", "contributor", "method", 'true_donor', 'donor_id'],
#     how="left",
# )

# df = percentile_df_org[(percentile_df_org['method']=='ON')&(percentile_df_org['contributor']=='Unknown2')]
# df=df.copy()
# df.loc[:,'mu_delta'] = df.loc[:,'mu_trace']-df.loc[:,'mu_mixture']
# df.loc[:,'sigma_delta'] = df.loc[:,'sigma_trace']-df.loc[:, 'sigma_mixture']
# df.loc[:, 'true_delta'] = df.loc[:,'true_log10_LR_trace']-df.loc[:,'true_log10_LR']

# mu_mixture_avg = df['mu_mixture'].mean()
# mu_trace_avg = df['mu_trace'].mean()
# mu_delta_avg = df['mu_delta'].mean()
# sigma_delta_avg = df['sigma_delta'].mean()
# true_log_LR_avg = df['true_log10_LR'].mean()
# true_log_LR_trace_avg = df['true_log10_LR_trace'].mean()
# true_delta_avg = df['true_delta'].mean()

# plt.figure(figsize=(12, 6))
# plt.hist(df['true_log10_LR'],bins=20, alpha=0.6, label='Single mixtures', color='skyblue', edgecolor='black')
# #plt.hist(df['true_log10_LR_trace'],bins=20, alpha=0.6, label='Rework mixtures', color='salmon', edgecolor='black')
# plt.axvline(true_log_LR_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average logLR = {true_log_LR_avg:.2f}')
# #plt.axvline(true_log_LR_trace_avg, color='red', linestyle='dashed', linewidth=2, label=f'true_log_LR_trace_avg = {true_log_LR_trace_avg:.2f}')
# plt.xlabel('True donor logLR')
# plt.ylabel('Frequency')
# #plt.title('Histogram of true donor logLR for single and rework mixtures')
# plt.grid()
# plt.legend()
# plt.show()

# df_rep1 = df#df[df['replicate']==1]

# plt.figure(figsize=(12, 6))
# #plt.hist(df['true_log10_LR'],bins=20, alpha=0.6, label='Single mixtures', color='skyblue', edgecolor='black')
# plt.hist(df_rep1['true_log10_LR_trace'],bins=20, alpha=0.6, label='Rework mixtures', color='skyblue', edgecolor='black')
# #plt.axvline(true_log_LR_avg, color='blue', linestyle='dashed', linewidth=2, label=f'true_log_LR_avg = {true_log_LR_avg:.2f}')
# plt.axvline(true_log_LR_trace_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average logLR = {true_log_LR_trace_avg:.2f}')
# plt.xlabel('True donor logLR')
# plt.ylabel('Frequency')
# #plt.title('Histogram of true donor logLR for single and rework mixtures')
# plt.grid()
# plt.legend()
# plt.show()


# # plt.figure(figsize=(12, 6))
# # plt.hist(df['mu_mixture'],bins=20, alpha=0.6, label='mu_mixture', color='skyblue', edgecolor='black')
# # plt.hist(df['mu_trace'],bins=20, alpha=0.6, label='mu_trace', color='salmon', edgecolor='black')
# # plt.axvline(mu_mixture_avg, color='blue', linestyle='dashed', linewidth=2, label=f'mu_mixture avg = {mu_mixture_avg:.2f}')
# # plt.axvline(mu_trace_avg, color='red', linestyle='dashed', linewidth=2, label=f'mu_trace avg = {mu_trace_avg:.2f}')
# # plt.xlabel('Increase in logLR of the true donor due to rework')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of mu_mixture and mu_trace')
# # plt.grid()
# # plt.legend()
# # plt.show()

# # plt.figure(figsize=(12, 6))
# # plt.hist(df['mu_delta'],bins=20, alpha=0.6, label='mu_delta', color='skyblue', edgecolor='black')
# # plt.axvline(mu_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'mu_delta avg = {mu_delta_avg:.2f}')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of mu_delta')
# # plt.grid()
# # plt.legend()
# # plt.show()

# # plt.figure(figsize=(12, 6))
# # plt.hist(df['sigma_delta'],bins=20, alpha=0.6, label='sigma_delta', color='skyblue', edgecolor='black')
# # plt.axvline(sigma_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'sigma_delta avg = {sigma_delta_avg:.2f}')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of sigma_delta')
# # plt.grid()
# # plt.legend()
# # plt.show()

# plt.figure(figsize=(12, 6))
# plt.hist(df['true_delta'],bins=20, alpha=0.6, color='skyblue', edgecolor='black') # label='Increase in logLR'
# plt.axvline(true_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average increase = {true_delta_avg:.2f}')
# plt.xlabel('Increase in logLR of the true donor due to Bayes instead of frequentist approach')
# plt.ylabel('Frequency')
# #plt.title('Histogram of true donor logLR increase due to rework')
# plt.grid()
# plt.legend()
# plt.show()

# mixture_types = ['A','B','C','D','E']
# for mixture_type in mixture_types:
#     df_type = df[df['mixture_type']==mixture_type]
#     true_delta_avg_type = df_type['true_delta'].mean()
#     plt.figure(figsize=(12, 6))
#     plt.hist(df_type['true_delta'],bins=20, alpha=0.6, label='mu_delta', color='skyblue', edgecolor='black')
#     plt.axvline(true_delta_avg_type, color='blue', linestyle='dashed', linewidth=2, label=f'true_delta avg {mixture_type} = {true_delta_avg_type:.2f}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of mu_delta {mixture_type}')
#     plt.grid()
#     plt.legend()
#     plt.show()
#     sigma_delta_avg_type = df_type['sigma_delta'].mean()
#     plt.figure(figsize=(12, 6))
#     plt.hist(df_type['sigma_delta'],bins=20, alpha=0.6, label='sigma_delta', color='skyblue', edgecolor='black')
#     plt.axvline(sigma_delta_avg_type, color='blue', linestyle='dashed', linewidth=2, label=f'sigma_delta avg {mixture_type} = {sigma_delta_avg_type:.2f}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of sigma_delta {mixture_type}')
#     plt.grid()
#     plt.legend()
#     plt.show()
    
    
# # =============================================================================
# # Scatterplot: single mixture true donor LR vs rework true donor LR
# # =============================================================================

# plt.figure(figsize=(8, 7))

# plt.scatter(
#     df["true_log10_LR"],
#     df["true_log10_LR_trace"],
#     alpha=0.7,
#     edgecolor="black"
# )

# # Add y = x reference line
# min_val = min(df["true_log10_LR"].min(), df["true_log10_LR_trace"].min())
# max_val = max(df["true_log10_LR"].max(), df["true_log10_LR_trace"].max())

# plt.plot(
#     [min_val, max_val],
#     [min_val, max_val],
#     linestyle="--",
#     color="black",
#     label="No change after rework"
# )

# # Optional: correlation
# corr = df["true_log10_LR"].corr(df["true_log10_LR_trace"])

# plt.xlabel("Single mixture true donor log10 LR")
# plt.ylabel("Rework mixture true donor log10 LR")
# plt.title(
#     f"Single mixture vs rework true donor LR\n"
#     f"Unknown2, method ON, correlation = {corr:.2f}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # =============================================================================
# # Scatterplot: single mixture true donor LR vs rework true donor LR
# # colored by mixture type
# # =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR"],
#         sub_df["true_log10_LR_trace"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type}"
#     )

# # Add y = x reference line
# min_val = min(df["true_log10_LR"].min(), df["true_log10_LR_trace"].min())
# max_val = max(df["true_log10_LR"].max(), df["true_log10_LR_trace"].max())

# plt.plot(
#     [min_val, max_val],
#     [min_val, max_val],
#     linestyle="--",
#     color="black",
#     label="No change after rework"
# )

# corr = df["true_log10_LR"].corr(df["true_log10_LR_trace"])

# plt.xlabel("Single mixture true donor log10 LR")
# plt.ylabel("Rework mixture true donor log10 LR")
# plt.title(
#     f"Single mixture vs rework true donor LR\n"
#     f"Unknown2, method ON, correlation = {corr:.2f}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # # Compute correlation
# # corr = df["n_bad_loci"].corr(df["mu_delta"])
# # print(f"Correlation between n_bad_loci and mu_delta: {corr:.4f}")

# # # Plot
# # plt.figure(figsize=(6, 4))
# # plt.scatter(df["n_bad_loci"], df["mu_delta"], color='teal', edgecolor='k', s=70)
# # plt.title("Correlation between n_bad_loci and mu_delta", fontsize=14)
# # plt.xlabel("n_bad_loci", fontsize=12)
# # plt.ylabel("mu_delta", fontsize=12)
# # plt.grid(True, linestyle="--", alpha=0.6)

# # # Annotate correlation value
# # plt.text(
# #     0.05, 0.95, f"r = {corr:.4f}",
# #     transform=plt.gca().transAxes,
# #     fontsize=12, color="darkred", ha="left", va="top"
# # )

# # plt.tight_layout()
# # plt.show()

# # --- Extract variables ---
# x = df["n_bad_loci"]
# y = df['true_delta'] #df["mu_delta"]

# # --- Compute correlation ---
# corr = np.corrcoef(x, y)[0, 1]
# print(f"Correlation between n_bad_loci and mu_delta: {corr:.4f}")

# # --- Fit regression line ---
# slope, intercept = np.polyfit(x, y, 1)
# print('slope = ', slope)
# y_pred = slope * x + intercept

# # --- Plot ---
# plt.figure(figsize=(7, 5))
# plt.scatter(x, y, color="teal", edgecolor="black", s=70, label="Single mixtures")
# plt.plot(x, y_pred, color="darkred", linewidth=2, label="Least squares fit") #f"delta_logLR = {slope:.2f}*N_loci_dropout + {intercept:.2f}")

# #plt.title("Number of loci with dropout vs gain in LR due to rework", fontsize=14)
# plt.xlabel("Number of loci with dropout", fontsize=12)
# plt.ylabel("Increase in logLR of the true donor due to rework", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)

# # # Show correlation on the plot
# # plt.text(0.05, 0.95, f"r = {corr:.4f}", transform=plt.gca().transAxes,
# #          fontsize=12, color="darkred", ha="left", va="top")

# plt.tight_layout()
# plt.show()

# pivot = pd.pivot_table(df, 
#                        values="n_bad_loci", 
#                        index="mixture_type", 
#                        aggfunc="mean")

# print(pivot)

























# -*- coding: utf-8 -*-
"""
Compare true donor log10 LRs from MLE and MCMC implementations.

x-axis:  true donor log10 LR from MLE
y-axis:  true donor log10 LR from MCMC
delta:   MCMC - MLE
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Paths
# =============================================================================

CSV_PATH_MLE = Path(
    r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv"
)

CSV_PATH_MCMC = Path(
    r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p_corrected_v2\results_v1\percentile_df_v1.csv"
)


# =============================================================================
# Settings
# =============================================================================

METHOD_TO_KEEP = "ON"
CONTRIBUTOR_TO_KEEP = "Unknown2"


# =============================================================================
# Read data
# =============================================================================

df_mle = pd.read_csv(CSV_PATH_MLE)
df_mcmc = pd.read_csv(CSV_PATH_MCMC)


# =============================================================================
# Keep only the columns needed for comparison
# =============================================================================

merge_cols = [
    "mixture",
    "dataset",
    "replicate",
    "mixture_type",
    "contributor",
    "method",
    "true_donor",
    "donor_id",
]

mle_cols = merge_cols + [
    "true_log10_LR",
    "mu_mixture",
    "sigma_mixture",
    "n_bad_loci",
    "degenerate_sample",
]

mcmc_cols = merge_cols + [
    "true_log10_LR",
    "mu_mixture",
    "sigma_mixture",
    "n_bad_loci",
    "degenerate_sample",
]


df_mle_small = (
    df_mle[mle_cols]
    .drop_duplicates(subset=merge_cols)
    .rename(
        columns={
            "true_log10_LR": "true_log10_LR_MLE",
            "mu_mixture": "mu_MLE",
            "sigma_mixture": "sigma_MLE",
            "n_bad_loci": "n_bad_loci_MLE",
            "degenerate_sample": "degenerate_sample_MLE",
        }
    )
)

df_mcmc_small = (
    df_mcmc[mcmc_cols]
    .drop_duplicates(subset=merge_cols)
    .rename(
        columns={
            "true_log10_LR": "true_log10_LR_MCMC",
            "mu_mixture": "mu_MCMC",
            "sigma_mixture": "sigma_MCMC",
            "n_bad_loci": "n_bad_loci_MCMC",
            "degenerate_sample": "degenerate_sample_MCMC",
        }
    )
)


# =============================================================================
# Merge MLE and MCMC rows
# =============================================================================

df = df_mle_small.merge(
    df_mcmc_small,
    on=merge_cols,
    how="inner",
    validate="one_to_one",
)

print(f"Rows in MLE file:  {len(df_mle)}")
print(f"Rows in MCMC file: {len(df_mcmc)}")
print(f"Rows after merge: {len(df)}")


# =============================================================================
# Filter to method/contributor of interest
# =============================================================================

df = df[
    (df["method"] == METHOD_TO_KEEP)
    & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
].copy()

# Convert to numeric, just to be safe
numeric_cols = [
    "true_log10_LR_MLE",
    "true_log10_LR_MCMC",
    "mu_MLE",
    "mu_MCMC",
    "sigma_MLE",
    "sigma_MCMC",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["true_log10_LR_MLE", "true_log10_LR_MCMC"]).copy()


# =============================================================================
# Calculate differences
# =============================================================================

df["delta_log10_LR_MCMC_minus_MLE"] = (
    df["true_log10_LR_MCMC"] - df["true_log10_LR_MLE"]
)

df["mu_delta_MCMC_minus_MLE"] = df["mu_MCMC"] - df["mu_MLE"]
df["sigma_delta_MCMC_minus_MLE"] = df["sigma_MCMC"] - df["sigma_MLE"]


# =============================================================================
# Summary
# =============================================================================

print("\nSummary: MCMC - MLE true donor log10 LR")
print(df["delta_log10_LR_MCMC_minus_MLE"].describe())

print("\nMean true donor log10 LR:")
print(f"MLE:  {df['true_log10_LR_MLE'].mean():.3f}")
print(f"MCMC: {df['true_log10_LR_MCMC'].mean():.3f}")
print(f"Mean difference MCMC - MLE: {df['delta_log10_LR_MCMC_minus_MLE'].mean():.3f}")


# =============================================================================
# Scatterplot 1:
# MLE true donor log10 LR vs MCMC true donor log10 LR
# =============================================================================

plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_MLE"],
#         sub_df["true_log10_LR_MCMC"],
#         alpha=0.75,
#         edgecolor="black",
#         marker=MIXTURE_TYPE_MARKERS.get(mixture_type, "o"),
#         label=f"Mixture type {mixture_type}",
#     )
sns.scatterplot(
    data=df,
    x="true_log10_LR_MLE",
    y="true_log10_LR_MCMC",
    hue="mixture_type",
    style="mixture_type",
    markers=MIXTURE_TYPE_MARKERS,
    hue_order=["A", "B", "C", "D", "E"],
    style_order=["A", "B", "C", "D", "E"],
    s=80,
    alpha=0.8,
)

# y = x reference line
min_val = min(
    df["true_log10_LR_MLE"].min(),
    df["true_log10_LR_MCMC"].min(),
)

max_val = max(
    df["true_log10_LR_MLE"].max(),
    df["true_log10_LR_MCMC"].max(),
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="black",
    label="MCMC = MLE",
)

corr = df["true_log10_LR_MLE"].corr(df["true_log10_LR_MCMC"])

plt.xlabel("True donor log10 LR, MLE")
plt.ylabel("True donor log10 LR, MCMC")
# plt.title(
#     f"True donor log10 LR: MLE vs MCMC\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, correlation = {corr:.2f}"
# )

plt.grid(alpha=0.3)
plt.legend(title="Mixture type")
plt.tight_layout()
plt.show()


# # =============================================================================
# # Scatterplot 2:
# # MLE true donor log10 LR vs difference MCMC - MLE
# # =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_MLE"],
#         sub_df["delta_log10_LR_MCMC_minus_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type}",
#     )

# plt.axhline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No difference",
# )

# mean_delta = df["delta_log10_LR_MCMC_minus_MLE"].mean()

# plt.axhline(
#     mean_delta,
#     linestyle=":",
#     color="black",
#     label=f"Mean difference = {mean_delta:.2f}",
# )

# plt.xlabel("True donor log10 LR, MLE")
# plt.ylabel("Difference in true donor log10 LR: MCMC - MLE")
# plt.title(
#     f"Difference in true donor log10 LR between MCMC and MLE\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Histogram:
# # Difference MCMC - MLE
# # =============================================================================

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df["delta_log10_LR_MCMC_minus_MLE"],
#     bins=20,
#     alpha=0.7,
#     edgecolor="black",
# )

# plt.axvline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No difference",
# )

# plt.axvline(
#     mean_delta,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Mean difference = {mean_delta:.2f}",
# )

# plt.xlabel("Difference in true donor log10 LR: MCMC - MLE")
# plt.ylabel("Frequency")
# plt.title(
#     f"Distribution of true donor log10 LR differences\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Optional: histogram per mixture type
# # =============================================================================

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     mean_delta_type = sub_df["delta_log10_LR_MCMC_minus_MLE"].mean()

#     plt.figure(figsize=(10, 6))

#     plt.hist(
#         sub_df["delta_log10_LR_MCMC_minus_MLE"],
#         bins=15,
#         alpha=0.7,
#         edgecolor="black",
#     )

#     plt.axvline(
#         0,
#         linestyle="--",
#         color="black",
#         label="No difference",
#     )

#     plt.axvline(
#         mean_delta_type,
#         linestyle=":",
#         color="black",
#         linewidth=2,
#         label=f"Mean difference = {mean_delta_type:.2f}",
#     )

#     plt.xlabel("Difference in true donor log10 LR: MCMC - MLE")
#     plt.ylabel("Frequency")
#     plt.title(
#         f"True donor log10 LR difference for mixture type {mixture_type}\n"
#         f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}"
#     )

#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # =============================================================================
# # Optional: average difference per mixture type
# # =============================================================================

# summary_by_type = (
#     df.groupby("mixture_type")["delta_log10_LR_MCMC_minus_MLE"]
#     .agg(["count", "mean", "std", "min", "max"])
#     .reset_index()
# )

# print("\nDifference MCMC - MLE per mixture type:")
# print(summary_by_type)













































# -*- coding: utf-8 -*-
"""
Compare true donor log10 LRs for rework MLE and rework MCMC.

Input:
    percentile_df_v1.csv = rework MLE
    percentile_df_v2.csv = rework MCMC

Main comparison:
    x = true donor log10 LR, rework MLE
    y = true donor log10 LR, rework MCMC

Difference:
    delta = MCMC - MLE
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Paths
# =============================================================================

CSV_PATH_REWORK_MLE = Path(
    r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MLE_2p\results_v1\percentile_df_v1.csv"
)

CSV_PATH_REWORK_MCMC = Path(
    r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_2p_corrected_v2\results_v2\percentile_df_v2.csv"
)


# =============================================================================
# Settings
# =============================================================================

METHOD_TO_KEEP = "ON"
CONTRIBUTOR_TO_KEEP = "Unknown2"


# =============================================================================
# Read data
# =============================================================================

df_mle = pd.read_csv(CSV_PATH_REWORK_MLE)
df_mcmc = pd.read_csv(CSV_PATH_REWORK_MCMC)

print("MLE columns:")
print(df_mle.columns.tolist())

print("\nMCMC columns:")
print(df_mcmc.columns.tolist())


# =============================================================================
# Merge keys
# =============================================================================
# These columns identify the same mixture/contributor/result row in both files.
# Do NOT include percentile, true_log10_LR, mu_mixture, sigma_mixture, etc.
# Those are the values we want to compare.

merge_cols = [
    "mixture",
    "dataset",
    "mixture_type",
    "contributor",
    "true_donor",
    "donor_id",
    "method",
]


# =============================================================================
# Keep and rename MLE columns
# =============================================================================

mle_cols = merge_cols + [
    "percentile",
    "true_log10_LR",
    "mu_mixture",
    "sigma_mixture",
    "n_bad_loci",
    "degenerate_sample",
]

df_mle_small = (
    df_mle[mle_cols]
    .drop_duplicates(subset=merge_cols)
    .rename(
        columns={
            "percentile": "percentile_MLE",
            "true_log10_LR": "true_log10_LR_MLE",
            "mu_mixture": "mu_MLE",
            "sigma_mixture": "sigma_MLE",
            "n_bad_loci": "n_bad_loci_MLE",
            "degenerate_sample": "degenerate_sample_MLE",
        }
    )
)


# =============================================================================
# Keep and rename MCMC columns
# =============================================================================

mcmc_cols = merge_cols + [
    "percentile",
    "true_log10_LR",
    "mu_mixture",
    "sigma_mixture",
    "n_bad_loci",
    "degenerate_sample",
]

df_mcmc_small = (
    df_mcmc[mcmc_cols]
    .drop_duplicates(subset=merge_cols)
    .rename(
        columns={
            "percentile": "percentile_MCMC",
            "true_log10_LR": "true_log10_LR_MCMC",
            "mu_mixture": "mu_MCMC",
            "sigma_mixture": "sigma_MCMC",
            "n_bad_loci": "n_bad_loci_MCMC",
            "degenerate_sample": "degenerate_sample_MCMC",
        }
    )
)


# =============================================================================
# Check for duplicated merge keys before merging
# =============================================================================

n_dup_mle = df_mle.duplicated(subset=merge_cols).sum()
n_dup_mcmc = df_mcmc.duplicated(subset=merge_cols).sum()

print(f"\nDuplicated MLE rows based on merge keys:  {n_dup_mle}")
print(f"Duplicated MCMC rows based on merge keys: {n_dup_mcmc}")


# =============================================================================
# Merge MLE and MCMC results
# =============================================================================

df = df_mle_small.merge(
    df_mcmc_small,
    on=merge_cols,
    how="inner",
    validate="one_to_one",
)

print(f"\nRows in rework MLE file:  {len(df_mle)}")
print(f"Rows in rework MCMC file: {len(df_mcmc)}")
print(f"Rows after merge:         {len(df)}")


# =============================================================================
# Filter to method and contributor of interest
# =============================================================================

df = df[
    (df["method"] == METHOD_TO_KEEP)
    & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
].copy()

print(f"\nRows after filtering to method={METHOD_TO_KEEP}, contributor={CONTRIBUTOR_TO_KEEP}: {len(df)}")


# =============================================================================
# Convert numeric columns
# =============================================================================

numeric_cols = [
    "true_log10_LR_MLE",
    "true_log10_LR_MCMC",
    "mu_MLE",
    "mu_MCMC",
    "sigma_MLE",
    "sigma_MCMC",
    "percentile_MLE",
    "percentile_MCMC",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove rows without both true donor LRs
df = df.dropna(subset=["true_log10_LR_MLE", "true_log10_LR_MCMC"]).copy()

print(f"Rows used in plots after dropping missing LR values: {len(df)}")


# =============================================================================
# Compute correct changes
# =============================================================================

df["delta_log10_LR_MCMC_minus_MLE"] = (
    df["true_log10_LR_MCMC"] - df["true_log10_LR_MLE"]
)

df["mu_delta_MCMC_minus_MLE"] = df["mu_MCMC"] - df["mu_MLE"]
df["sigma_delta_MCMC_minus_MLE"] = df["sigma_MCMC"] - df["sigma_MLE"]


# =============================================================================
# Summary
# =============================================================================

n_points = len(df)
corr = df["true_log10_LR_MLE"].corr(df["true_log10_LR_MCMC"])
mean_delta = df["delta_log10_LR_MCMC_minus_MLE"].mean()

print("\nSummary: rework MCMC - rework MLE true donor log10 LR")
print(df["delta_log10_LR_MCMC_minus_MLE"].describe())

print("\nMean true donor log10 LR:")
print(f"Rework MLE:  {df['true_log10_LR_MLE'].mean():.3f}")
print(f"Rework MCMC: {df['true_log10_LR_MCMC'].mean():.3f}")
print(f"Mean difference MCMC - MLE: {mean_delta:.3f}")
print(f"Correlation: {corr:.3f}")
print(f"Number of plotted points: {n_points}")


# =============================================================================
# Scatterplot 1:
# rework MLE true donor log10 LR vs rework MCMC true donor log10 LR
# =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_MLE"],
#         sub_df["true_log10_LR_MCMC"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )
plt.figure(figsize=(8, 7))

sns.scatterplot(
    data=df,
    x="true_log10_LR_MLE",
    y="true_log10_LR_MCMC",
    hue="mixture_type",
    style="mixture_type",
    markers=MIXTURE_TYPE_MARKERS,
    hue_order=["A", "B", "C", "D", "E"],
    style_order=["A", "B", "C", "D", "E"],
    s=80,
    alpha=0.8,
)
# y = x reference line
min_val = min(
    df["true_log10_LR_MLE"].min(),
    df["true_log10_LR_MCMC"].min(),
)

max_val = max(
    df["true_log10_LR_MLE"].max(),
    df["true_log10_LR_MCMC"].max(),
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="black",
    label="MCMC = MLE",
)

plt.xlabel("True donor log10 LR, rework MLE")
plt.ylabel("True donor log10 LR, rework MCMC")
# plt.title(
#     f"Rework true donor log10 LR: MLE vs MCMC\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, "
#     f"correlation = {corr:.2f}, n = {n_points}"
# )

plt.grid(alpha=0.3)
plt.legend(title="Mixture type")
plt.tight_layout()
plt.show()


# # =============================================================================
# # Scatterplot 2:
# # rework MLE true donor log10 LR vs difference MCMC - MLE
# # =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_MLE"],
#         sub_df["delta_log10_LR_MCMC_minus_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# plt.axhline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No difference",
# )

# plt.axhline(
#     mean_delta,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Mean difference = {mean_delta:.2f}",
# )

# plt.xlabel("True donor log10 LR, rework MLE")
# plt.ylabel("Difference in true donor log10 LR: rework MCMC - rework MLE")
# plt.title(
#     f"Difference in rework true donor log10 LR between MCMC and MLE\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Histogram:
# # difference MCMC - MLE
# # =============================================================================

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df["delta_log10_LR_MCMC_minus_MLE"],
#     bins=20,
#     alpha=0.7,
#     edgecolor="black",
# )

# plt.axvline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No difference",
# )

# plt.axvline(
#     mean_delta,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Mean difference = {mean_delta:.2f}",
# )

# plt.xlabel("Difference in true donor log10 LR: rework MCMC - rework MLE")
# plt.ylabel("Frequency")
# plt.title(
#     f"Distribution of rework true donor log10 LR differences\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Summary table per mixture type
# # =============================================================================

# summary_by_type = (
#     df.groupby("mixture_type")["delta_log10_LR_MCMC_minus_MLE"]
#     .agg(["count", "mean", "std", "min", "max"])
#     .reset_index()
# )

# print("\nDifference rework MCMC - rework MLE per mixture type:")
# print(summary_by_type)


















# # -*- coding: utf-8 -*-
# """
# Compare Bayesian true donor log10 LRs for single mixtures and rework mixtures.

# Input:
#     percentile_df_v1.csv = Bayesian single-mixture results
#     percentile_df_v2.csv = Bayesian rework-mixture results

# Main comparison:
#     x = true donor log10 LR, Bayesian single mixture
#     y = true donor log10 LR, Bayesian rework mixture

# Difference:
#     gain = rework - single

# Reference line in scatterplot:
#     y = x + average_gain
# """

# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# # =============================================================================
# # Paths
# # =============================================================================

# CSV_PATH_SINGLE_BAYES = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p_corrected_v2\results_v1\percentile_df_v1.csv"
# )

# CSV_PATH_REWORK_BAYES = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_2p_corrected_v2\results_v2\percentile_df_v2.csv"
# )


# # =============================================================================
# # Settings
# # =============================================================================

# METHOD_TO_KEEP = "ON"
# CONTRIBUTOR_TO_KEEP = "Unknown2"


# # =============================================================================
# # Read data
# # =============================================================================

# df_single = pd.read_csv(CSV_PATH_SINGLE_BAYES)
# df_rework = pd.read_csv(CSV_PATH_REWORK_BAYES)

# print("Single Bayesian columns:")
# print(df_single.columns.tolist())

# print("\nRework Bayesian columns:")
# print(df_rework.columns.tolist())


# # =============================================================================
# # Merge keys
# # =============================================================================
# # These identify the same mixture/contributor/result in both files.
# # I do NOT include "mixture", because single and rework mixture names may differ.

# merge_cols = [
#     "dataset",
#     "mixture_type",
#     "contributor",
#     "true_donor",
#     "donor_id",
#     "method",
# ]

# print("\nMerge columns:")
# print(merge_cols)


# # =============================================================================
# # Keep and rename single-mixture columns
# # =============================================================================

# single_cols = merge_cols + [
#     "true_log10_LR",
#     "mu_mixture",
#     "sigma_mixture",
#     "n_bad_loci",
#     "degenerate_sample",
# ]

# single_cols = [col for col in single_cols if col in df_single.columns]

# df_single_small = (
#     df_single[single_cols]
#     .drop_duplicates(subset=merge_cols)
#     .rename(
#         columns={
#             "true_log10_LR": "true_log10_LR_single",
#             "mu_mixture": "mu_single",
#             "sigma_mixture": "sigma_single",
#             "n_bad_loci": "n_bad_loci_single",
#             "degenerate_sample": "degenerate_sample_single",
#         }
#     )
# )


# # =============================================================================
# # Keep and rename rework-mixture columns
# # =============================================================================

# rework_cols = merge_cols + [
#     "true_log10_LR",
#     "mu_mixture",
#     "sigma_mixture",
#     "n_bad_loci",
#     "degenerate_sample",
# ]

# rework_cols = [col for col in rework_cols if col in df_rework.columns]

# df_rework_small = (
#     df_rework[rework_cols]
#     .drop_duplicates(subset=merge_cols)
#     .rename(
#         columns={
#             "true_log10_LR": "true_log10_LR_rework",
#             "mu_mixture": "mu_rework",
#             "sigma_mixture": "sigma_rework",
#             "n_bad_loci": "n_bad_loci_rework",
#             "degenerate_sample": "degenerate_sample_rework",
#         }
#     )
# )


# # =============================================================================
# # Check duplicated merge keys
# # =============================================================================

# n_dup_single = df_single.duplicated(subset=merge_cols).sum()
# n_dup_rework = df_rework.duplicated(subset=merge_cols).sum()

# print(f"\nDuplicated single rows based on merge keys: {n_dup_single}")
# print(f"Duplicated rework rows based on merge keys: {n_dup_rework}")


# # =============================================================================
# # Merge single and rework Bayesian results
# # =============================================================================

# df = df_single_small.merge(
#     df_rework_small,
#     on=merge_cols,
#     how="inner",
#     validate="one_to_one",
# )

# print(f"\nRows in Bayesian single file: {len(df_single)}")
# print(f"Rows in Bayesian rework file: {len(df_rework)}")
# print(f"Rows after merge:            {len(df)}")


# # =============================================================================
# # Filter to method and contributor of interest
# # =============================================================================

# df = df[
#     (df["method"] == METHOD_TO_KEEP)
#     & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
# ].copy()

# print(
#     f"\nRows after filtering to method={METHOD_TO_KEEP}, "
#     f"contributor={CONTRIBUTOR_TO_KEEP}: {len(df)}"
# )


# # =============================================================================
# # Convert numeric columns
# # =============================================================================

# numeric_cols = [
#     "true_log10_LR_single",
#     "true_log10_LR_rework",
#     "mu_single",
#     "mu_rework",
#     "sigma_single",
#     "sigma_rework",
# ]

# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")


# # Remove rows without both true donor LRs
# df = df.dropna(subset=["true_log10_LR_single", "true_log10_LR_rework"]).copy()

# print(f"Rows used in plots after dropping missing LR values: {len(df)}")


# # =============================================================================
# # Compute correct gain
# # =============================================================================

# df["delta_log10_LR_rework_minus_single"] = (
#     df["true_log10_LR_rework"] - df["true_log10_LR_single"]
# )

# if "mu_single" in df.columns and "mu_rework" in df.columns:
#     df["mu_delta_rework_minus_single"] = df["mu_rework"] - df["mu_single"]

# if "sigma_single" in df.columns and "sigma_rework" in df.columns:
#     df["sigma_delta_rework_minus_single"] = df["sigma_rework"] - df["sigma_single"]


# # =============================================================================
# # Summary
# # =============================================================================

# n_points = len(df)
# corr = df["true_log10_LR_single"].corr(df["true_log10_LR_rework"])
# average_gain = df["delta_log10_LR_rework_minus_single"].mean()

# print("\nSummary: Bayesian rework - Bayesian single true donor log10 LR")
# print(df["delta_log10_LR_rework_minus_single"].describe())

# print("\nMean true donor log10 LR:")
# print(f"Bayesian single: {df['true_log10_LR_single'].mean():.3f}")
# print(f"Bayesian rework: {df['true_log10_LR_rework'].mean():.3f}")
# print(f"Average gain rework - single: {average_gain:.3f}")
# print(f"Correlation: {corr:.3f}")
# print(f"Number of plotted points: {n_points}")


# # =============================================================================
# # Scatterplot 1:
# # Bayesian single true donor log10 LR vs Bayesian rework true donor log10 LR
# # with reference line y = x + average_gain
# # =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["true_log10_LR_rework"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# # Reference line: y = x + average_gain
# x = df["true_log10_LR_single"]
# y = df["true_log10_LR_rework"]

# x_line = np.linspace(x.min(), x.max(), 100)
# y_line = x_line + average_gain

# plt.plot(
#     x_line,
#     y_line,
#     linestyle="--",
#     color="black",
#     linewidth=2,
#     label=f"y = x + average gain = x + {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, Bayesian single mixture")
# plt.ylabel("True donor log10 LR, Bayesian rework mixture")
# plt.title(
#     f"Bayesian true donor log10 LR: single vs rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, "
#     f"correlation = {corr:.2f}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Scatterplot 2:
# # Bayesian single true donor log10 LR vs gain after rework
# # =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["delta_log10_LR_rework_minus_single"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# plt.axhline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axhline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, Bayesian single mixture")
# plt.ylabel("Gain in true donor log10 LR: Bayesian rework - Bayesian single")
# plt.title(
#     f"Bayesian true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Histogram:
# # Bayesian rework - Bayesian single
# # =============================================================================

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df["delta_log10_LR_rework_minus_single"],
#     bins=20,
#     alpha=0.7,
#     edgecolor="black",
# )

# plt.axvline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axvline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("Gain in true donor log10 LR: Bayesian rework - Bayesian single")
# plt.ylabel("Frequency")
# plt.title(
#     f"Distribution of Bayesian true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Summary table per mixture type
# # =============================================================================

# summary_by_type = (
#     df.groupby("mixture_type")["delta_log10_LR_rework_minus_single"]
#     .agg(["count", "mean", "std", "min", "max"])
#     .reset_index()
# )

# print("\nBayesian rework - Bayesian single per mixture type:")
# print(summary_by_type)

























# # -*- coding: utf-8 -*-
# """
# Compare frequentist / MLE true donor log10 LRs for single mixtures and rework mixtures.

# Input:
#     percentile_df_v3.csv = frequentist single-mixture results
#     percentile_df_v1.csv = frequentist rework-mixture results

# Main comparison:
#     x = true donor log10 LR, frequentist single mixture
#     y = true donor log10 LR, frequentist rework mixture

# Difference:
#     gain = rework - single

# Reference line in scatterplot:
#     y = x + average_gain
# """

# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# # =============================================================================
# # Paths
# # =============================================================================

# CSV_PATH_SINGLE_MLE = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv"
# )

# CSV_PATH_REWORK_MLE = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MLE_2p\results_v1\percentile_df_v1.csv"
# )


# # =============================================================================
# # Settings
# # =============================================================================

# METHOD_TO_KEEP = "ON"
# CONTRIBUTOR_TO_KEEP = "Unknown2"


# # =============================================================================
# # Read data
# # =============================================================================

# df_single = pd.read_csv(CSV_PATH_SINGLE_MLE)
# df_rework = pd.read_csv(CSV_PATH_REWORK_MLE)

# print("Single MLE columns:")
# print(df_single.columns.tolist())

# print("\nRework MLE columns:")
# print(df_rework.columns.tolist())


# # =============================================================================
# # Merge keys
# # =============================================================================
# # These identify the same dataset / mixture type / contributor in both files.
# # Do NOT include true_log10_LR, mu_mixture, sigma_mixture, etc.
# # Those are values we want to compare.
# #
# # I also do NOT include "mixture", because single and rework mixture names
# # may differ.

# merge_cols = [
#     "dataset",
#     "mixture_type",
#     "contributor",
#     "true_donor",
#     "donor_id",
#     "method",
# ]

# print("\nMerge columns:")
# print(merge_cols)


# # =============================================================================
# # Keep and rename single-mixture columns
# # =============================================================================

# single_cols = merge_cols + [
#     "true_log10_LR",
#     "mu_mixture",
#     "sigma_mixture",
#     "n_bad_loci",
#     "degenerate_sample",
# ]

# single_cols = [col for col in single_cols if col in df_single.columns]

# df_single_small = (
#     df_single[single_cols]
#     .drop_duplicates(subset=merge_cols)
#     .rename(
#         columns={
#             "true_log10_LR": "true_log10_LR_single_MLE",
#             "mu_mixture": "mu_single_MLE",
#             "sigma_mixture": "sigma_single_MLE",
#             "n_bad_loci": "n_bad_loci_single_MLE",
#             "degenerate_sample": "degenerate_sample_single_MLE",
#         }
#     )
# )


# # =============================================================================
# # Keep and rename rework-mixture columns
# # =============================================================================

# rework_cols = merge_cols + [
#     "true_log10_LR",
#     "mu_mixture",
#     "sigma_mixture",
#     "n_bad_loci",
#     "degenerate_sample",
# ]

# rework_cols = [col for col in rework_cols if col in df_rework.columns]

# df_rework_small = (
#     df_rework[rework_cols]
#     .drop_duplicates(subset=merge_cols)
#     .rename(
#         columns={
#             "true_log10_LR": "true_log10_LR_rework_MLE",
#             "mu_mixture": "mu_rework_MLE",
#             "sigma_mixture": "sigma_rework_MLE",
#             "n_bad_loci": "n_bad_loci_rework_MLE",
#             "degenerate_sample": "degenerate_sample_rework_MLE",
#         }
#     )
# )


# # =============================================================================
# # Check duplicated merge keys
# # =============================================================================

# n_dup_single = df_single.duplicated(subset=merge_cols).sum()
# n_dup_rework = df_rework.duplicated(subset=merge_cols).sum()

# print(f"\nDuplicated single rows based on merge keys: {n_dup_single}")
# print(f"Duplicated rework rows based on merge keys: {n_dup_rework}")


# # =============================================================================
# # Merge single and rework MLE results
# # =============================================================================

# df = df_single_small.merge(
#     df_rework_small,
#     on=merge_cols,
#     how="inner",
#     validate="one_to_one",
# )

# print(f"\nRows in frequentist single file: {len(df_single)}")
# print(f"Rows in frequentist rework file: {len(df_rework)}")
# print(f"Rows after merge:               {len(df)}")


# # =============================================================================
# # Filter to method and contributor of interest
# # =============================================================================

# df = df[
#     (df["method"] == METHOD_TO_KEEP)
#     & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
# ].copy()

# print(
#     f"\nRows after filtering to method={METHOD_TO_KEEP}, "
#     f"contributor={CONTRIBUTOR_TO_KEEP}: {len(df)}"
# )


# # =============================================================================
# # Convert numeric columns
# # =============================================================================

# numeric_cols = [
#     "true_log10_LR_single_MLE",
#     "true_log10_LR_rework_MLE",
#     "mu_single_MLE",
#     "mu_rework_MLE",
#     "sigma_single_MLE",
#     "sigma_rework_MLE",
# ]

# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")


# # Remove rows without both true donor LRs
# df = df.dropna(
#     subset=["true_log10_LR_single_MLE", "true_log10_LR_rework_MLE"]
# ).copy()

# print(f"Rows used in plots after dropping missing LR values: {len(df)}")


# # =============================================================================
# # Compute correct gain
# # =============================================================================

# df["delta_log10_LR_rework_minus_single_MLE"] = (
#     df["true_log10_LR_rework_MLE"] - df["true_log10_LR_single_MLE"]
# )

# if "mu_single_MLE" in df.columns and "mu_rework_MLE" in df.columns:
#     df["mu_delta_rework_minus_single_MLE"] = (
#         df["mu_rework_MLE"] - df["mu_single_MLE"]
#     )

# if "sigma_single_MLE" in df.columns and "sigma_rework_MLE" in df.columns:
#     df["sigma_delta_rework_minus_single_MLE"] = (
#         df["sigma_rework_MLE"] - df["sigma_single_MLE"]
#     )


# # =============================================================================
# # Summary
# # =============================================================================

# n_points = len(df)
# corr = df["true_log10_LR_single_MLE"].corr(df["true_log10_LR_rework_MLE"])
# average_gain = df["delta_log10_LR_rework_minus_single_MLE"].mean()

# print("\nSummary: frequentist rework - frequentist single true donor log10 LR")
# print(df["delta_log10_LR_rework_minus_single_MLE"].describe())

# print("\nMean true donor log10 LR:")
# print(f"Frequentist single: {df['true_log10_LR_single_MLE'].mean():.3f}")
# print(f"Frequentist rework: {df['true_log10_LR_rework_MLE'].mean():.3f}")
# print(f"Average gain rework - single: {average_gain:.3f}")
# print(f"Correlation: {corr:.3f}")
# print(f"Number of plotted points: {n_points}")


# # =============================================================================
# # Scatterplot 1:
# # Frequentist single true donor log10 LR vs frequentist rework true donor log10 LR
# # with reference line y = x + average_gain
# # =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single_MLE"],
#         sub_df["true_log10_LR_rework_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# # Reference line: y = x + average_gain
# x = df["true_log10_LR_single_MLE"]
# y = df["true_log10_LR_rework_MLE"]

# x_line = np.linspace(x.min(), x.max(), 100)
# y_line = x_line + average_gain

# plt.plot(
#     x_line,
#     y_line,
#     linestyle="--",
#     color="black",
#     linewidth=2,
#     label=f"y = x + average gain = x + {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, frequentist single mixture")
# plt.ylabel("True donor log10 LR, frequentist rework mixture")
# plt.title(
#     f"Frequentist true donor log10 LR: single vs rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, "
#     f"correlation = {corr:.2f}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Scatterplot 2:
# # Frequentist single true donor log10 LR vs gain after rework
# # =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single_MLE"],
#         sub_df["delta_log10_LR_rework_minus_single_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# plt.axhline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axhline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, frequentist single mixture")
# plt.ylabel("Gain in true donor log10 LR: frequentist rework - frequentist single")
# plt.title(
#     f"Frequentist true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Histogram:
# # Frequentist rework - frequentist single
# # =============================================================================

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df["delta_log10_LR_rework_minus_single_MLE"],
#     bins=20,
#     alpha=0.7,
#     edgecolor="black",
# )

# plt.axvline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axvline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("Gain in true donor log10 LR: frequentist rework - frequentist single")
# plt.ylabel("Frequency")
# plt.title(
#     f"Distribution of frequentist true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Summary table per mixture type
# # =============================================================================

# summary_by_type = (
#     df.groupby("mixture_type")["delta_log10_LR_rework_minus_single_MLE"]
#     .agg(["count", "mean", "std", "min", "max"])
#     .reset_index()
# )

# print("\nFrequentist rework - frequentist single per mixture type:")
# print(summary_by_type)










# # -*- coding: utf-8 -*-
# """
# Compare frequentist / MLE true donor log10 LRs for single mixtures and rework mixtures.

# Important:
#     There are three single mixtures for each rework mixture.
#     Therefore the merge must be many-to-one:

#         single rows  -> rework row
#         87 rows      -> 29 rows

# Input:
#     percentile_df_v3.csv = frequentist single-mixture results
#     percentile_df_v1.csv = frequentist rework-mixture results

# Main comparison:
#     x = true donor log10 LR, frequentist single mixture
#     y = true donor log10 LR, frequentist rework mixture

# Difference:
#     gain = rework - single

# Reference line:
#     y = x + average_gain
# """

# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


# # =============================================================================
# # Paths
# # =============================================================================

# CSV_PATH_SINGLE_MLE = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv"
# )

# CSV_PATH_REWORK_MLE = Path(
#     r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MLE_2p\results_v1\percentile_df_v1.csv"
# )


# # =============================================================================
# # Settings
# # =============================================================================

# METHOD_TO_KEEP = "ON"
# CONTRIBUTOR_TO_KEEP = "Unknown2"


# # =============================================================================
# # Read data
# # =============================================================================

# df_single = pd.read_csv(CSV_PATH_SINGLE_MLE)
# df_rework = pd.read_csv(CSV_PATH_REWORK_MLE)

# print("Single MLE columns:")
# print(df_single.columns.tolist())

# print("\nRework MLE columns:")
# print(df_rework.columns.tolist())


# # =============================================================================
# # Merge keys
# # =============================================================================
# # These keys define the rework mixture.
# # Do NOT include the single-mixture replicate / mixture name here,
# # because three single mixtures should point to the same rework mixture.

# merge_cols = [
#     "dataset",
#     "mixture_type",
#     "contributor",
#     "method",
#     "true_donor",
#     "donor_id",
# ]

# print("\nMerge columns:")
# print(merge_cols)


# # =============================================================================
# # Prepare rework dataframe: one row per rework mixture / contributor
# # =============================================================================

# rework_cols = merge_cols + [
#     "true_log10_LR",
#     "mu_mixture",
#     "sigma_mixture",
#     "n_bad_loci",
#     "degenerate_sample",
# ]

# rework_cols = [col for col in rework_cols if col in df_rework.columns]

# df_rework_small = (
#     df_rework[rework_cols]
#     .drop_duplicates(subset=merge_cols)
#     .rename(
#         columns={
#             "true_log10_LR": "true_log10_LR_rework_MLE",
#             "mu_mixture": "mu_rework_MLE",
#             "sigma_mixture": "sigma_rework_MLE",
#             "n_bad_loci": "n_bad_loci_rework_MLE",
#             "degenerate_sample": "degenerate_sample_rework_MLE",
#         }
#     )
# )


# # =============================================================================
# # Check that rework side is unique
# # =============================================================================

# n_dup_rework = df_rework_small.duplicated(subset=merge_cols).sum()
# print(f"\nDuplicated rework rows after drop_duplicates: {n_dup_rework}")

# if n_dup_rework > 0:
#     raise ValueError("Rework dataframe is not unique on merge_cols.")


# # =============================================================================
# # Merge rework values onto all single-mixture rows
# # =============================================================================
# # This is the key correction:
# # many single rows are allowed to match one rework row.

# df = df_single.merge(
#     df_rework_small,
#     on=merge_cols,
#     how="left",
#     validate="many_to_one",
# )

# print(f"\nRows in single MLE file:  {len(df_single)}")
# print(f"Rows in rework MLE file:  {len(df_rework)}")
# print(f"Rows after merge:         {len(df)}")


# # =============================================================================
# # Filter to method and contributor of interest
# # =============================================================================

# df = df[
#     (df["method"] == METHOD_TO_KEEP)
#     & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
# ].copy()

# print(
#     f"\nRows after filtering to method={METHOD_TO_KEEP}, "
#     f"contributor={CONTRIBUTOR_TO_KEEP}: {len(df)}"
# )


# # =============================================================================
# # Rename single-mixture columns
# # =============================================================================

# df = df.rename(
#     columns={
#         "true_log10_LR": "true_log10_LR_single_MLE",
#         "mu_mixture": "mu_single_MLE",
#         "sigma_mixture": "sigma_single_MLE",
#         "n_bad_loci": "n_bad_loci_single_MLE",
#         "degenerate_sample": "degenerate_sample_single_MLE",
#     }
# )


# # =============================================================================
# # Convert numeric columns
# # =============================================================================

# numeric_cols = [
#     "true_log10_LR_single_MLE",
#     "true_log10_LR_rework_MLE",
#     "mu_single_MLE",
#     "mu_rework_MLE",
#     "sigma_single_MLE",
#     "sigma_rework_MLE",
# ]

# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")


# # Remove rows without both true donor LRs
# df = df.dropna(
#     subset=["true_log10_LR_single_MLE", "true_log10_LR_rework_MLE"]
# ).copy()

# print(f"Rows used in plots after dropping missing LR values: {len(df)}")


# # =============================================================================
# # Sanity check: expected number of points
# # =============================================================================

# n_points = len(df)

# print("\nNumber of plotted points:")
# print(n_points)

# print("\nNumber of rows per mixture_type:")
# print(df["mixture_type"].value_counts().sort_index())

# print("\nNumber of single rows per rework key:")
# rows_per_rework = (
#     df.groupby(merge_cols)
#     .size()
#     .reset_index(name="n_single_rows_pointing_to_rework")
# )

# print(rows_per_rework["n_single_rows_pointing_to_rework"].value_counts().sort_index())
# print(rows_per_rework.head())

# # You expect mostly/always 3 here.
# # If you see 1, then you accidentally collapsed the single replicates.
# # If you see more than 3, then the merge keys are too broad.


# # =============================================================================
# # Compute gain
# # =============================================================================

# df["delta_log10_LR_rework_minus_single_MLE"] = (
#     df["true_log10_LR_rework_MLE"] - df["true_log10_LR_single_MLE"]
# )

# if "mu_single_MLE" in df.columns and "mu_rework_MLE" in df.columns:
#     df["mu_delta_rework_minus_single_MLE"] = (
#         df["mu_rework_MLE"] - df["mu_single_MLE"]
#     )

# if "sigma_single_MLE" in df.columns and "sigma_rework_MLE" in df.columns:
#     df["sigma_delta_rework_minus_single_MLE"] = (
#         df["sigma_rework_MLE"] - df["sigma_single_MLE"]
#     )


# # =============================================================================
# # Summary
# # =============================================================================

# corr = df["true_log10_LR_single_MLE"].corr(df["true_log10_LR_rework_MLE"])
# average_gain = df["delta_log10_LR_rework_minus_single_MLE"].mean()

# print("\nSummary: frequentist rework - frequentist single true donor log10 LR")
# print(df["delta_log10_LR_rework_minus_single_MLE"].describe())

# print("\nMean true donor log10 LR:")
# print(f"Frequentist single: {df['true_log10_LR_single_MLE'].mean():.3f}")
# print(f"Frequentist rework: {df['true_log10_LR_rework_MLE'].mean():.3f}")
# print(f"Average gain rework - single: {average_gain:.3f}")
# print(f"Correlation: {corr:.3f}")
# print(f"Number of plotted points: {n_points}")


# # =============================================================================
# # Scatterplot:
# # Frequentist single true donor log10 LR vs frequentist rework true donor log10 LR
# # with reference line y = x + average_gain
# # =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single_MLE"],
#         sub_df["true_log10_LR_rework_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# # Reference line: y = x + average_gain
# x = df["true_log10_LR_single_MLE"]
# x_line = np.linspace(x.min(), x.max(), 100)
# y_line = x_line + average_gain

# plt.plot(
#     x_line,
#     y_line,
#     linestyle="--",
#     color="black",
#     linewidth=2,
#     label=f"y = x + average gain = x + {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, frequentist single mixture")
# plt.ylabel("True donor log10 LR, frequentist rework mixture")
# plt.title(
#     f"Frequentist true donor log10 LR: single vs rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, "
#     f"correlation = {corr:.2f}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Scatterplot:
# # Frequentist single true donor log10 LR vs gain after rework
# # =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single_MLE"],
#         sub_df["delta_log10_LR_rework_minus_single_MLE"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )

# plt.axhline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axhline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("True donor log10 LR, frequentist single mixture")
# plt.ylabel("Gain in true donor log10 LR: frequentist rework - frequentist single")
# plt.title(
#     f"Frequentist true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend(title="Mixture type")
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Histogram:
# # Frequentist rework - frequentist single
# # =============================================================================

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df["delta_log10_LR_rework_minus_single_MLE"],
#     bins=20,
#     alpha=0.7,
#     edgecolor="black",
# )

# plt.axvline(
#     0,
#     linestyle="--",
#     color="black",
#     label="No LR gain",
# )

# plt.axvline(
#     average_gain,
#     linestyle=":",
#     color="black",
#     linewidth=2,
#     label=f"Average gain = {average_gain:.2f}",
# )

# plt.xlabel("Gain in true donor log10 LR: frequentist rework - frequentist single")
# plt.ylabel("Frequency")
# plt.title(
#     f"Distribution of frequentist true donor log10 LR gain after rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
# )

# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # =============================================================================
# # Summary table per mixture type
# # =============================================================================

# summary_by_type = (
#     df.groupby("mixture_type")["delta_log10_LR_rework_minus_single_MLE"]
#     .agg(["count", "mean", "std", "min", "max"])
#     .reset_index()
# )

# print("\nFrequentist rework - frequentist single per mixture type:")
# print(summary_by_type)














# -*- coding: utf-8 -*-
"""
Compare true donor log10 LRs for single mixtures and rework mixtures.

This script supports two approaches:

    APPROACH = "MLE"
        Single mixtures:
            output_single_MLE_2p_v2/results_v3/percentile_df_v3.csv
        Rework mixtures:
            output_rework_MLE_2p/results_v1/percentile_df_v1.csv

    APPROACH = "BAYES"
        Single mixtures:
            output_single_MCMC_2p_corrected_v2/results_v1/percentile_df_v1.csv
        Rework mixtures:
            output_rework_MCMC_2p_corrected_v2/results_v2/percentile_df_v2.csv

Important:
    There are three single mixtures for each rework mixture.

    Therefore the merge must be many-to-one:

        87 single rows  -> 29 rework rows

Main comparison:
    x = true donor log10 LR, single mixture
    y = true donor log10 LR, rework mixture

Difference:
    gain = rework - single

Reference line in scatterplot:
    y = x + average_gain
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# Settings
# =============================================================================

APPROACH = "BAYES"
#APPROACH = "MLE"

# METHOD_TO_KEEP = "ON"
# CONTRIBUTOR_TO_KEEP = "Unknown2"

METHOD_TO_KEEP = "ON"
CONTRIBUTOR_TO_KEEP = "Unknown2"

# Use the same mixture-type symbols as in mixture_proportions_Bayes.py
MIXTURE_TYPE_MARKERS = {
    "A": "o",   # circle
    "B": "X",   # cross
    "C": "s",   # square
    "D": "P",   # plus-filled
    "E": "D",   # diamond
}

# =============================================================================
# Paths
# =============================================================================

if APPROACH == "MLE":
    CSV_PATH_SINGLE = Path(
        r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv"
    )

    CSV_PATH_REWORK = Path(
        r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MLE_2p\results_v1\percentile_df_v1.csv"
    )

    APPROACH_LABEL = "Frequentist / MLE"
    SHORT_LABEL = "MLE"

elif APPROACH == "BAYES":
    CSV_PATH_SINGLE = Path(
        r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MCMC_2p_corrected_v2\results_v1\percentile_df_v1.csv"
    )

    CSV_PATH_REWORK = Path(
        r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MCMC_2p_corrected_v2\results_v2\percentile_df_v2.csv"
    )

    APPROACH_LABEL = "Bayesian / MCMC"
    SHORT_LABEL = "Bayes"

else:
    raise ValueError("APPROACH must be either 'MLE' or 'BAYES'.")


# =============================================================================
# Read data
# =============================================================================

df_single = pd.read_csv(CSV_PATH_SINGLE)
df_rework = pd.read_csv(CSV_PATH_REWORK)

print("=" * 80)
print(f"Running approach: {APPROACH_LABEL}")
print("=" * 80)

print("\nSingle-mixture file:")
print(CSV_PATH_SINGLE)

print("\nRework-mixture file:")
print(CSV_PATH_REWORK)

print("\nSingle columns:")
print(df_single.columns.tolist())

print("\nRework columns:")
print(df_rework.columns.tolist())


# =============================================================================
# Merge keys
# =============================================================================
# These keys define the shared rework mixture / contributor identity.
#
# IMPORTANT:
# Do NOT include:
#   - "mixture"
#   - "replicate"
#   - "replicate_number"
#
# because three single mixtures should point to the same rework mixture.

merge_cols = [
    "dataset",
    "mixture_type",
    "contributor",
    "method",
    "true_donor",
    "donor_id",
]

missing_single_keys = [col for col in merge_cols if col not in df_single.columns]
missing_rework_keys = [col for col in merge_cols if col not in df_rework.columns]

if missing_single_keys:
    raise ValueError(f"Missing merge columns in single file: {missing_single_keys}")

if missing_rework_keys:
    raise ValueError(f"Missing merge columns in rework file: {missing_rework_keys}")

print("\nMerge columns:")
print(merge_cols)


# =============================================================================
# Prepare rework dataframe: one row per rework mixture / contributor
# =============================================================================

rework_cols = merge_cols + [
    "true_log10_LR",
    "mu_mixture",
    "sigma_mixture",
    "n_bad_loci",
    "degenerate_sample",
]

rework_cols = [col for col in rework_cols if col in df_rework.columns]

df_rework_small = (
    df_rework[rework_cols]
    .drop_duplicates(subset=merge_cols)
    .rename(
        columns={
            "true_log10_LR": "true_log10_LR_rework",
            "mu_mixture": "mu_rework",
            "sigma_mixture": "sigma_rework",
            "n_bad_loci": "n_bad_loci_rework",
            "degenerate_sample": "degenerate_sample_rework",
        }
    )
)


# =============================================================================
# Check that rework side is unique
# =============================================================================

n_dup_rework = df_rework_small.duplicated(subset=merge_cols).sum()
print(f"\nDuplicated rework rows after drop_duplicates: {n_dup_rework}")

if n_dup_rework > 0:
    raise ValueError("Rework dataframe is not unique on merge_cols.")


# =============================================================================
# Merge rework values onto all single-mixture rows
# =============================================================================
# This is the important part:
# many single rows are allowed to match one rework row.

df = df_single.merge(
    df_rework_small,
    on=merge_cols,
    how="left",
    validate="many_to_one",
)

print(f"\nRows in single file: {len(df_single)}")
print(f"Rows in rework file: {len(df_rework)}")
print(f"Rows after merge:    {len(df)}")


# =============================================================================
# Filter to method and contributor of interest
# =============================================================================

df = df[
    (df["method"] == METHOD_TO_KEEP)
    & (df["contributor"] == CONTRIBUTOR_TO_KEEP)
].copy()

print(
    f"\nRows after filtering to method={METHOD_TO_KEEP}, "
    f"contributor={CONTRIBUTOR_TO_KEEP}: {len(df)}"
)


# =============================================================================
# Rename single-mixture columns
# =============================================================================

df = df.rename(
    columns={
        "true_log10_LR": "true_log10_LR_single",
        "mu_mixture": "mu_single",
        "sigma_mixture": "sigma_single",
        "n_bad_loci": "n_bad_loci_single",
        "degenerate_sample": "degenerate_sample_single",
    }
)


# =============================================================================
# Convert numeric columns
# =============================================================================

numeric_cols = [
    "true_log10_LR_single",
    "true_log10_LR_rework",
    "mu_single",
    "mu_rework",
    "sigma_single",
    "sigma_rework",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove rows without both true donor LRs
df = df.dropna(
    subset=["true_log10_LR_single", "true_log10_LR_rework"]
).copy()

print(f"Rows used in plots after dropping missing LR values: {len(df)}")


# =============================================================================
# Sanity checks
# =============================================================================

n_points = len(df)

print("\nNumber of plotted points:")
print(n_points)

print("\nNumber of rows per mixture_type:")
print(df["mixture_type"].value_counts().sort_index())

print("\nNumber of single rows per rework key:")
rows_per_rework = (
    df.groupby(merge_cols)
    .size()
    .reset_index(name="n_single_rows_pointing_to_rework")
)

print(rows_per_rework["n_single_rows_pointing_to_rework"].value_counts().sort_index())

if n_points != 87:
    print(
        "\nWARNING: Expected 87 plotted points for Unknown2/method ON, "
        f"but got {n_points}."
    )

if not rows_per_rework["n_single_rows_pointing_to_rework"].eq(3).all():
    print(
        "\nWARNING: Not every rework key has exactly 3 single rows pointing to it."
    )
    print(rows_per_rework.head(20))


# =============================================================================
# Compute gain
# =============================================================================

df["delta_log10_LR_rework_minus_single"] = (
    df["true_log10_LR_rework"] - df["true_log10_LR_single"]
)

if "mu_single" in df.columns and "mu_rework" in df.columns:
    df["mu_delta_rework_minus_single"] = df["mu_rework"] - df["mu_single"]

if "sigma_single" in df.columns and "sigma_rework" in df.columns:
    df["sigma_delta_rework_minus_single"] = (
        df["sigma_rework"] - df["sigma_single"]
    )


# =============================================================================
# Additional figures:
#   1. n_bad_loci vs LR gain after rework
#   2. minor-donor MLE mixture proportion vs LR gain after rework
# =============================================================================

from scipy.stats import pearsonr, spearmanr
import seaborn as sns


# -----------------------------------------------------------------------------
# Read single-mixture MLE mixture proportions from results.json
# -----------------------------------------------------------------------------
# For a 2-person mixture, DNAStatistX stores the first mixture proportion.
# In your setup this is the major donor proportion, so the minor donor proportion
# is 1 - phi_major.

def read_phi_major_from_results_json(results_json_path):
    results_json = pd.read_json(results_json_path)
    h2_results = results_json.loc["H2", "hypothesesResults"]
    model_parameters = h2_results["modelParameters"]
    mixture_proportions = model_parameters["mixtureProportions"]

    return float(mixture_proportions[0])


# Root folder:
# .../output_single_MLE_2p_v2/results_v3/percentile_df_v3.csv
# becomes
# .../output_single_MLE_2p_v2/mixtures
single_mixture_folder = CSV_PATH_SINGLE.parents[1] / "mixtures"

phi_major_by_mixture = {}

for mixture in df["mixture"].dropna().unique():
    results_json_path = single_mixture_folder / str(mixture) / "results_dnax.json"

    if results_json_path.exists():
        phi_major_by_mixture[mixture] = read_phi_major_from_results_json(
            results_json_path
        )
    else:
        print(f"[WARNING] Missing results.json for mixture {mixture}: {results_json_path}")

df["phi_major_single_MLE"] = df["mixture"].map(phi_major_by_mixture)
df["phi_minor_single_MLE"] = 1.0 - df["phi_major_single_MLE"]


# -----------------------------------------------------------------------------
# Generic plotting function with same layout/style as mixture_proportions_Bayes.py
# -----------------------------------------------------------------------------

def plot_gain_against_variable(
    df_plot,
    x_col,
    x_label,
    title,
):
    y_col = "delta_log10_LR_rework_minus_single"

    local_df = df_plot.dropna(subset=[x_col, y_col, "mixture_type"]).copy()

    local_df[x_col] = pd.to_numeric(local_df[x_col], errors="coerce")
    local_df[y_col] = pd.to_numeric(local_df[y_col], errors="coerce")

    local_df = local_df[
        np.isfinite(local_df[x_col])
        & np.isfinite(local_df[y_col])
    ].copy()

    if len(local_df) < 3:
        print(f"[WARNING] Not enough points for {x_col}. n = {len(local_df)}")
        return

    pearson_r, pearson_p = pearsonr(local_df[x_col], local_df[y_col])
    spearman_r, spearman_p = spearmanr(local_df[x_col], local_df[y_col])

    print(f"\nCorrelation: {x_col} vs LR gain")
    print("-" * 80)
    print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.3g}")
    print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}")
    print(f"n = {len(local_df)}")

    plt.figure(figsize=(8, 6))

    ax = sns.scatterplot(
        data=local_df,
        x=x_col,
        y=y_col,
        hue="mixture_type",
        style="mixture_type",
        hue_order=["A", "B", "C", "D", "E"],
        style_order=["A", "B", "C", "D", "E"],
        s=80,
        alpha=0.8,
    )

    # sns.regplot(
    #     data=local_df,
    #     x=x_col,
    #     y=y_col,
    #     scatter=False,
    #     color="black",
    #     line_kws={"linewidth": 2},
    #     ax=ax,
    # )
    
    # Least-squares fit line without confidence band
    slope, intercept = np.polyfit(local_df[x_col], local_df[y_col], 1)
    
    x_line = np.linspace(local_df[x_col].min(), local_df[x_col].max(), 200)
    y_line = slope * x_line + intercept
    
    ax.plot(
        x_line,
        y_line,
        color="black",
        linewidth=2,
        linestyle = 'dashed',
        label=f"Least squares fit: y = {slope:.2f}x + {intercept:.2f}",
    )

    # Add regression line to legend
    #ax.lines[-1].set_label("Least squares fit")

    plt.axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.7,
    )

    plt.xlabel(x_label)
    plt.ylabel("Increase in true minor-donor log10 LR due to rework")
    # plt.title(
    #     f"{title}\n"
    #     f"Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}, n = {len(local_df)}"
    # )

    plt.grid(alpha=0.3)
    plt.legend(title="Mixture type")
    plt.tight_layout()
    plt.show()


# # -----------------------------------------------------------------------------
# # Figure 1: n_bad_loci vs increase in true minor-donor log10 LR
# # -----------------------------------------------------------------------------

# plot_gain_against_variable(
#     df_plot=df,
#     x_col="n_bad_loci_single",
#     x_label="Number of loci with dropout in the single mixture",
#     title=(
#         f"{APPROACH_LABEL}: dropout loci vs LR gain after rework\n"
#         f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}"
#     ),
# )


# # -----------------------------------------------------------------------------
# # Figure 2: minor-donor MLE mixture proportion vs increase in true minor-donor log10 LR
# # -----------------------------------------------------------------------------

# plot_gain_against_variable(
#     df_plot=df,
#     x_col="phi_minor_single_MLE",
#     x_label="Minor-donor MLE mixture proportion in the single mixture",
#     title=(
#         f"{APPROACH_LABEL}: minor-donor mixture proportion vs LR gain after rework\n"
#         f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}"
#     ),
# )


# =============================================================================
# Summary
# =============================================================================

corr = df["true_log10_LR_single"].corr(df["true_log10_LR_rework"])
average_gain = df["delta_log10_LR_rework_minus_single"].mean()

print("\nSummary: rework - single true donor log10 LR")
print(df["delta_log10_LR_rework_minus_single"].describe())

print("\nMean true donor log10 LR:")
print(f"{APPROACH_LABEL} single: {df['true_log10_LR_single'].mean():.3f}")
print(f"{APPROACH_LABEL} rework: {df['true_log10_LR_rework'].mean():.3f}")
print(f"Average gain rework - single: {average_gain:.3f}")
print(f"Correlation: {corr:.3f}")
print(f"Number of plotted points: {n_points}")


# =============================================================================
# Scatterplot 1:
# Single true donor log10 LR vs rework true donor log10 LR
# with reference line y = x + average_gain
# =============================================================================

# plt.figure(figsize=(8, 7))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["true_log10_LR_rework"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )
    
# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["true_log10_LR_rework"],
#         alpha=0.75,
#         edgecolor="black",
#         marker=MIXTURE_TYPE_MARKERS.get(mixture_type, "o"),
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )
    
plt.figure(figsize=(8, 7))

# sns.scatterplot(
#     data=df,
#     x="true_log10_LR_single",
#     y="true_log10_LR_rework",
#     hue="mixture_type",
#     style="mixture_type",
#     hue_order=["A", "B", "C", "D", "E"],
#     style_order=["A", "B", "C", "D", "E"],
#     s=80,
#     alpha=0.8,
# )

sns.scatterplot(
    data=df,
    x="true_log10_LR_single",
    y="true_log10_LR_rework",
    hue="mixture_type",
    style="mixture_type",
    markers=MIXTURE_TYPE_MARKERS,
    hue_order=["A", "B", "C", "D", "E"],
    style_order=["A", "B", "C", "D", "E"],
    s=80,
    alpha=0.8,
)

# Reference line: y = x + average_gain
x = df["true_log10_LR_single"]

x_line = np.linspace(x.min(), x.max(), 100)
y_line = x_line + average_gain

plt.plot(
    x_line,
    y_line,
    linestyle="--",
    color="black",
    linewidth=2,
    label=f"y = x + average gain = x + {average_gain:.2f}",
)

plt.xlabel(f"True minor donor log10 LR, {APPROACH_LABEL} single mixture")
plt.ylabel(f"True minor donor log10 LR, {APPROACH_LABEL} rework mixture")
# plt.title(
#     f"{APPROACH_LABEL} true donor log10 LR: single vs rework\n"
#     f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, "
#     f"correlation = {corr:.2f}, n = {n_points}"
# )

plt.grid(alpha=0.3)
plt.legend(title="Mixture type")
plt.tight_layout()
plt.show()


# =============================================================================
# Scatterplot 2:
# Single true donor log10 LR vs gain after rework
# =============================================================================

# plt.figure(figsize=(8, 6))

# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["delta_log10_LR_rework_minus_single"],
#         alpha=0.75,
#         edgecolor="black",
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )
    
# for mixture_type, sub_df in df.groupby("mixture_type"):
#     plt.scatter(
#         sub_df["true_log10_LR_single"],
#         sub_df["delta_log10_LR_rework_minus_single"],
#         alpha=0.75,
#         edgecolor="black",
#         marker=MIXTURE_TYPE_MARKERS.get(mixture_type, "o"),
#         label=f"Mixture type {mixture_type} (n={len(sub_df)})",
#     )
    
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="true_log10_LR_single",
    y="delta_log10_LR_rework_minus_single",
    hue="mixture_type",
    style="mixture_type",
    hue_order=["A", "B", "C", "D", "E"],
    style_order=["A", "B", "C", "D", "E"],
    s=80,
    alpha=0.8,
)

plt.axhline(
    0,
    linestyle="--",
    color="black",
    label="No LR gain",
)

plt.axhline(
    average_gain,
    linestyle=":",
    color="black",
    linewidth=2,
    label=f"Average gain = {average_gain:.2f}",
)

plt.xlabel(f"True donor log10 LR, {APPROACH_LABEL} single mixture")
plt.ylabel(f"Gain in true donor log10 LR: {APPROACH_LABEL} rework - single")
plt.title(
    f"{APPROACH_LABEL} true donor log10 LR gain after rework\n"
    f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
)

plt.grid(alpha=0.3)
plt.legend(title="Mixture type")
plt.tight_layout()
plt.show()


# =============================================================================
# Histogram:
# Rework - single
# =============================================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["delta_log10_LR_rework_minus_single"],
    bins=20,
    alpha=0.7,
    edgecolor="black",
)

plt.axvline(
    0,
    linestyle="--",
    color="black",
    label="No LR gain",
)

plt.axvline(
    average_gain,
    linestyle=":",
    color="black",
    linewidth=2,
    label=f"Average gain = {average_gain:.2f}",
)

plt.xlabel(f"Gain in true donor log10 LR: {APPROACH_LABEL} rework - single")
plt.ylabel("Frequency")
plt.title(
    f"Distribution of {APPROACH_LABEL} true donor log10 LR gain after rework\n"
    f"{CONTRIBUTOR_TO_KEEP}, method {METHOD_TO_KEEP}, n = {n_points}"
)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# Summary table per mixture type
# =============================================================================

summary_by_type = (
    df.groupby("mixture_type")["delta_log10_LR_rework_minus_single"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)

print(f"\n{APPROACH_LABEL} rework - single per mixture type:")
print(summary_by_type)
