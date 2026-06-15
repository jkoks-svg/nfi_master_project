# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 08:47:28 2026

@author: jortk
"""


from pathlib import Path
#from decimal import Decimal, InvalidOperation

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# PATHS
# ----------------------------
#BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin")
#version_org = 'v1'
#version_traces = 'v1'
CSV_PATH_ORG = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_single_MLE_2p_v2\results_v3\percentile_df_v3.csv")
#BASE / "output_HT_2p_nodropin_runF" /f"results_{version_org}" / f"percentile_df_{version_org}.csv" 
CSV_PATH_TRACES = Path(r"C:\Users\jortk\Documents\inputs\input_2p_Bayes\output_rework_MLE_2p\results_v1\percentile_df_v1.csv")
#BASE / "output_HT_2p_nodropin_runF_traces" /f"results_{version_traces}" / f"percentile_df_{version_traces}.csv" 

percentile_df_org = pd.read_csv(CSV_PATH_ORG)
percentile_df_traces = pd.read_csv(CSV_PATH_TRACES)

trace_params = (
    percentile_df_traces[
        [
            "dataset",
            "mixture_type",
            "contributor",
            "method",
            "true_donor",
            'donor_id',
            "mu_mixture",
            "sigma_mixture",
            "true_log10_LR"
        ]
    ]
    .drop_duplicates()
    .rename(columns={
        "mu_mixture": "mu_trace",
        "sigma_mixture": "sigma_trace",
        "true_log10_LR":"true_log10_LR_trace"
    })
)

percentile_df_org = percentile_df_org.merge(
    trace_params,
    on=["dataset", "mixture_type", "contributor", "method", 'true_donor', 'donor_id'],
    how="left",
)

df = percentile_df_org[(percentile_df_org['method']=='ON')&(percentile_df_org['contributor']=='Unknown2')]
df=df.copy()
df.loc[:,'mu_delta'] = df.loc[:,'mu_trace']-df.loc[:,'mu_mixture']
df.loc[:,'sigma_delta'] = df.loc[:,'sigma_trace']-df.loc[:, 'sigma_mixture']
df.loc[:, 'true_delta'] = df.loc[:,'true_log10_LR_trace']-df.loc[:,'true_log10_LR']

mu_mixture_avg = df['mu_mixture'].mean()
mu_trace_avg = df['mu_trace'].mean()
mu_delta_avg = df['mu_delta'].mean()
sigma_delta_avg = df['sigma_delta'].mean()
true_log_LR_avg = df['true_log10_LR'].mean()
true_log_LR_trace_avg = df['true_log10_LR_trace'].mean()
true_delta_avg = df['true_delta'].mean()

plt.figure(figsize=(12, 6))
plt.hist(df['true_log10_LR'],bins=20, alpha=0.6, label='Single mixtures', color='skyblue', edgecolor='black')
#plt.hist(df['true_log10_LR_trace'],bins=20, alpha=0.6, label='Rework mixtures', color='salmon', edgecolor='black')
plt.axvline(true_log_LR_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average logLR = {true_log_LR_avg:.2f}')
#plt.axvline(true_log_LR_trace_avg, color='red', linestyle='dashed', linewidth=2, label=f'true_log_LR_trace_avg = {true_log_LR_trace_avg:.2f}')
plt.xlabel('True donor logLR')
plt.ylabel('Frequency')
#plt.title('Histogram of true donor logLR for single and rework mixtures')
plt.grid()
plt.legend()
plt.show()

df_rep1 = df[df['replicate']==1]

plt.figure(figsize=(12, 6))
#plt.hist(df['true_log10_LR'],bins=20, alpha=0.6, label='Single mixtures', color='skyblue', edgecolor='black')
plt.hist(df_rep1['true_log10_LR_trace'],bins=20, alpha=0.6, label='Rework mixtures', color='skyblue', edgecolor='black')
#plt.axvline(true_log_LR_avg, color='blue', linestyle='dashed', linewidth=2, label=f'true_log_LR_avg = {true_log_LR_avg:.2f}')
plt.axvline(true_log_LR_trace_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average logLR = {true_log_LR_trace_avg:.2f}')
plt.xlabel('True donor logLR')
plt.ylabel('Frequency')
#plt.title('Histogram of true donor logLR for single and rework mixtures')
plt.grid()
plt.legend()
plt.show()


# plt.figure(figsize=(12, 6))
# plt.hist(df['mu_mixture'],bins=20, alpha=0.6, label='mu_mixture', color='skyblue', edgecolor='black')
# plt.hist(df['mu_trace'],bins=20, alpha=0.6, label='mu_trace', color='salmon', edgecolor='black')
# plt.axvline(mu_mixture_avg, color='blue', linestyle='dashed', linewidth=2, label=f'mu_mixture avg = {mu_mixture_avg:.2f}')
# plt.axvline(mu_trace_avg, color='red', linestyle='dashed', linewidth=2, label=f'mu_trace avg = {mu_trace_avg:.2f}')
# plt.xlabel('Increase in logLR of the true donor due to rework')
# plt.ylabel('Frequency')
# plt.title('Histogram of mu_mixture and mu_trace')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.hist(df['mu_delta'],bins=20, alpha=0.6, label='mu_delta', color='skyblue', edgecolor='black')
# plt.axvline(mu_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'mu_delta avg = {mu_delta_avg:.2f}')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of mu_delta')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.hist(df['sigma_delta'],bins=20, alpha=0.6, label='sigma_delta', color='skyblue', edgecolor='black')
# plt.axvline(sigma_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'sigma_delta avg = {sigma_delta_avg:.2f}')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of sigma_delta')
# plt.grid()
# plt.legend()
# plt.show()

plt.figure(figsize=(12, 6))
plt.hist(df['true_delta'],bins=20, alpha=0.6, color='skyblue', edgecolor='black') # label='Increase in logLR'
plt.axvline(true_delta_avg, color='blue', linestyle='dashed', linewidth=2, label=f'Average increase = {true_delta_avg:.2f}')
plt.xlabel('Increase in logLR of the true donor due to rework')
plt.ylabel('Frequency')
#plt.title('Histogram of true donor logLR increase due to rework')
plt.grid()
plt.legend()
plt.show()

mixture_types = ['A','B','C','D','E']
for mixture_type in mixture_types:
    df_type = df[df['mixture_type']==mixture_type]
    true_delta_avg_type = df_type['true_delta'].mean()
    plt.figure(figsize=(12, 6))
    plt.hist(df_type['true_delta'],bins=20, alpha=0.6, label='mu_delta', color='skyblue', edgecolor='black')
    plt.axvline(true_delta_avg_type, color='blue', linestyle='dashed', linewidth=2, label=f'true_delta avg {mixture_type} = {true_delta_avg_type:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of mu_delta {mixture_type}')
    plt.grid()
    plt.legend()
    plt.show()
    sigma_delta_avg_type = df_type['sigma_delta'].mean()
    plt.figure(figsize=(12, 6))
    plt.hist(df_type['sigma_delta'],bins=20, alpha=0.6, label='sigma_delta', color='skyblue', edgecolor='black')
    plt.axvline(sigma_delta_avg_type, color='blue', linestyle='dashed', linewidth=2, label=f'sigma_delta avg {mixture_type} = {sigma_delta_avg_type:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of sigma_delta {mixture_type}')
    plt.grid()
    plt.legend()
    plt.show()

# # Compute correlation
# corr = df["n_bad_loci"].corr(df["mu_delta"])
# print(f"Correlation between n_bad_loci and mu_delta: {corr:.4f}")

# # Plot
# plt.figure(figsize=(6, 4))
# plt.scatter(df["n_bad_loci"], df["mu_delta"], color='teal', edgecolor='k', s=70)
# plt.title("Correlation between n_bad_loci and mu_delta", fontsize=14)
# plt.xlabel("n_bad_loci", fontsize=12)
# plt.ylabel("mu_delta", fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.6)

# # Annotate correlation value
# plt.text(
#     0.05, 0.95, f"r = {corr:.4f}",
#     transform=plt.gca().transAxes,
#     fontsize=12, color="darkred", ha="left", va="top"
# )

# plt.tight_layout()
# plt.show()

# --- Extract variables ---
x = df["n_bad_loci"]
y = df['true_delta'] #df["mu_delta"]

# --- Compute correlation ---
corr = np.corrcoef(x, y)[0, 1]
print(f"Correlation between n_bad_loci and mu_delta: {corr:.4f}")

# --- Fit regression line ---
slope, intercept = np.polyfit(x, y, 1)
print('slope = ', slope)
y_pred = slope * x + intercept

# --- Plot ---
plt.figure(figsize=(7, 5))
plt.scatter(x, y, color="teal", edgecolor="black", s=70, label="Single mixtures")
plt.plot(x, y_pred, color="darkred", linewidth=2, label="Least squares fit") #f"delta_logLR = {slope:.2f}*N_loci_dropout + {intercept:.2f}")

#plt.title("Number of loci with dropout vs gain in LR due to rework", fontsize=14)
plt.xlabel("Number of loci with dropout", fontsize=12)
plt.ylabel("Increase in logLR of the true donor due to rework", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# # Show correlation on the plot
# plt.text(0.05, 0.95, f"r = {corr:.4f}", transform=plt.gca().transAxes,
#          fontsize=12, color="darkred", ha="left", va="top")

plt.tight_layout()
plt.show()

pivot = pd.pivot_table(df, 
                       values="n_bad_loci", 
                       index="mixture_type", 
                       aggfunc="mean")

print(pivot)









