# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:15:41 2026

@author: jortk
"""

from pathlib import Path
import pandas as pd
import numpy as np

def parse2(s):
    rep, rest = s.split('_')
    dataset_number = int(rest[0])
    mixture_type = rest[1]
    n_contributors = int(rest[2])
    return int(rep), dataset_number, mixture_type, n_contributors,rest

def json2params(results_json):
    H2_results = results_json.loc["H2", "hypothesesResults"]
    modelParameters = H2_results['modelParameters']
    #ROOT = mixture_folder / mixture
    mx = modelParameters['mixtureProportions'][0]
    eph = modelParameters["expectedPeakHeight"]
    phv = modelParameters["peakHeightVariance"]
    deg = modelParameters["degradationSlope"]
    return mx,eph,phv, deg
    

def check_mixture_proportions(
    mixture_folder,
    traces_folder,
):
    """
    Generate synthetic mixtures using DNAStatistX.

    Parameters
    ----------
    mixture_folder : Path
        Folder containing mixture directories.
    base_path : Path
        Root project directory.
    test_mode : str
        One of ['test0','test1','test2','test3'] controlling MX1 sampling.
    """

    mixture_folder = Path(mixture_folder)

    mixture_list = [item.name for item in mixture_folder.iterdir() if item.is_dir()]
    
    mixture_props_dicts = []
    
    for mixture in mixture_list:
        mixture_props = {}
        
        results_json = pd.read_json(mixture_folder / f"{mixture}/results.json")
        # H2_results = results_json.loc["H2", "hypothesesResults"]
        # modelParameters = H2_results['modelParameters']
        # #ROOT = mixture_folder / mixture
        # mx = modelParameters['mixtureProportions'][0]
        # eph = modelParameters["expectedPeakHeight"]
        # phv = modelParameters["peakHeightVariance"]
        mx,eph,phv, deg = json2params(results_json)
        
        mixture_props['mixture'] = mixture
        parsed_mixture = parse2(mixture)
        mixture_props['replicate_number'] = parsed_mixture[0]
        mixture_props['dataset_number'] = parsed_mixture[1]
        mixture_props['mixture_type'] = parsed_mixture[2]
        mixture_props['n_contributors'] = parsed_mixture[3]
        rest = parsed_mixture[4]
        
        mixture_props['phi_major_dnax'] = mx
        mixture_props['expectedPeakHeight'] = eph
        mixture_props['peakHeightVariance'] = phv
        mixture_props['degradationSlope'] =deg
        
        key = (
            mixture_props['n_contributors'],
            mixture_props['mixture_type'],
            "Unknown1"  # assuming major contributor
        )
        
        mixture_props['phi_major_theor'] = MIXTURE_TYPE_MAP.get(key)
        mixture_props['diff'] = mixture_props['phi_major_theor'] - mixture_props['phi_major_dnax']
        mixture_props['diff_absolute'] = np.abs(mixture_props['diff'])
        
        results_json_combined = pd.read_json(traces_folder / f"{rest}/results.json")
        mx_combined,eph_combined,phv_combined, deg_combined = json2params(results_json_combined)
        mixture_props['phi_major_dnax_combined'] = mx_combined
        mixture_props['expectedPeakHeight_combined'] = eph_combined
        mixture_props['peakHeightVariance_combined'] = phv_combined
        mixture_props['degradationSlope_combined'] = deg_combined
        mixture_props['diff_combined'] = mixture_props['phi_major_theor'] - mixture_props['phi_major_dnax_combined']
        mixture_props['diff_absolute_combined'] = np.abs(mixture_props['diff_combined'])
        
        #mixture_props['phi_major_theor'] = 0.01
        mixture_props_dicts.append(mixture_props)
    
    return mixture_props_dicts
        
MIXTURE_TYPE_MAP = {
    (2, 'A',"Unknown1") : 300/450,
    (2, 'A',"Unknown2") : 150/450,
    (2, 'B',"Unknown1") : 300/330,
    (2, 'B',"Unknown2") : 30/330,
    (2, 'C',"Unknown1") : 150/300,
    (2, 'C',"Unknown2") : 150/300,
    (2, 'D',"Unknown1") : 150/180,
    (2, 'D',"Unknown2") : 30/180,
    (2, 'E',"Unknown1") : 600/630,
    (2, 'E',"Unknown2") : 30/630,
    (3, 'A',"Unknown1") : 300/600,
    (3, 'A',"Unknown2") : 150/600,
    (3, 'A',"Unknown3") : 150/600,
    (3, 'B',"Unknown1") : 300/360,
    (3, 'B',"Unknown2") : 30/360,
    (3, 'B',"Unknown3") : 30/360,
    (3, 'C',"Unknown1") : 150/360,
    (3, 'C',"Unknown2") : 150/360,
    (3, 'C',"Unknown3") : 60/360,
    (3, 'D',"Unknown1") : 150/240,
    (3, 'D',"Unknown2") : 30/240,
    (3, 'D',"Unknown3") : 60/240,
    (3, 'E',"Unknown1") : 600/690,
    (3, 'E',"Unknown2") : 30/690,
    (3, 'E',"Unknown3") : 60/690,
    }

            
# def main():

mixture_folder = Path(r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_runF_Qallele_fixed_100samples\mixtures")
#Path(r"C:\Users\jortk\Documents\input_HT_2p_nodropin\output_runF_test3\mixtures")

traces_folder = Path(r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces\mixtures") 
#Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_HT_2p_nodropin\output_HT_2p_nodropin_runF_traces\mixtures")


mixture_props_dicts = check_mixture_proportions(
    mixture_folder=mixture_folder,
    traces_folder = traces_folder,
)

df = pd.DataFrame(mixture_props_dicts)
dfA = df[(df['mixture_type']=='A')]
dfB = df[(df['mixture_type']=='B')]
dfC = df[df['mixture_type']=='C']
dfA_236 = df[(df['mixture_type']=='A') & (df['dataset_number'].isin([2,3,6]))]


# =============================================================================
# Scatterplot: diff_absolute vs interval_score_95
# =============================================================================

from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# This is the percentile_df that was saved by your read_csv script
percentile_path = Path(
    r"C:\Users\jortk\Documents\inputs\input_HT_2p_nodropin\output_runF_Qallele_fixed_100samples\results_v1\percentile_df_v1.csv"
)

percentile_df = pd.read_csv(percentile_path)

# Do NOT change percentile_df["mixture"].
# Instead create a temporary key:
# mixture = 1A2 and replicate_number = 1 becomes mixture_key = 1_1A2
percentile_df["mixture_key"] = (
    percentile_df["replicate_number"].astype(str)
    + "_"
    + percentile_df["mixture"].astype(str)
)

# Merge interval_score_95 from percentile_df with diff_absolute from df
merged_df = percentile_df.merge(
    df[[
        "mixture",
        "diff_absolute",
        "diff",
        "phi_major_dnax",
        "phi_major_theor",
        "diff_absolute_combined",
        "phi_major_dnax_combined",
    ]],
    left_on="mixture_key",
    right_on="mixture",
    how="left",
    suffixes=("_percentile", "_props"),
)

# # Convert columns to numeric
# merged_df["diff_absolute"] = pd.to_numeric(
#     merged_df["diff_absolute"],
#     errors="coerce"
# )

# merged_df["interval_score_95"] = pd.to_numeric(
#     merged_df["interval_score_95"],
#     errors="coerce"
# )

# Convert columns to numeric
numeric_cols = [
    "diff_absolute",
    "diff_absolute_combined",
    "phi_major_dnax",
    "phi_major_dnax_combined",
    "interval_score_95",
]

for col in numeric_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

# Difference between combined and single-replicate estimated major mixture proportion
merged_df["abs_phi_combined_minus_single"] = np.abs(
    merged_df["phi_major_dnax_combined"] - merged_df["phi_major_dnax"]
)



# # Keep only rows where both values exist
# plot_df = merged_df.dropna(
#     subset=["diff_absolute", "interval_score_95"]
# ).copy()

# # Safety check
# print("\nScatterplot data check")
# print("-" * 80)
# print(f"Rows in percentile_df: {len(percentile_df)}")
# print(f"Rows after merge:      {len(merged_df)}")
# print(f"Rows used in plot:     {len(plot_df)}")

# # # Correlations
# # pearson_r, pearson_p = pearsonr(
# #     plot_df["diff_absolute"],
# #     plot_df["interval_score_95"]
# # )

# # spearman_r, spearman_p = spearmanr(
# #     plot_df["diff_absolute"],
# #     plot_df["interval_score_95"]
# # )

# # print("\nCorrelation between diff_absolute and interval_score_95")
# # print("-" * 80)
# # print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.3g}")
# # print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}")

# # # Scatterplot
# # plt.figure(figsize=(8, 6))

# # sns.scatterplot(
# #     data=plot_df,
# #     x="diff_absolute",
# #     y="interval_score_95",
# #     hue="contributor",
# #     style="mixture_type",
# #     s=80,
# #     alpha=0.8,
# # )

# # sns.regplot(
# #     data=plot_df,
# #     x="diff_absolute",
# #     y="interval_score_95",
# #     scatter=False,
# #     color="black",
# # )

# # plt.xlabel("Absolute error in estimated major mixture proportion")
# # plt.ylabel("95% interval score")
# # plt.title(
# #     "diff_absolute vs interval_score_95\n"
# #     f"Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}"
# # )

# # plt.grid(alpha=0.3)
# # plt.tight_layout()
# # plt.show()

# # Keep only Unknown2
# plot_df_u2 = plot_df[plot_df["contributor"] == "Unknown2"].copy()

# # Remove missing / non-finite values
# plot_df_u2 = plot_df_u2[
#     np.isfinite(plot_df_u2["diff_absolute"]) &
#     np.isfinite(plot_df_u2["interval_score_95"])
# ]

# pearson_r, pearson_p = pearsonr(
#     plot_df_u2["diff_absolute"],
#     plot_df_u2["interval_score_95"]
# )

# spearman_r, spearman_p = spearmanr(
#     plot_df_u2["diff_absolute"],
#     plot_df_u2["interval_score_95"]
# )

# print("\nUnknown2: correlation between diff_absolute and interval_score_95")
# print("-" * 80)
# print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.3g}")
# print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}")
# print(f"n = {len(plot_df_u2)}")

# plt.figure(figsize=(8, 6))

# sns.scatterplot(
#     data=plot_df_u2,
#     x="diff_absolute",
#     y="interval_score_95",
#     hue="mixture_type",
#     style="mixture_type",
#     s=80,
#     alpha=0.8,
# )

# sns.regplot(
#     data=plot_df_u2,
#     x="diff_absolute",
#     y="interval_score_95",
#     scatter=False,
#     color="black",
# )

# plt.xlabel("Absolute error in estimated major mixture proportion")
# plt.ylabel("95% interval score")
# plt.title(
#     "Unknown2: diff_absolute vs interval_score_95\n"
#     f"Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}"
# )

# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8, 6))

# sns.scatterplot(
#     data=plot_df_u2,
#     x="diff_absolute",
#     y="interval_score_95",
#     hue="mixture_type",
#     style="mixture_type",
#     s=80,
#     alpha=0.8,
# )

# plt.yscale("symlog", linthresh=1)

# plt.xlabel("Absolute error in estimated major mixture proportion")
# plt.ylabel("95% interval score, symlog scale")
# plt.title(
#     "Unknown2: diff_absolute vs interval_score_95, log-like y-axis\n"
#     f"Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}"
# )

# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()


# Keep only Unknown2
plot_df = merged_df[merged_df["contributor"] == "Unknown2"].copy()

# def plot_interval_score_comparison(
#     df_plot,
#     x_col,
#     x_label,
#     title_prefix="Unknown2",
#     use_symlog=False,
# ):
#     needed = [x_col, "interval_score_95"]
#     local_df = df_plot.dropna(subset=needed).copy()

#     local_df = local_df[
#         np.isfinite(local_df[x_col]) &
#         np.isfinite(local_df["interval_score_95"])
#     ]

#     if len(local_df) < 3:
#         print(f"[WARNING] Not enough rows to plot {x_col}. n = {len(local_df)}")
#         return

#     pearson_r, pearson_p = pearsonr(
#         local_df[x_col],
#         local_df["interval_score_95"]
#     )

#     spearman_r, spearman_p = spearmanr(
#         local_df[x_col],
#         local_df["interval_score_95"]
#     )

#     print(f"\n{title_prefix}: correlation between {x_col} and interval_score_95")
#     print("-" * 80)
#     print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.3g}")
#     print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}")
#     print(f"n = {len(local_df)}")

#     plt.figure(figsize=(8, 6))

#     sns.scatterplot(
#         data=local_df,
#         x=x_col,
#         y="interval_score_95",
#         hue="mixture_type",
#         style="mixture_type",
#         s=80,
#         alpha=0.8,
#     )

#     if not use_symlog:
#         sns.regplot(
#             data=local_df,
#             x=x_col,
#             y="interval_score_95",
#             scatter=False,
#             color="black",
#         )

#     if use_symlog:
#         plt.yscale("symlog", linthresh=1)
#         y_label = "95% interval score, symlog scale"
#         title_suffix = "symlog y-axis"
#     else:
#         y_label = "95% interval score"
#         title_suffix = "linear y-axis"

#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(
#         f"{title_prefix}: {x_col} vs interval_score_95\n"
#         f"{title_suffix}; Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}"
#     )

#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

def plot_interval_score_comparison(
    df_plot,
    x_col,
    x_label,
    title_prefix="Unknown2",
    use_log1p_y=True,
):
    needed = [x_col, "interval_score_95"]
    local_df = df_plot.dropna(subset=needed).copy()

    local_df = local_df[
        np.isfinite(local_df[x_col]) &
        np.isfinite(local_df["interval_score_95"])
    ]

    if len(local_df) < 3:
        print(f"[WARNING] Not enough rows to plot {x_col}. n = {len(local_df)}")
        return

    # NEW: log-transform the interval score
    # This keeps zeros valid: log1p(0) = 0
    # local_df["log1p_interval_score_95"] = np.log1p(
    #     local_df["interval_score_95"]
    # )
    local_df["log1p_interval_score_95"] = np.log10(
        1 + local_df["interval_score_95"]
    )

    if use_log1p_y:
        y_col = "log1p_interval_score_95"
        y_label = "log_10(1 + 95% interval score)"
        scale_text = "log_10_1p-transformed y"
    else:
        y_col = "interval_score_95"
        y_label = "95% interval score"
        scale_text = "linear y"

    pearson_r, pearson_p = pearsonr(
        local_df[x_col],
        local_df[y_col]
    )

    spearman_r, spearman_p = spearmanr(
        local_df[x_col],
        local_df[y_col]
    )

    print(f"\n{title_prefix}: correlation between {x_col} and {y_col}")
    print("-" * 80)
    print(f"Pearson r  = {pearson_r:.3f}, p = {pearson_p:.3g}")
    print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}")
    print(f"n = {len(local_df)}")
    
    
    # Least-squares fit: y = intercept + slope * x
    x = local_df[x_col].to_numpy()
    y = local_df[y_col].to_numpy()
    
    slope, intercept = np.polyfit(x, y, deg=1)
    
    fit_text = f"Least-squares fit: y = {intercept:.3f} + {slope:.3f}x"
    print(fit_text)
        

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=local_df,
        x=x_col,
        y=y_col,
        hue="mixture_type",
        style="mixture_type",
        s=80,
        alpha=0.8,
    )

    # sns.regplot(
    #     data=local_df,
    #     x=x_col,
    #     y=y_col,
    #     scatter=False,
    #     color="black",
    # )
    
    ax = plt.gca()

    sns.regplot(
        data=local_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color="black",
        line_kws={"linestyle": "--"},  # dashed black line
        ci=None,   # removes the grey confidence interval
        ax=ax,
    )

    ax.text(
        0.03,
        0.97,
        fit_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(
    #     f"{title_prefix}: {x_col} vs {y_col}\n"
    #     f"{scale_text}; Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}"
    # )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# # 1. interval_score_95 vs diff_absolute
# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="diff_absolute",
#     x_label="Absolute error in estimated major mixture proportion",
# )

# # 2. interval_score_95 vs diff_absolute_combined
# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="diff_absolute_combined",
#     x_label="Absolute error in combined estimated major mixture proportion",
# )

# # 3. interval_score_95 vs abs(phi_major_dnax_combined - phi_major_dnax)
# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="abs_phi_combined_minus_single",
#     x_label="Absolute difference between combined and single-replicate major mixture proportion",
# )

# # Optional: log-like y-axis versions
# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="diff_absolute",
#     x_label="Absolute error in estimated major mixture proportion",
#     use_symlog=True,
# )

# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="diff_absolute_combined",
#     x_label="Absolute error in combined estimated major mixture proportion",
#     use_symlog=True,
# )

# plot_interval_score_comparison(
#     df_plot=plot_df,
#     x_col="abs_phi_combined_minus_single",
#     x_label="Absolute difference between combined and single-replicate major mixture proportion",
#     use_symlog=True,
# )

# 1. log(1 + interval_score_95) vs diff_absolute
plot_interval_score_comparison(
    df_plot=plot_df,
    x_col="diff_absolute",
    x_label="Absolute error in estimated major mixture proportion",
    use_log1p_y=True,
)

# 2. log(1 + interval_score_95) vs diff_absolute_combined
plot_interval_score_comparison(
    df_plot=plot_df,
    x_col="diff_absolute_combined",
    x_label="Absolute error in combined estimated major mixture proportion",
    use_log1p_y=True,
)

# 3. log(1 + interval_score_95) vs abs(phi_major_dnax_combined - phi_major_dnax)
plot_interval_score_comparison(
    df_plot=plot_df,
    x_col="abs_phi_combined_minus_single",
    x_label="Absolute difference between rework- and single-mixture MLE of minor donor mixture proportion",
    use_log1p_y=True,
)




#import matplotlib.pyplot as plt

# Sort dataframe
subset = df[df["diff_absolute"] > 0.03]
df_sorted = subset.sort_values(by="diff_absolute", ascending=False)


# Plot
plt.figure()
plt.bar(df_sorted["mixture"], df_sorted["diff_absolute"])

plt.xticks(rotation=45)
plt.xlabel("Mixture")
plt.ylabel("Absolute Difference")
plt.title("Absolute Differences (Highest to Lowest)")

plt.tight_layout()
plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Columns you want to correlate
# cols = [
#     "phi_major_dnax",
#     "expectedPeakHeight",
#     "peakHeightVariance",
#     "degradationSlope"
# ]

# # df = df[(df['mixture']=='1_1A2')
# #         |(df['mixture']=='2_1A2')
# #         |(df['mixture']=='3_1A2')]


# # Make sure the columns are numeric
# df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# # Correlation matrix
# corr_matrix = df[cols].corr(method="pearson")

# print(corr_matrix)

# plt.figure(figsize=(7, 5))

# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     cmap="coolwarm",
#     vmin=-1,
#     vmax=1,
#     center=0,
#     square=True
# )

# plt.title("Correlation matrix")
# plt.tight_layout()
# plt.show()

# spearman_corr = df[cols].corr(method="spearman")
# print(spearman_corr)


# if __name__ == "__main__":
#     main()


# =============================================================================
# Within-underlying-mixture deviation analysis
# =============================================================================

import seaborn as sns
import matplotlib.pyplot as plt

# Parameters to compare
param_cols = [
    "phi_major_dnax",
    "expectedPeakHeight",
    "peakHeightVariance",
    "degradationSlope"
]

# Make sure these are numeric
df[param_cols] = df[param_cols].apply(pd.to_numeric, errors="coerce")

# Define the underlying mixture.
# Example:
# 1_1A2, 2_1A2, 3_1A2 should be grouped together.
group_cols = [
    "dataset_number",
    "mixture_type",
    "n_contributors"
]

# Create a readable label for plotting
df["underlying_mixture"] = (
    df["dataset_number"].astype(str)
    #+ "_"
    + df["mixture_type"].astype(str)
    #+ "_"
    + df["n_contributors"].astype(str)
)

# Calculate within-mixture means
for col in param_cols:
    df[col + "_mean_within_mixture"] = (
        df.groupby(group_cols)[col].transform("mean")
    )

# Calculate deviations from the underlying-mixture mean
for col in param_cols:
    df[col + "_deviation"] = (
        df[col] - df[col + "_mean_within_mixture"]
    )
    

#overrule phi_major_dnax_mean_within_mixture
df['phi_major_dnax_mean_within_mixture'] = df['phi_major_theor']
df['phi_major_dnax_deviation'] = df['phi_major_dnax'] - df['phi_major_dnax_mean_within_mixture']

# # Optional: standardized deviations.
# # This expresses deviations in within-mixture SD units.
# for col in param_cols:
#     df[col + "_z_within_mixture"] = (
#         df.groupby(group_cols)[col]
#         .transform(lambda x: (x - x.mean()) / x.std(ddof=1))
#     )
    
# =============================================================================
# Correlation of within-mixture deviations
# =============================================================================

deviation_cols = [col + "_deviation" for col in param_cols]

within_corr = df[deviation_cols].corr(method="pearson")

print("\nCorrelation matrix of within-mixture deviations:")
print(within_corr)

plt.figure(figsize=(8, 6))

sns.heatmap(
    within_corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True
)

plt.title("Correlation of within-mixture deviations")
plt.tight_layout()
plt.show()



target = "phi_major_dnax_deviation"

phi_correlations = (
    within_corr[target]
    .drop(target)
    .sort_values()
)

print("\nCorrelations with phi_major_dnax deviation:")
print(phi_correlations)

plt.figure(figsize=(7, 4))

phi_correlations.plot(kind="barh")

plt.axvline(0, color="black", linewidth=1)
plt.xlim(-1, 1)
plt.xlabel("Correlation with phi_major_dnax deviation")
plt.title("Within-mixture correlation with phi_major_dnax")
plt.tight_layout()
plt.show()


# =============================================================================
# Scatterplots: phi deviation vs other parameter deviations
# =============================================================================

for col in [
    "expectedPeakHeight_deviation",
    "peakHeightVariance_deviation",
    "degradationSlope_deviation"
]:
    plt.figure(figsize=(6, 5))

    sns.scatterplot(
        data=df,
        x="phi_major_dnax_deviation",
        y=col,
        hue="mixture_type",
        style="n_contributors",
        s=80
    )

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)

    plt.xlabel("phi_major_dnax deviation from configuration value")
    plt.ylabel(col.replace("_deviation", "") + " deviation from mixture mean")
    plt.title(f"Within-mixture deviations: phi_major_dnax vs {col.replace('_deviation', '')}")

    plt.tight_layout()
    plt.show()