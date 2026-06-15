# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:29:54 2026

@author: jortk
"""

# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm

# ============================================================
# 1. Enter the 16 configurations
#    A = 0, B = 1
# ============================================================

data = [
    # step1, step2, step3, step4, coverage, IVS95
    # Frequentist table: step 4 = A
    ("A", "A", "A", "A", 69.0, 50.5),
    ("A", "A", "B", "A", 73.6, 44.7),
    ("A", "B", "A", "A", 77.0, 30.2),
    ("A", "B", "B", "A", 73.6, 34.3),
    ("B", "A", "A", "A", 70.1, 47.1),
    ("B", "A", "B", "A", 72.4, 48.0),
    ("B", "B", "A", "A", 78.2, 28.0),
    ("B", "B", "B", "A", 79.3, 29.0),

    # Bayesian table: step 4 = B
    ("A", "A", "A", "B", 70.1, 41.7),
    ("A", "A", "B", "B", 74.7, 37.4),
    ("A", "B", "A", "B", 80.5, 23.7),
    ("A", "B", "B", "B", 74.7, 26.4),
    ("B", "A", "A", "B", 73.6, 39.1),
    ("B", "A", "B", "B", 73.6, 39.4),
    ("B", "B", "A", "B", 80.5, 21.0),
    ("B", "B", "B", "B", 81.6, 21.6),
]

df = pd.DataFrame(
    data,
    columns=["step1", "step2", "step3", "step4", "coverage", "ivs95"]
)

# Encode A/B as 0/1
for j in range(1, 5):
    df[f"x{j}"] = (df[f"step{j}"] == "B").astype(int)

# Optional: readable configuration label
df["configuration"] = (
    df["step1"] + df["step2"] + df["step3"] + df["step4"]
)

print("\nInput data:")
print(df[["configuration", "coverage", "ivs95", "x1", "x2", "x3", "x4"]])


# ============================================================
# 2. Function to fit OLS and extract step contributions
# ============================================================

def fit_step_model(df, y_col):
    X = df[["x1", "x2", "x3", "x4"]]
    X = sm.add_constant(X)

    model = sm.OLS(df[y_col], X).fit()

    coef_table = pd.DataFrame({
        "term": ["Intercept", "Step 1: A -> B", "Step 2: A -> B",
                 "Step 3: A -> B", "Step 4: A -> B"],
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "p_value": model.pvalues.values,
    })

    coef_table["coefficient"] = coef_table["coefficient"].round(3)
    coef_table["std_error"] = coef_table["std_error"].round(3)
    coef_table["p_value"] = coef_table["p_value"].round(4)

    return model, coef_table


# ============================================================
# 3. Fit model for coverage
# ============================================================

coverage_model, coverage_effects = fit_step_model(df, "coverage")

print("\n" + "=" * 60)
print("OLS model for coverage")
print("=" * 60)
print(coverage_effects)

print("\nFull statsmodels summary:")
print(coverage_model.summary())


# ============================================================
# 4. Fit model for interval score
# ============================================================

ivs_model, ivs_effects = fit_step_model(df, "ivs95")

print("\n" + "=" * 60)
print("OLS model for IVS95")
print("=" * 60)
print(ivs_effects)

print("\nFull statsmodels summary:")
print(ivs_model.summary())


# ============================================================
# 5. Combine effects into one table
# ============================================================

combined = pd.DataFrame({
    "step": ["Step 1", "Step 2", "Step 3", "Step 4"],
    "coverage_effect_B_minus_A": coverage_model.params[["x1", "x2", "x3", "x4"]].values,
    "ivs95_effect_B_minus_A": ivs_model.params[["x1", "x2", "x3", "x4"]].values,
})

combined["coverage_effect_B_minus_A"] = combined["coverage_effect_B_minus_A"].round(3)
combined["ivs95_effect_B_minus_A"] = combined["ivs95_effect_B_minus_A"].round(3)

# For IVS95, lower is better, so the improvement is minus the coefficient
combined["ivs95_improvement_when_switching_to_B"] = (
    -combined["ivs95_effect_B_minus_A"]
).round(3)

print("\n" + "=" * 60)
print("Estimated step contributions")
print("=" * 60)
print(combined)


# ============================================================
# 6. Optional: fitted values and residuals
# ============================================================

df["coverage_fitted"] = coverage_model.fittedvalues
df["coverage_residual"] = coverage_model.resid

df["ivs95_fitted"] = ivs_model.fittedvalues
df["ivs95_residual"] = ivs_model.resid

print("\n" + "=" * 60)
print("Fitted values and residuals")
print("=" * 60)
print(df[
    [
        "configuration",
        "coverage", "coverage_fitted", "coverage_residual",
        "ivs95", "ivs95_fitted", "ivs95_residual"
    ]
].round(3))















from itertools import combinations
from math import factorial
import pandas as pd


# ============================================================
# Shapley contribution analysis
# ============================================================

STEPS = [1, 2, 3, 4]


def make_value_dict(df, y_col, higher_is_better=True):
    """
    Creates a dictionary mapping each subset of B-steps to the corresponding outcome.

    Example:
        frozenset()         -> AAAA
        frozenset({2, 4})   -> ABAB
        frozenset({1,2,3,4})-> BBBB

    If higher_is_better=False, values are converted to improvements relative to AAAA.
    This is useful for IVS95, because lower interval score is better.
    """
    value = {}

    baseline = df.loc[
        (df["x1"] == 0) & (df["x2"] == 0) & (df["x3"] == 0) & (df["x4"] == 0),
        y_col
    ].iloc[0]

    for _, row in df.iterrows():
        subset = frozenset(j for j in STEPS if row[f"x{j}"] == 1)

        if higher_is_better:
            value[subset] = row[y_col]
        else:
            value[subset] = baseline - row[y_col]

    return value


def shapley_values(value_dict):
    """
    Computes Shapley values for the four algorithmic steps.
    """
    n = len(STEPS)
    shapley = {j: 0.0 for j in STEPS}

    for j in STEPS:
        other_steps = [k for k in STEPS if k != j]

        for r in range(n):
            for subset_tuple in combinations(other_steps, r):
                subset = frozenset(subset_tuple)

                weight = (
                    factorial(len(subset))
                    * factorial(n - len(subset) - 1)
                    / factorial(n)
                )

                marginal_gain = value_dict[subset | {j}] - value_dict[subset]
                shapley[j] += weight * marginal_gain

    return shapley


# Coverage: higher is better
coverage_values = make_value_dict(df, "coverage", higher_is_better=True)
coverage_shapley = shapley_values(coverage_values)

# IVS95: lower is better, so convert to improvement
ivs_values = make_value_dict(df, "ivs95", higher_is_better=False)
ivs_shapley = shapley_values(ivs_values)

shapley_table = pd.DataFrame({
    "step": ["Step 1", "Step 2", "Step 3", "Step 4"],
    "coverage_contribution": [coverage_shapley[j] for j in STEPS],
    "ivs95_improvement_contribution": [ivs_shapley[j] for j in STEPS],
})

total_coverage_gain = coverage_values[frozenset(STEPS)] - coverage_values[frozenset()]
total_ivs_improvement = ivs_values[frozenset(STEPS)] - ivs_values[frozenset()]

shapley_table["coverage_share_%"] = (
    100 * shapley_table["coverage_contribution"] / total_coverage_gain
)

shapley_table["ivs95_share_%"] = (
    100 * shapley_table["ivs95_improvement_contribution"] / total_ivs_improvement
)

print("\n" + "=" * 60)
print("Shapley contribution analysis")
print("=" * 60)
print(f"Total coverage gain AAAA -> BBBB: {total_coverage_gain:.3f}")
print(f"Total IVS95 improvement AAAA -> BBBB: {total_ivs_improvement:.3f}")
print(shapley_table.round(3))
















from itertools import combinations
import numpy as np
import pandas as pd


# ============================================================
# Full factorial contrast analysis
# ============================================================

def factorial_contrasts(df, y_col):
    """
    Computes factorial contrasts for all main effects and interactions.

    Uses z-coding:
        A = -1
        B = +1

    Main effect contrast:
        average difference B - A

    Interaction contrasts:
        difference-in-differences, averaged over remaining steps.
    """

    local = df.copy()

    for j in STEPS:
        local[f"z{j}"] = 2 * local[f"x{j}"] - 1

    rows = []

    for order in range(1, len(STEPS) + 1):
        for subset in combinations(STEPS, order):
            product = np.prod([local[f"z{j}"] for j in subset], axis=0)

            # coefficient in orthogonal +/- coded model
            coef = np.mean(local[y_col] * product)

            # convert to ordinary factorial contrast scale
            contrast = (2 ** order) * coef

            rows.append({
                "term": ":".join([f"Step {j}" for j in subset]),
                "order": order,
                "contrast": contrast,
            })

    return pd.DataFrame(rows)


coverage_contrasts = factorial_contrasts(df, "coverage")
ivs_contrasts = factorial_contrasts(df, "ivs95")

print("\n" + "=" * 60)
print("Factorial contrasts for coverage")
print("=" * 60)
print(coverage_contrasts.round(3))

print("\n" + "=" * 60)
print("Factorial contrasts for IVS95")
print("=" * 60)
print(ivs_contrasts.round(3))