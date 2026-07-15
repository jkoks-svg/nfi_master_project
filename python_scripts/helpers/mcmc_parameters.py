# -*- coding: utf-8 -*-
"""
MCMC for estimating (mu, sigma, phi) from a single DNA mixture.

Assumptions:
- current gamma/drop-in likelihood is used as-is
- C and lam are fixed from config
- implemented for N_contributors = 2
- priors:
    mu    ~ Uniform(0, 4e4)
    sigma ~ Uniform(0, 1)
    phi   ~ Dirichlet(2, 1), i.e. phi = [phi1, 1-phi1]

Posterior target:
    p(mu, sigma, phi | E) ∝ p(E | mu, sigma, phi) p(mu)p(sigma)p(phi)

where p(E | mu, sigma, phi) is the mixture marginal likelihood:
    product over loci of
        sum over genotype tuples [
            genotype_prior(g_tuple) * likelihood(E_locus | g_tuple, params)
        ]
"""

from __future__ import annotations

import math
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.special import logsumexp

from .euroformix_gamma_model import log_likelihood_locus
from .genotypes import (
    generate_genotypes,
    build_genotypes,
    genotype_prior,
    build_observed_alleles_from_dict,
    canonical_allele_label,
)


# --------------------------------------------------
# numerics
# --------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_log(x: float) -> float:
    return -np.inf if x <= 0 else math.log(x)


# --------------------------------------------------
# transformed parameterization
# --------------------------------------------------
# We run random-walk MH on unconstrained variables:
#   z_mu, z_sigma, z_phi
#
# and transform to:
#   mu    = 40000 * sigmoid(z_mu)
#   sigma = sigmoid(z_sigma)
#   phi1  = sigmoid(z_phi)
#   phi   = [phi1, 1-phi1]
#
# Because we sample in z-space, the target needs the Jacobian.
# --------------------------------------------------


def unpack_params_from_z(
    z: np.ndarray,
    mu_1: float,
    sigma_1: float,
) -> Tuple[float, float, float, List[float]]:
    z_mu, z_sigma, z_phi, z_beta = z

    mu = mu_1 * sigmoid(z_mu)
    sigma = sigma_1 * sigmoid(z_sigma)
    #phi1 = sigmoid(z_phi)
    # phi1 is now constrained to (0.5, 1)
    phi1 = 0.5 + 0.5 * sigmoid(z_phi)
    beta = sigmoid(z_beta)

    phi = [phi1, 1.0 - phi1]
    return mu, sigma, beta, phi


def log_prior_plus_jacobian(
    z: np.ndarray,
    mu_1: float,
    sigma_1: float,
) -> float:
    """
    Priors:
        mu    ~ Uniform(0, mu_1)
        sigma ~ Uniform(0, sigma_1)
        beta  ~ Uniform(0, 1)
        phi1  ~ Beta(2,1), restricted to phi1 > 0.5

    Since MH is done in z-space, add log|Jacobian|.
    Constants are omitted.
    """
    mu, sigma, beta, phi = unpack_params_from_z(z, mu_1=mu_1, sigma_1=sigma_1)
    phi1 = phi[0]

    if not (0.0 < mu < mu_1):
        return -np.inf
    if not (0.0 < sigma < sigma_1):
        return -np.inf
    if not (0.0 < beta < 1.0):
        return -np.inf
    if not (0.5 < phi1 < 1.0):
        return -np.inf

    # phi1 ~ Beta(2,1), density proportional to phi1,
    # now restricted to phi1 > 0.5.
    # The normalizing constant for the truncation can be omitted in MH.
    log_prior_theta = safe_log(phi1)

    s_mu = mu / mu_1
    s_sigma = sigma / sigma_1
    s_phi = sigmoid(z[2])   # because phi1 = 0.5 + 0.5 * sigmoid(z_phi)

    log_jac = (
        math.log(mu_1)
        + safe_log(s_mu)
        + safe_log(1.0 - s_mu)

        + math.log(sigma_1)
        + safe_log(s_sigma)
        + safe_log(1.0 - s_sigma)

        # beta = sigmoid(z_beta)
        + safe_log(beta)
        + safe_log(1.0 - beta)

        # phi1 = 0.5 + 0.5 * sigmoid(z_phi)
        # dphi1/dz_phi = 0.5 * sigmoid(z_phi) * (1 - sigmoid(z_phi))
        + math.log(0.5)
        + safe_log(s_phi)
        + safe_log(1.0 - s_phi)
    )

    return log_prior_theta + log_jac


# --------------------------------------------------
# genotype-space marginal likelihood
# --------------------------------------------------

def _observed_set_to_numeric_sorted(observed_sets_locus: set) -> List[Any]:
    alleles = []
    for x in observed_sets_locus:
        s = canonical_allele_label(x)
        if s == "Ø":
            continue
        v = float(s)
        alleles.append(int(v) if v.is_integer() else v)
    alleles.sort()
    return alleles

def allele_to_repeat_length(allele):
    allele_str = str(allele)

    if "." not in allele_str:
        return float(allele_str)

    whole, extra_nt = allele_str.split(".")
    return float(whole) + int(extra_nt) / 4

def avg_phi_allele_getter(observed_alleles, allele_freq_dict, locus):
    observed = {
        str(a)
        for a in observed_alleles.get(locus, {}).keys()
    }

    remaining = {
        allele: freq
        for allele, freq in allele_freq_dict.items()
        if str(allele) not in observed
    }

    if not remaining:
        raise ValueError(
            f"No alleles left after removing observed alleles for locus {locus}."
        )

    total_freq = sum(remaining.values())

    avg_phi_allele = sum(
        allele_to_repeat_length(allele) * (freq / total_freq)
        for allele, freq in remaining.items()
    )

    return avg_phi_allele

# def avg_phi_allele_getter(observed_alleles, allele_freq_dict, locus):
#     observed = {
#         str(a)
#         for a in observed_alleles.get(locus, {}).keys()
#     }

#     remaining = {
#         allele: freq
#         for allele, freq in allele_freq_dict.items()
#         if str(allele) not in observed
#     }

#     if not remaining:
#         raise ValueError(
#             f"No alleles left after removing observed alleles for locus {locus}."
#         )

#     total_freq = sum(remaining.values())
    
#     avg_phi_allele = sum(
#         float(allele) * (freq / total_freq)
#         for allele, freq in remaining.items()
#     )

#     return avg_phi_allele #round(avg_phi_allele,2)

def precompute_mcmc_locus_data(
    observed_alleles,
    freq_dict,
    RARE_FREQ,
    N_contributors,
    candidate_pairs_by_locus=None,
):
    """
    Precompute all genotype-related objects that do not depend on
    mu, sigma, phi, or beta.

    Returns:
        dict[locus] -> {
            "allele_freq_dict": ...,
            "avg_phi_allele": ...,
            "rows": [
                {
                    "g_tuple": ...,
                    "genotypes": ...,
                    "log_prior": ...
                },
                ...
            ]
        }
    """
    observed_sets = build_observed_alleles_from_dict(observed_alleles)

    precomputed = {}

    for locus in observed_alleles:
        if locus not in observed_sets:
            continue

        allele_freq_dict = freq_dict[locus]

        avg_phi_allele = avg_phi_allele_getter(
            observed_alleles,
            allele_freq_dict,
            locus,
        )

        if candidate_pairs_by_locus is not None:
            genotype_tuples = candidate_pairs_by_locus.get(locus)

            if not genotype_tuples:
                raise ValueError(
                    f"No candidate genotype pairs found for locus {locus}."
                )
        else:
            alleles_for_genotypes = _observed_set_to_numeric_sorted(
                observed_sets[locus]
            )
            genos = generate_genotypes(alleles_for_genotypes)
            genotype_tuples = product(genos, repeat=N_contributors)

        prior_cache = {}
        rows = []

        for g_tuple in genotype_tuples:
            genotypes = build_genotypes(g_tuple, locus)

            prior = 1.0
            for g in g_tuple:
                if g not in prior_cache:
                    prior_cache[g] = genotype_prior(
                        g=g,
                        locus=locus,
                        freq_dict=freq_dict,
                        observed_alleles=observed_sets,
                        RARE_FREQ=RARE_FREQ,
                    )

                prior *= prior_cache[g]

            if prior <= 0:
                continue

            rows.append({
                "g_tuple": g_tuple,
                "genotypes": genotypes,
                "log_prior": math.log(prior),
            })

        precomputed[locus] = {
            "allele_freq_dict": allele_freq_dict,
            "avg_phi_allele": avg_phi_allele,
            "rows": rows,
        }

    return precomputed

def log_marginal_likelihood_locus_precomputed(
    observed_alleles,
    locus,
    all_alleles,
    threshold,
    RARE_FREQ,
    params,
    kit_properties_df,
    precomputed_mcmc_data,
):
    locus_data = precomputed_mcmc_data[locus]

    log_terms = []

    for row in locus_data["rows"]:
        logL, _ = log_likelihood_locus(
            params=params,
            observed=observed_alleles,
            genotypes=row["genotypes"],
            locus=locus,
            all_alleles=all_alleles,
            threshold=threshold,
            allele_freq_dict=locus_data["allele_freq_dict"],
            RARE_FREQ=RARE_FREQ,
            kit_properties_df=kit_properties_df,
            avg_phi_allele=locus_data["avg_phi_allele"],
        )

        log_terms.append(row["log_prior"] + float(logL))

    if not log_terms:
        return -np.inf

    return float(logsumexp(log_terms))


def log_marginal_likelihood_mixture(
    observed_alleles: Dict[str, Dict[Any, float]],
    all_alleles_by_locus: Dict[str, List[Any]],
    thresholds: Dict[str, float],
    freq_dict: Dict[str, Dict[str, float]],
    RARE_FREQ: float,
    params: Tuple[List[float], float, float, float, float, float],
    N_contributors: int,
    kit_properties_df,
    candidate_pairs_by_locus=None,#=candidate_pairs_by_locus,
    precomputed_mcmc_data=None,
) -> float:
    
    total = 0.0

    for locus in observed_alleles:
        if precomputed_mcmc_data is not None:
            log_ml = log_marginal_likelihood_locus_precomputed(
                observed_alleles=observed_alleles,
                locus=locus,
                all_alleles=all_alleles_by_locus[locus],
                threshold=thresholds[locus],
                RARE_FREQ=RARE_FREQ,
                params=params,
                kit_properties_df=kit_properties_df,
                precomputed_mcmc_data=precomputed_mcmc_data,
            )
        # else: #should not be used anymore
        #     print("something wrong with precomputing log_marginal_likelihood_mixture")
        #     # log_ml = log_marginal_likelihood_locus(
        #     #     observed_alleles=observed_alleles,
        #     #     locus=locus,
        #     #     all_alleles=all_alleles_by_locus[locus],
        #     #     threshold=thresholds[locus],
        #     #     freq_dict=freq_dict,
        #     #     RARE_FREQ=RARE_FREQ,
        #     #     params=params,
        #     #     N_contributors=N_contributors,
        #     #     kit_properties_df=kit_properties_df,
        #     #     candidate_pairs_by_locus=candidate_pairs_by_locus,
        #     # )
        else:
            raise ValueError(
                "precomputed_mcmc_data must be provided. "
                "The non-precomputed likelihood path is disabled."
            )
            
        if not np.isfinite(log_ml):
            return -np.inf
    
        total += log_ml
    
    return total




def log_posterior_z(
    z: np.ndarray,
    observed_alleles: Dict[str, Dict[Any, float]],
    all_alleles_by_locus: Dict[str, List[Any]],
    thresholds: Dict[str, float],
    freq_dict: Dict[str, Dict[str, float]],
    RARE_FREQ: float,
    C: float,
    lam: float,
    N_contributors: int,
    mu_1: float,
    sigma_1: float,
    kit_properties_df,
    candidate_pairs_by_locus=None, #=candidate_pairs_by_locus,
    precomputed_mcmc_data=None,
) -> float:
    #lp = log_prior_plus_jacobian(z)
    lp = log_prior_plus_jacobian(z, mu_1=mu_1, sigma_1=sigma_1)
    if not np.isfinite(lp):
        return -np.inf

    #mu, sigma, phi = unpack_params_from_z(z)
    #mu, sigma, phi = unpack_params_from_z(z, mu_1=mu_1, sigma_1=sigma_1)
    mu, sigma, beta, phi = unpack_params_from_z(z, mu_1=mu_1, sigma_1=sigma_1)
    # avoid degenerate sigma very near 0 causing numerical problems
    if sigma <= 1e-8:
    #if sigma <= min(1e-8, sigma_1 * 1e-12): 
        return -np.inf

    params = (phi, mu, sigma, C, lam, beta)

    ll = log_marginal_likelihood_mixture(
        observed_alleles=observed_alleles,
        all_alleles_by_locus=all_alleles_by_locus,
        thresholds=thresholds,
        freq_dict=freq_dict,
        RARE_FREQ=RARE_FREQ,
        params=params,
        N_contributors=N_contributors,
        kit_properties_df=kit_properties_df,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
        precomputed_mcmc_data=precomputed_mcmc_data,
    )
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def log_posterior_z_rework(
    z: np.ndarray,
    observed_alleles_list: List[Dict[str, Dict[Any, float]]],
    all_alleles_by_locus: Dict[str, List[Any]],
    thresholds_list: List[Dict[str, float]],
    freq_dict: Dict[str, Dict[str, float]],
    RARE_FREQ: float,
    C: float,
    lam: float,
    N_contributors: int,
    mu_1: float,
    sigma_1: float,
    kit_properties_df,
    candidate_pairs_by_locus=None,
    precomputed_mcmc_data_list=None,
) -> float:
    """
    Rework target:

        log p(theta | E1, E2, E3)
        =
        log prior(theta)
        +
        sum_r log p(E_r | theta)

    where each E_r is one replicate profile.

    If candidate_pairs_by_locus is supplied, each replicate likelihood
    is evaluated only on those genotype pairs.
    """
    lp = log_prior_plus_jacobian(z, mu_1=mu_1, sigma_1=sigma_1)
    if not np.isfinite(lp):
        return -np.inf

    mu, sigma, beta, phi = unpack_params_from_z(
        z,
        mu_1=mu_1,
        sigma_1=sigma_1,
    )

    if sigma <= 1e-8:
        return -np.inf

    params = (phi, mu, sigma, C, lam, beta)

    total_ll = 0.0

    for r, (observed_alleles, thresholds) in enumerate(
        zip(observed_alleles_list, thresholds_list)
    ):
        precomputed_mcmc_data = None
        if precomputed_mcmc_data_list is not None:
            precomputed_mcmc_data = precomputed_mcmc_data_list[r]

        ll = log_marginal_likelihood_mixture(
            observed_alleles=observed_alleles,
            all_alleles_by_locus=all_alleles_by_locus,
            thresholds=thresholds,
            freq_dict=freq_dict,
            RARE_FREQ=RARE_FREQ,
            params=params,
            N_contributors=N_contributors,
            kit_properties_df=kit_properties_df,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
            precomputed_mcmc_data=precomputed_mcmc_data,
        )

        if not np.isfinite(ll):
            return -np.inf

        total_ll += ll

    return lp + total_ll


def run_mcmc_for_rework_parameters(
    observed_alleles_list: List[Dict[str, Dict[Any, float]]],
    all_alleles_by_locus: Dict[str, List[Any]],
    thresholds_list: List[Dict[str, float]],
    freq_dict: Dict[str, Dict[str, float]],
    RARE_FREQ: float,
    C: float,
    lam: float,
    N_contributors: int,
    n_iter: int,
    burnin: int,
    thin: int,
    proposal_scales: Tuple[float, float, float, float],
    init_mu: float,
    init_sigma: float,
    init_phi1: float,
    init_beta: float,
    seed: int,
    mu_1: float,
    sigma_1: float,
    kit_properties_df,
    candidate_pairs_by_locus=None#, =candidate_pairs_by_locus,
) -> Dict[str, Any]:
    """
    Random-walk Metropolis-Hastings for a combined rework mixture.

    The likelihood target is the product of the three replicate marginal
    likelihoods, implemented as a sum of log marginal likelihoods.
    """
    if N_contributors != 2:
        raise NotImplementedError(
            "This implementation currently assumes N_contributors = 2."
        )

    if len(observed_alleles_list) != len(thresholds_list):
        raise ValueError(
            "observed_alleles_list and thresholds_list must have the same length."
        )

    rng = np.random.default_rng(seed)
    
    precomputed_mcmc_data_list = [
        precompute_mcmc_locus_data(
            observed_alleles=observed_alleles,
            freq_dict=freq_dict,
            RARE_FREQ=RARE_FREQ,
            N_contributors=N_contributors,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
        )
        for observed_alleles in observed_alleles_list
    ]

    def logit(p: float) -> float:
        p = min(max(p, 1e-12), 1 - 1e-12)
        return math.log(p / (1 - p))

    init_mu_scaled = min(max(init_mu / mu_1, 1e-12), 1 - 1e-12)
    init_sigma_scaled = min(max(init_sigma / sigma_1, 1e-12), 1 - 1e-12)

    if init_phi1 <= 0.5:
        raise ValueError("init_phi1 must be > 0.5 when enforcing phi1 > phi2.")

    init_phi_scaled = 2.0 * (init_phi1 - 0.5)
    init_phi_scaled = min(max(init_phi_scaled, 1e-12), 1 - 1e-12)

    init_beta = min(max(init_beta, 1e-12), 1 - 1e-12)

    z = np.array(
        [
            logit(init_mu_scaled),
            logit(init_sigma_scaled),
            logit(init_phi_scaled),
            logit(init_beta),
        ],
        dtype=float,
    )

    # current_lp = log_posterior_z_rework(
    #     z=z,
    #     observed_alleles_list=observed_alleles_list,
    #     all_alleles_by_locus=all_alleles_by_locus,
    #     thresholds_list=thresholds_list,
    #     freq_dict=freq_dict,
    #     RARE_FREQ=RARE_FREQ,
    #     C=C,
    #     lam=lam,
    #     N_contributors=N_contributors,
    #     mu_1=mu_1,
    #     sigma_1=sigma_1,
    #     kit_properties_df=kit_properties_df,
    #     candidate_pairs_by_locus=candidate_pairs_by_locus,
    # )
    
    current_lp = log_posterior_z_rework(
        z=z,
        observed_alleles_list=observed_alleles_list,
        all_alleles_by_locus=all_alleles_by_locus,
        thresholds_list=thresholds_list,
        freq_dict=freq_dict,
        RARE_FREQ=RARE_FREQ,
        C=C,
        lam=lam,
        N_contributors=N_contributors,
        mu_1=mu_1,
        sigma_1=sigma_1,
        kit_properties_df=kit_properties_df,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
        precomputed_mcmc_data_list=precomputed_mcmc_data_list,
    )

    chain_z = []
    accept_count = 0
    proposal_scales = np.asarray(proposal_scales, dtype=float)

    for i in range(n_iter):
        if i % 100 == 0:
            print("REWORK MCMC iter = " + str(i) + " = " + str(i * 100 / n_iter) + " %")

        z_prop = z + rng.normal(loc=0.0, scale=proposal_scales, size=4)

        # prop_lp = log_posterior_z_rework(
        #     z=z_prop,
        #     observed_alleles_list=observed_alleles_list,
        #     all_alleles_by_locus=all_alleles_by_locus,
        #     thresholds_list=thresholds_list,
        #     freq_dict=freq_dict,
        #     RARE_FREQ=RARE_FREQ,
        #     C=C,
        #     lam=lam,
        #     N_contributors=N_contributors,
        #     mu_1=mu_1,
        #     sigma_1=sigma_1,
        #     kit_properties_df=kit_properties_df,
        #     candidate_pairs_by_locus=candidate_pairs_by_locus,
        # )
        
        prop_lp = log_posterior_z_rework(
            z=z_prop,
            observed_alleles_list=observed_alleles_list,
            all_alleles_by_locus=all_alleles_by_locus,
            thresholds_list=thresholds_list,
            freq_dict=freq_dict,
            RARE_FREQ=RARE_FREQ,
            C=C,
            lam=lam,
            N_contributors=N_contributors,
            mu_1=mu_1,
            sigma_1=sigma_1,
            kit_properties_df=kit_properties_df,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
            precomputed_mcmc_data_list=precomputed_mcmc_data_list,
        )

        log_alpha = prop_lp - current_lp

        if math.log(rng.uniform()) < log_alpha:
            z = z_prop
            current_lp = prop_lp
            accept_count += 1

        chain_z.append(z.copy())

    chain_z = np.asarray(chain_z)
    kept = chain_z[burnin::thin]

    mus = []
    sigmas = []
    betas = []
    phi1s = []
    phi2s = []

    for zi in kept:
        mu, sigma, beta, phi = unpack_params_from_z(
            zi,
            mu_1=mu_1,
            sigma_1=sigma_1,
        )
        mus.append(mu)
        sigmas.append(sigma)
        betas.append(beta)
        phi1s.append(phi[0])
        phi2s.append(phi[1])

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    betas = np.asarray(betas)
    phi1s = np.asarray(phi1s)
    phi2s = np.asarray(phi2s)

    return {
        "acceptance_rate": accept_count / n_iter,
        "n_iter": n_iter,
        "burnin": burnin,
        "thin": thin,
        "samples": {
            "mu": mus,
            "sigma": sigmas,
            "beta": betas,
            "phi1": phi1s,
            "phi2": phi2s,
        },
        "posterior_mean": {
            "mu": float(np.mean(mus)),
            "sigma": float(np.mean(sigmas)),
            "beta": float(np.mean(betas)),
            "phi": [float(np.mean(phi1s)), float(np.mean(phi2s))],
        },
        "posterior_median": {
            "mu": float(np.median(mus)),
            "sigma": float(np.median(sigmas)),
            "beta": float(np.median(betas)),
            "phi": [float(np.median(phi1s)), float(np.median(phi2s))],
        },
        "posterior_ci95": {
            "mu": [float(np.quantile(mus, 0.025)), float(np.quantile(mus, 0.975))],
            "sigma": [float(np.quantile(sigmas, 0.025)), float(np.quantile(sigmas, 0.975))],
            "beta": [float(np.quantile(betas, 0.025)), float(np.quantile(betas, 0.975))],
            "phi1": [float(np.quantile(phi1s, 0.025)), float(np.quantile(phi1s, 0.975))],
            "phi2": [float(np.quantile(phi2s, 0.025)), float(np.quantile(phi2s, 0.975))],
        },
    }









# --------------------------------------------------
# MCMC sampler
# --------------------------------------------------

def run_mcmc_for_parameters(
    observed_alleles: Dict[str, Dict[Any, float]],
    all_alleles_by_locus: Dict[str, List[Any]],
    thresholds: Dict[str, float],
    freq_dict: Dict[str, Dict[str, float]],
    RARE_FREQ: float,
    C: float,
    lam: float,
    N_contributors: int,# = 2,
    n_iter: int,# = 20000,
    burnin: int,# = 5000,
    thin: int,# = 10,
    #proposal_scales: Tuple[float, float, float],# = (0.12, 0.10, 0.10),
    proposal_scales: Tuple[float, float, float, float],
    init_mu: float,# = 1500.0,
    init_sigma: float,# = 0.15,
    init_phi1: float,# = 0.7,
    init_beta: float,
    seed: int,# | None = 12345,
    mu_1: float,
    sigma_1: float,
    kit_properties_df,
    candidate_pairs_by_locus=None,#,=candidate_pairs_by_locus,
    
) -> Dict[str, Any]:
    """
    Random-walk Metropolis-Hastings in transformed space.

    Returns posterior samples and summaries for:
        mu, sigma, phi1, phi2
    """
    if N_contributors != 2:
        raise NotImplementedError("This implementation currently assumes N_contributors = 2.")

    rng = np.random.default_rng(seed)
    
    precomputed_mcmc_data = precompute_mcmc_locus_data(
        observed_alleles=observed_alleles,
        freq_dict=freq_dict,
        RARE_FREQ=RARE_FREQ,
        N_contributors=N_contributors,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
    )

    # inverse transforms for initialization
    def logit(p: float) -> float:
        p = min(max(p, 1e-12), 1 - 1e-12)
        return math.log(p / (1 - p))


    
    init_mu_scaled = min(max(init_mu / mu_1, 1e-12), 1 - 1e-12)
    init_sigma_scaled = min(max(init_sigma / sigma_1, 1e-12), 1 - 1e-12)
    

    if init_phi1 <= 0.5:
        raise ValueError("init_phi1 must be > 0.5 when enforcing phi1 > phi2.")
    
    init_phi_scaled = 2.0 * (init_phi1 - 0.5)
    init_phi_scaled = min(max(init_phi_scaled, 1e-12), 1 - 1e-12)
    
    init_beta = min(max(init_beta, 1e-12), 1 - 1e-12)
    
    z = np.array([
        logit(init_mu_scaled),
        logit(init_sigma_scaled),
        logit(init_phi_scaled),
        logit(init_beta),
    ], dtype=float)
    
    

    current_lp = log_posterior_z(
        z=z,
        observed_alleles=observed_alleles,
        all_alleles_by_locus=all_alleles_by_locus,
        thresholds=thresholds,
        freq_dict=freq_dict,
        RARE_FREQ=RARE_FREQ,
        C=C,
        lam=lam,
        N_contributors=N_contributors,
        mu_1=mu_1,
        sigma_1=sigma_1,
        kit_properties_df=kit_properties_df,
        candidate_pairs_by_locus=candidate_pairs_by_locus,
        precomputed_mcmc_data=precomputed_mcmc_data,
    )

    chain_z = []
    accept_count = 0

    proposal_scales = np.asarray(proposal_scales, dtype=float)
    #proposal_scales: Tuple[float, float, float, float],

    for i in range(n_iter):
        if i % 100 == 0:
            print('MCMC iter = '+str(i)+' = '+str(i*100/n_iter)+ ' %')
        z_prop = z + rng.normal(loc=0.0, scale=proposal_scales, size=4)


        prop_lp = log_posterior_z(
            z=z_prop,
            observed_alleles=observed_alleles,
            all_alleles_by_locus=all_alleles_by_locus,
            thresholds=thresholds,
            freq_dict=freq_dict,
            RARE_FREQ=RARE_FREQ,
            C=C,
            lam=lam,
            N_contributors=N_contributors,
            mu_1=mu_1,
            sigma_1=sigma_1,
            kit_properties_df=kit_properties_df,
            candidate_pairs_by_locus=candidate_pairs_by_locus,
            precomputed_mcmc_data=precomputed_mcmc_data,
        )

        log_alpha = prop_lp - current_lp
        if math.log(rng.uniform()) < log_alpha:
            z = z_prop
            current_lp = prop_lp
            accept_count += 1

        chain_z.append(z.copy())

    chain_z = np.asarray(chain_z)

    kept = chain_z[burnin::thin]

    mus = []
    sigmas = []
    betas = []
    phi1s = []
    phi2s = []


    for zi in kept:

        mu, sigma, beta, phi = unpack_params_from_z(zi, mu_1=mu_1, sigma_1=sigma_1)
        mus.append(mu)
        sigmas.append(sigma)
        betas.append(beta)
        phi1s.append(phi[0])
        phi2s.append(phi[1])
    

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    betas = np.asarray(betas)
    phi1s = np.asarray(phi1s)
    phi2s = np.asarray(phi2s)

    return {
        "acceptance_rate": accept_count / n_iter,
        "n_iter": n_iter,
        "burnin": burnin,
        "thin": thin,
        
        "samples": {
            "mu": mus,
            "sigma": sigmas,
            "beta": betas,
            "phi1": phi1s,
            "phi2": phi2s,
        },
        "posterior_mean": {
            "mu": float(np.mean(mus)),
            "sigma": float(np.mean(sigmas)),
            "beta": float(np.mean(betas)),
            "phi": [float(np.mean(phi1s)), float(np.mean(phi2s))],
        },
        "posterior_median": {
            "mu": float(np.median(mus)),
            "sigma": float(np.median(sigmas)),
            "beta": float(np.median(betas)),
            "phi": [float(np.median(phi1s)), float(np.median(phi2s))],
        },
        "posterior_ci95": {
            "mu": [float(np.quantile(mus, 0.025)), float(np.quantile(mus, 0.975))],
            "sigma": [float(np.quantile(sigmas, 0.025)), float(np.quantile(sigmas, 0.975))],
            "beta": [float(np.quantile(betas, 0.025)), float(np.quantile(betas, 0.975))],
            "phi1": [float(np.quantile(phi1s, 0.025)), float(np.quantile(phi1s, 0.975))],
            "phi2": [float(np.quantile(phi2s, 0.025)), float(np.quantile(phi2s, 0.975))],
        },

    }


