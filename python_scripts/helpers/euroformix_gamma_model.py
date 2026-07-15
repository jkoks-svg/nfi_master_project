# -*- coding: utf-8 -*-
"""
Gamma model (signal) + exponential drop-in (Bleka / EuroForMix style)
No stutter, no epsilon
"""

import numpy as np
from scipy.special import gammaln, gammainc
from functools import lru_cache

@lru_cache(maxsize=None)
def _allele_length_cached(kit_name, locus, allele_key, avg_phi_allele, slope, offset):
    if allele_key != "Ø":
        whole, part = (allele_key + ".").split(".")[:2]
        part = part if part != "" else "0"
        return offset + int(part) + int(whole) * slope
    # else:
    #     whole, part = (avg_phi_allele + ".").split(".")[:2]
    #     part = part if part != "" else "0"
    #     return offset + int(part)/4 + int(whole) * slope
    return offset + avg_phi_allele * slope

def allele_length_getter(locus, allele, kit_properties_df, avg_phi_allele, kit_name="PPF6C"):
    props = kit_properties_df.get((kit_name, locus))

    if props is None:
        raise ValueError(f"No kit properties found for kit={kit_name}, locus={locus}")

    slope = props["slope"]
    offset = props["offset"]

    if slope is None or offset is None:
        raise ValueError(f"Missing slope/offset for kit={kit_name}, locus={locus}")

    return _allele_length_cached(
        kit_name,
        locus,
        str(allele),
        float(avg_phi_allele),
        float(slope),
        float(offset),
    )

@lru_cache(maxsize=None)
def _build_allele_length_map_cached(
    kit_name,
    locus,
    all_alleles_tuple,
    avg_phi_allele,
    slope,
    offset,
):
    allele_lengths = {}

    for allele in all_alleles_tuple:
        allele_key = str(allele)

        whole, part = (allele_key + ".").split(".")[:2]
        part = part if part != "" else "0"

        allele_lengths[allele] = offset + int(part) + int(whole) * slope

    allele_lengths["Ø"] = offset + avg_phi_allele * slope

    return allele_lengths


def build_allele_length_map(locus, all_alleles, kit_properties_df, avg_phi_allele, kit_name="PPF6C"):
    props = kit_properties_df.get((kit_name, locus))

    if props is None:
        raise ValueError(f"No kit properties found for kit={kit_name}, locus={locus}")

    slope = props["slope"]
    offset = props["offset"]

    if slope is None or offset is None:
        raise ValueError(f"Missing slope/offset for kit={kit_name}, locus={locus}")

    return _build_allele_length_map_cached(
        kit_name,
        locus,
        tuple(all_alleles),
        float(avg_phi_allele),
        float(slope),
        float(offset),
    )

# --------------------------------------------------
# DNA contribution
# --------------------------------------------------

def prop_DNA_amount(allele, phi0, phi1, geno0_locus, geno1_locus, beta, allele_lengths):
    n0 = geno0_locus.get(allele, 0)
    n1 = geno1_locus.get(allele, 0)

    prop = phi0 * n0 + phi1 * n1

    allele_length = allele_lengths[allele]
    return prop * beta ** ((allele_length - 125) / 100)


# --------------------------------------------------
# Gamma parameters (signal)
# --------------------------------------------------
def gamma_logpdf_fast(h, shape, scale):
    if h <= 0:
        return -np.inf

    return (
        (shape - 1.0) * np.log(h)
        - h / scale
        - gammaln(shape)
        - shape * np.log(scale)
    )


def gamma_logcdf_fast(x, shape, scale):
    return _gamma_logcdf_fast_cached(
        float(x),
        float(shape),
        float(scale),
    )


@lru_cache(maxsize=500_000)
def _gamma_logcdf_fast_cached(x, shape, scale):
    if x <= 0:
        return -np.inf

    cdf = gammainc(shape, x / scale)

    if cdf <= 0:
        return -np.inf

    return float(np.log(cdf))

# --------------------------------------------------
# Locus likelihood
# --------------------------------------------------

def log_likelihood_locus(params, observed, genotypes, locus, all_alleles, threshold, allele_freq_dict, RARE_FREQ, kit_properties_df, avg_phi_allele):

    phi, mu, sigma, C, lam, beta = params
    
    phi0 = phi[0]
    phi1 = phi[1]
    
    sigma2 = sigma * sigma
    scale = mu * sigma2
    
    log_C = np.log(C)
    log_1_minus_C = np.log(1.0 - C)
    log_lam = np.log(lam)
    
    geno0_locus = genotypes[0].get(locus, {})
    geno1_locus = genotypes[1].get(locus, {})
    observed_alleles = observed.get(locus, {})

    active_alleles = set(observed_alleles.keys()) | set(geno0_locus.keys()) | set(geno1_locus.keys())
    
    if "Ø" in observed_alleles or "Ø" in geno0_locus or "Ø" in geno1_locus:
        active_alleles.add("Ø")
    
    allele_lengths = build_allele_length_map(
        locus,
        all_alleles,
        kit_properties_df,
        avg_phi_allele,
    )
    
    degradation_factors = {
        allele: beta ** ((length - 125) / 100)
        for allele, length in allele_lengths.items()
    }
    
    geno0_get = geno0_locus.get
    geno1_get = geno1_locus.get
    obs_get = observed_alleles.get
    deg_get = degradation_factors.__getitem__
    
    logL = 0.0
    dropin_set_size = 0
    
    for allele in active_alleles:
        h = float(obs_get(allele, 0.0))#h = obs_get(allele, 0)
    
        n0 = geno0_get(allele, 0)
        n1 = geno1_get(allele, 0)
    
        prop = (phi0 * n0 + phi1 * n1) * deg_get(allele)
        # ----------------------------------------
        # Case 1: explained by contributors
        # ----------------------------------------
        if prop > 0:
            #shape, scale = gamma_params(prop, mu, sigma)
            shape = prop / sigma2
            if h >= threshold:
                logL += gamma_logpdf_fast(h, shape, scale)
            else:
                logL += gamma_logcdf_fast(threshold, shape, scale)
        # ----------------------------------------
        # Case 2: NOT explained → drop-in model
        # ----------------------------------------
        else:
            if h >= threshold:
                allele_freq = allele_freq_dict.get(str(allele))
                if allele_freq is None or allele_freq <= 0:
                    allele_freq = RARE_FREQ
    
                #logL += np.log(C) + np.log(allele_freq) + log_dropin_pdf(h, lam, threshold)
                logL += log_C + np.log(allele_freq) + log_lam - lam * (h - threshold)
                dropin_set_size += 1
    
    if dropin_set_size == 0:
        logL += log_1_minus_C
        
    return logL, dropin_set_size


