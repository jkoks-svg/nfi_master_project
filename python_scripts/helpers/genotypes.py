# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:07:23 2026

@author: jortk
"""

from itertools import combinations_with_replacement
import pandas as pd

def build_frequency_dict(freq_df):
    freq_dict = {}

    for locus in freq_df.columns[1:]:
        
        if locus == "PentaD":
            #print("Penta D is being fixed")
            locus_fixed = "Penta D"
        elif locus == "PentaE":
            #print("Penta E is being fixed")
            locus_fixed = "Penta E"
        else:
            locus_fixed = locus
        
        freq_dict[locus_fixed] = {}

        for _, row in freq_df.iterrows():
            allele_raw = row["Allele"]
            freq = row[locus]
            
            if not pd.isna(freq):
                allele = canonical_allele_label(allele_raw)  
                freq_dict[locus_fixed][allele] = float(freq)

    return freq_dict

def generate_genotypes(alleles):
    genos = []

    genos.extend(combinations_with_replacement(alleles, 2))

    for a in alleles:
        genos.append((a, "Ø"))

    genos.append(("Ø", "Ø"))

    return genos

def genotype_to_dict(g):
    a, b = g

    # if a == "Ø" and b == "Ø":
    #     return {}

    # if a == "Ø":
    #     return {b: 1}

    # if b == "Ø":
    #     return {a: 1}

    if a == b:
        return {a: 2}

    return {a: 1, b: 1}

def build_genotypes(g_tuple, locus):
    return [
        {locus: genotype_to_dict(g)}
        for g in g_tuple
    ]


def canonical_allele_label(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "Ø":
        return s
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
        else:
            return str(v)
    except ValueError:
        return s

def compute_pU(locus, freq_dict, observed_alleles, RARE_FREQ):
    locus_map = freq_dict.get(locus, {})
    obs_set = observed_alleles.get(locus, set())

    total_obs = sum(
        freq for allele, freq in locus_map.items()
        if allele in obs_set
    )

    pU = 1.0 - total_obs

    EPS = RARE_FREQ**2

    return max(pU, EPS)


def build_observed_alleles_from_dict(observed):
    """
    Convert your observed peak dict to allele sets
    """
    observed_sets = {}

    for locus, alleles in observed.items():
        observed_sets[locus] = set(
            canonical_allele_label(a) for a in alleles.keys()
        )

    return observed_sets


def safe_freq(p, allele, RARE_FREQ):
    f = p.get(allele)
    if f is None or f <= 0:
        return RARE_FREQ
    return f


def genotype_prior(g, locus, freq_dict, observed_alleles, RARE_FREQ):
    """
    Correct genotype prior with dropout-aware handling
    """
    p = freq_dict[locus]

    a, b = g

    la = canonical_allele_label(a)
    lb = canonical_allele_label(b)

    pU = compute_pU(locus, freq_dict, observed_alleles, RARE_FREQ)
    
    EPS = RARE_FREQ**2

    # -------------------------
    # CASE 1: full dropout
    # -------------------------
    if la == "Ø" and lb == "Ø":
        return max(pU**2, EPS)

    # -------------------------
    # CASE 2: single allele + dropout
    # -------------------------
    if la == "Ø" and lb != "Ø":
        pb = safe_freq(p, lb, RARE_FREQ)
        return max(2 * pb * pU, EPS)

    if lb == "Ø" and la != "Ø":
        pa = safe_freq(p, la, RARE_FREQ)
        return max(2 * pa * pU, EPS)

    # -------------------------
    # CASE 3: normal genotype
    # -------------------------
    pa = safe_freq(p, la, RARE_FREQ)
    pb = safe_freq(p, lb, RARE_FREQ)

    if la == lb:
        return max(pa**2, EPS)
    else:
        return max(2 * pa * pb, EPS)








