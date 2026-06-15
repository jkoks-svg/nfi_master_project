import numpy as np
import pandas as pd
from pathlib import Path
import time

# =========================================================
# Utility functions
# =========================================================

def normalize_locus(name):
    return name.replace(" ", "")


def canonical_allele(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def canonical_pair(a1, a2):
    return tuple(sorted([canonical_allele(a1), canonical_allele(a2)]))


# =========================================================
# Frequency preparation
# =========================================================

def build_population_freq_dict(freq_df):
    freq_dict = {}

    for locus in freq_df.columns[1:]:
        col = freq_df[["Allele", locus]].dropna()

        alleles = col["Allele"].apply(canonical_allele).to_numpy()
        probs = col[locus].astype(float).to_numpy()
        probs /= probs.sum()

        freq_dict[normalize_locus(locus)] = (alleles, probs)

    return freq_dict


# =========================================================
# Observed alleles
# =========================================================

def get_observed_alleles(df):

    observed = {}

    for locus, g in df.groupby("Locus"):

        alleles = set()
        allele_cols = [c for c in df.columns if c.startswith("Unknown")]
            
        for col in allele_cols:
        
            for a in g[col].dropna():

                if a != "Ø":
                    alleles.add(canonical_allele(a))

        observed[normalize_locus(locus)] = alleles

    return observed


# =========================================================
# Phi allele construction
# =========================================================

def build_phi_freq_dict(freq_dict, observed):

    phi_dict = {}

    for locus, (alleles, probs) in freq_dict.items():

        seen = observed.get(locus, set())

        mask = np.array([a not in seen for a in alleles])

        phi_alleles = alleles[mask]
        phi_probs = probs[mask]

        if len(phi_alleles) == 0:
            phi_alleles = alleles
            phi_probs = probs

        phi_probs /= phi_probs.sum()

        phi_dict[locus] = (phi_alleles, phi_probs)

    return phi_dict


def build_phi_genotype_dict(phi_freq_dict):

    phi_genotype_dict = {}

    for locus, (alleles, probs) in phi_freq_dict.items():

        genotypes = []
        gprobs = []

        for i, a in enumerate(alleles):
            for j, b in enumerate(alleles[i:], start=i):

                if a == b:
                    p = probs[i] ** 2
                else:
                    p = 2 * probs[i] * probs[j]

                genotypes.append((a, b))
                gprobs.append(p)

        gprobs = np.array(gprobs)
        gprobs /= gprobs.sum()

        phi_genotype_dict[locus] = (genotypes, gprobs)

    return phi_genotype_dict


# =========================================================
# Sampling functions
# =========================================================

def sample_phi_allele(locus, phi_freq_dict, DEBUG):

    locus = normalize_locus(locus)

    alleles, probs = phi_freq_dict[locus]

    sampled = np.random.choice(alleles, p=probs)

    if DEBUG:
        prob = probs[list(alleles).index(sampled)]

        print("\n[DEBUG] Phi allele sampling")
        print(f"  Locus: {locus}")
        print(f"  Possible φ alleles: {list(alleles)}")
        print(f"  Probabilities: {np.round(probs,5)}")
        print(f"  Sampled allele: {sampled}")
        print(f"  Sample probability: {prob:.5f}")

    return sampled

def sample_phi_genotype(locus, phi_genotype_dict, DEBUG):

    locus = normalize_locus(locus)

    genotypes, probs = phi_genotype_dict[locus]

    idx = np.random.choice(len(genotypes), p=probs)

    sampled = genotypes[idx]

    if DEBUG:

        print("\n[DEBUG] Phi genotype sampling")
        print(f"  Locus: {locus}")
        print(f"  Number of φ genotypes: {len(genotypes)}")
        print(f"  Sampled genotype: {sampled}")
        print(f"  Sample probability: {probs[idx]:.5f}")

    return sampled


# =========================================================
# Joint genotype sampling
# =========================================================

def sample_joint_genotypes(df, N, phi_freq_dict, phi_genotype_dict, observed, DEBUG):

    COL_MARKER = "Locus"
    COL_PROB = "Probability"

    df = df.copy()

    allele_cols = [c for c in df.columns if c.startswith("Unknown")]
    contributors = sorted(set(c.split("-")[0] for c in allele_cols))

    for col in allele_cols:
        df[col] = df[col].apply(canonical_allele)

    grouped = []

    for locus, g in df.groupby(COL_MARKER):

        w = g[COL_PROB].astype(float).to_numpy()
        w /= w.sum()

        grouped.append((locus, g.reset_index(drop=True), w))

    samples = []

    for _ in range(N):

        sample_dict = {}

        for locus, g, w in grouped:

            row = g.iloc[np.random.choice(len(g), p=w)]
            
            sample_dict[locus] = {}

            for contributor in contributors:
            
                a1 = row[f"{contributor}-Allele1"]
                a2 = row[f"{contributor}-Allele2"]
            
                if DEBUG:
                    print(f"  {contributor} sampled:", (a1, a2))
            
                if (a1 in ["Ø", None]) and (a2 in ["Ø", None]):
                    a1, a2 = sample_phi_genotype(locus, phi_genotype_dict, DEBUG)
            
                else:
                    if a1 in ["Ø", None]:
                        a1 = sample_phi_allele(locus, phi_freq_dict, DEBUG)
            
                    if a2 in ["Ø", None]:
                        a2 = sample_phi_allele(locus, phi_freq_dict, DEBUG)
            
                sample_dict[locus][contributor] = canonical_pair(a1, a2)
            
        samples.append(sample_dict)

    return samples


# =========================================================
# Write sampled genotypes
# =========================================================

def write_samples(samples, output_root):

    output_root.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples, 1):

        sample_dir = output_root / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)

        loci = list(s.keys())
        
        
        contributors = list(next(iter(samples[0].values())).keys())

        for contributor in contributors:
        
            df = pd.DataFrame({
                "SampleName": contributor,
                "Marker": loci,
                "Allele1": [s[l][contributor][0] for l in loci],
                "Allele2": [s[l][contributor][1] for l in loci],
            })
        
            df.to_csv(sample_dir / f"{contributor}_genotype.csv", index=False)
        

# =========================================================
# Main callable function
# =========================================================

def run_joint_sampling(
    mixture_folder,
    RESOURCES_PATH,
    N_samples,
    DEBUG,
):
    freq_path = RESOURCES_PATH / 'NFI_frequencies.csv'
    freq_df = pd.read_csv(freq_path)
    freq_df["Allele"] = freq_df["Allele"].astype(str)

    freq_dict = build_population_freq_dict(freq_df)

    mixture_list = [p.name for p in mixture_folder.iterdir() if p.is_dir()]

    for mixture in mixture_list:

        results_path = mixture_folder / mixture / "results_joint_deconvolution_H2_clean.csv"
        try:
            results_df = pd.read_csv(results_path, sep="\t")
            results_df_test = results_df.groupby("Locus")
        except:
            results_df = pd.read_csv(results_path)
            results_df_test = results_df.groupby("Locus")    
        observed = get_observed_alleles(results_df)
        

        phi_freq_dict = build_phi_freq_dict(freq_dict, observed)
        phi_genotype_dict = build_phi_genotype_dict(phi_freq_dict)

        samples = sample_joint_genotypes(
            results_df,
            N_samples,
            phi_freq_dict,
            phi_genotype_dict,
            observed, 
            DEBUG
        )

        out = mixture_folder / mixture / "sampled_joint_genotypes"

        write_samples(samples, out)

        print(f"Generated {N_samples} samples for {mixture}")




def main():
    #from pathlib import Path
    
    BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")

    mixture_folder = (
        BASE
        / "inputs"
        / "input_HT_2p_nodropin"
        / "output_HT_2p_nodropin_runF_test_simulations"
        / "mixtures"
    )

    RESOURCES_PATH = BASE / "resources" 
    
    DEBUG = False
    N_SAMPLES = 10

    run_joint_sampling(
        mixture_folder=mixture_folder,
        RESOURCES_PATH = RESOURCES_PATH,
        N_samples=N_SAMPLES,
        DEBUG = DEBUG
    )

    
if __name__ == "__main__":
    main()