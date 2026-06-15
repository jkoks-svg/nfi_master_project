# dropin_remover.py
# -*- coding: utf-8 -*-
"""
Remove drop-in alleles from raw mixture .txt files using known donor profiles.

Filename convention (from your dataset doc)
------------------------------------------
<replicate>_<dataset><mixtureType><nContributors>.txt

Example:
  1_3A2.txt  means:
    replicate      = 1
    dataset number = 3
    mixture type   = A
    contributors   = 2

This script uses the *structured batch loop* (mixture type × dataset × replicate × contributors)
instead of parsing filenames.

Donor selection
---------------
Per dataset number, donors are the ordered donor list:
  dataset 1: A, B, C, D, E    -> donor file IDs: 1A, 1B, 1C, 1D, 1E
  dataset 2: F, G, H, I, J    ->                 2F, 2G, 2H, 2I, 2J
  dataset 3: K, L, M, N, O    ->                 3K, 3L, 3M, 3N, 3O
  dataset 4: P, Q, R, S, T    ->                 4P, 4Q, 4R, 4S, 4T
  dataset 5: U, V, W, X, Y    ->                 5U, 5V, 5W, 5X, 5Y
  dataset 6: Z, AA, AB, AC, AD->                 6Z, 6AA, 6AB, 6AC, 6AD

For nContributors = N, we take the first N donors from the dataset list (as per your doc).

Drop-in definition (implemented)
--------------------------------
At each marker, keep only mixture alleles that appear in ANY of the donors used for that mixture.
All other alleles are removed (drop-in), and allele/height columns are left-shifted.

Unknown marker handling
-----------------------
If a marker is absent in all donor profiles:
- UNKNOWN_MARKER_POLICY = "keep": keep the locus row unchanged (no removal)
- UNKNOWN_MARKER_POLICY = "drop": REMOVE THE ENTIRE ROW from the output mixture file
- UNKNOWN_MARKER_POLICY = "error": raise an error

Outputs
-------
- Cleaned mixture .txt: same format as input, but drop-ins removed (and unknown-marker rows removed if policy="drop").
- Per-mixture human-readable .txt log: which alleles were removed per marker, and whether rows were removed.

"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

# =========================
# CONFIG: PATHS
# =========================

BASE = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new")

#MIXTURE_DIR = BASE / "inputs" / "input_HT_2p5p" / "mixtures"
MIXTURE_DIR = Path(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\Dataset PP6FC Mixtures\HT_2p5p")
OUT_MIXTURE_DIR = BASE / "inputs" / "input_HT_3p_nodropin" / "mixtures_3p_BDE"
OUT_LOG_DIR = BASE / "inputs" / "input_HT_3p_nodropin" / "mixtures_log"

DONOR_DIR = BASE / "Dataset PP6FC Mixtures" / "Donoren"

OUT_MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG: BATCH LOOP
# =========================

RUN_MODE = "batch"  # "batch" or "single"

MIXTURE_TYPES = ["B", "D", "E"]
DATASETS = [1, 2, 3, 4, 5, 6]
REPLICATES = [1, 2, 3]
N_CONTRIBUTORS_LIST = [3]  # [2, 3, 4, 5] for full 2p-5p dataset

# If a marker is absent in all donor profiles:
# - "keep": keep all alleles at that marker (no removal)
# - "drop": REMOVE THE ENTIRE MARKER ROW from the output .txt
# - "error": raise an error
UNKNOWN_MARKER_POLICY = "drop"
LOG_SKIPPED_UNKNOWN_MARKERS = True

# =========================
# DONOR DEFINITIONS
# =========================

DATASET_DONOR_ORDER: Dict[int, List[str]] = {
    1: ["1A", "1B", "1C", "1D", "1E"],
    2: ["2F", "2G", "2H", "2I", "2J"],
    3: ["3K", "3L", "3M", "3N", "3O"],
    4: ["4P", "4Q", "4R", "4S", "4T"],
    5: ["5U", "5V", "5W", "5X", "5Y"],
    6: ["6Z", "6AA", "6AB", "6AC", "6AD"],
}

_donor_cache: Dict[str, pd.DataFrame] = {}

# =========================
# ALLELE NORMALIZATION
# =========================

def canonical_allele(val) -> Optional[str]:
    """Normalize allele strings: '19' == '19.0' -> '19'; keep microvariants like '30.2'."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d == d.to_integral_value():
        return str(int(d))
    s2 = format(d.normalize(), "f")
    if "E" in s2 or "e" in s2:
        s2 = f"{d:f}"
    return s2

# =========================
# MIXTURE COLUMN HELPERS
# =========================

def find_allele_height_pairs(columns: Sequence[str]) -> List[Tuple[str, str, int]]:
    """Find ('Allele i', 'Height i', i) pairs in the mixture file."""
    pairs: List[Tuple[str, str, int]] = []
    colset = set(columns)
    for col in columns:
        c = col.strip()
        if c.lower().startswith("allele "):
            try:
                idx = int(c.split()[-1])
            except Exception:
                continue
            a = f"Allele {idx}"
            h = f"Height {idx}"
            if h in colset:
                pairs.append((a, h, idx))
    pairs.sort(key=lambda x: x[2])
    return pairs

# =========================
# DONOR LOADING / ALLOWED SETS
# =========================

def load_donor(donor_id: str) -> pd.DataFrame:
    """Load donor profile {DONOR_DIR}/{donor_id}.csv with caching."""
    if donor_id not in _donor_cache:
        path = DONOR_DIR / f"{donor_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Donor file not found: {path}")
        df = pd.read_csv(path, sep=";", dtype=str)

        required = {"Marker", "Allele1", "Allele2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Donor file {path} missing columns: {sorted(missing)}")

        df["Allele1"] = df["Allele1"].apply(canonical_allele)
        df["Allele2"] = df["Allele2"].apply(canonical_allele)
        _donor_cache[donor_id] = df

    return _donor_cache[donor_id]


def donors_for_mixture(dataset_number: int, n_contributors: int) -> List[str]:
    """Return donor IDs for this dataset + number of contributors, using the ordered donor list."""
    if dataset_number not in DATASET_DONOR_ORDER:
        raise ValueError(f"Unknown dataset_number: {dataset_number}")
    donor_list = DATASET_DONOR_ORDER[dataset_number]
    if n_contributors < 2 or n_contributors > 5:
        raise ValueError(f"n_contributors must be 2..5, got {n_contributors}")
    return donor_list[:n_contributors]


def allowed_alleles_by_marker(donor_ids: List[str]) -> Dict[str, Set[str]]:
    """marker -> union of all donor alleles at that marker."""
    out: Dict[str, Set[str]] = {}
    for donor_id in donor_ids:
        df = load_donor(donor_id)
        for _, r in df.iterrows():
            marker = str(r["Marker"]).strip()
            s = out.setdefault(marker, set())
            a1 = r.get("Allele1")
            a2 = r.get("Allele2")
            if a1:
                s.add(str(a1))
            if a2:
                s.add(str(a2))
    return out

# =========================
# LOGGING
# =========================

@dataclass
class LogEntry:
    marker: str
    observed_alleles: List[str]
    allowed_alleles: List[str]
    removed_dropins: List[str]
    policy: str  # "removed" | "kept_unknown" | "dropped_unknown_row_removed"


def write_log_txt(
    log_entries: List[LogEntry],
    log_txt: Path,
    mixture_path: Path,
    donor_ids: List[str],
) -> None:
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(f"Mixture file: {mixture_path}\n")
        f.write(f"Donors used:  {', '.join(donor_ids)}\n")
        f.write(f"UNKNOWN_MARKER_POLICY: {UNKNOWN_MARKER_POLICY}\n")
        f.write("-" * 80 + "\n\n")

        if not log_entries:
            f.write("No drop-ins removed; no markers logged.\n")
            return

        for e in log_entries:
            f.write(f"Marker: {e.marker}\n")
            f.write(f"  Observed alleles: {', '.join(e.observed_alleles) if e.observed_alleles else '(none)'}\n")
            f.write(f"  Allowed alleles:  {', '.join(e.allowed_alleles) if e.allowed_alleles else '(none)'}\n")

            if e.policy == "removed":
                f.write(f"  Removed drop-ins: {', '.join(e.removed_dropins) if e.removed_dropins else '(none)'}\n")
            elif e.policy == "kept_unknown":
                f.write("  Policy: marker unknown in donors -> kept locus row (no removal)\n")
            elif e.policy == "dropped_unknown_row_removed":
                f.write(f"  Removed drop-ins: {', '.join(e.removed_dropins) if e.removed_dropins else '(none)'}\n")
                f.write("  Policy: marker unknown in donors -> LOCUS ROW REMOVED FROM OUTPUT\n")

            f.write("\n")

# =========================
# CORE: ONE MIXTURE
# =========================

def remove_dropins_one_mixture(
    mixture_txt: Path,
    donor_ids: List[str],
    out_txt: Path,
    log_txt: Optional[Path] = None,
) -> None:
    allowed_map = allowed_alleles_by_marker(donor_ids)

    mix = pd.read_csv(mixture_txt, sep="\t", dtype=str, keep_default_na=False)
    if "Marker" not in mix.columns:
        raise ValueError(f"Mixture file {mixture_txt} has no 'Marker' column.")

    pairs = find_allele_height_pairs(mix.columns)
    if not pairs:
        raise ValueError(f"Mixture file {mixture_txt} has no 'Allele i'/'Height i' column pairs.")

    log_entries: List[LogEntry] = []
    rows_to_drop: List[int] = []  # <-- NEW: drop full rows when policy="drop"

    for i, row in mix.iterrows():
        marker = str(row["Marker"]).strip()
        allowed = allowed_map.get(marker, set())

        # observed in original order
        obs_pairs: List[Tuple[str, str, str]] = []  # (a_raw, h_raw, a_norm)
        obs_norm: List[str] = []

        for allele_col, height_col, _idx in pairs:
            a_raw = row.get(allele_col, "")
            h_raw = row.get(height_col, "")
            a_norm = canonical_allele(a_raw)
            if a_norm is None:
                continue
            obs_pairs.append((a_raw, h_raw, a_norm))
            obs_norm.append(a_norm)

        # decide keep/remove
        if not allowed:
            if UNKNOWN_MARKER_POLICY == "keep":
                kept_pairs = [(a_raw, h_raw) for (a_raw, h_raw, _a_norm) in obs_pairs]

                if LOG_SKIPPED_UNKNOWN_MARKERS and obs_norm:
                    log_entries.append(
                        LogEntry(
                            marker=marker,
                            observed_alleles=obs_norm,
                            allowed_alleles=[],
                            removed_dropins=[],
                            policy="kept_unknown",
                        )
                    )

            elif UNKNOWN_MARKER_POLICY == "drop":
                # NEW BEHAVIOR: remove the entire marker row from output
                rows_to_drop.append(i)

                if LOG_SKIPPED_UNKNOWN_MARKERS and obs_norm:
                    log_entries.append(
                        LogEntry(
                            marker=marker,
                            observed_alleles=obs_norm,
                            allowed_alleles=[],
                            removed_dropins=obs_norm[:],
                            policy="dropped_unknown_row_removed",
                        )
                    )

                # skip any editing; row will be removed
                continue

            else:  # "error"
                raise ValueError(f"Marker '{marker}' missing in donors (UNKNOWN_MARKER_POLICY='error').")

        else:
            kept_pairs: List[Tuple[str, str]] = []
            removed: List[str] = []

            for a_raw, h_raw, a_norm in obs_pairs:
                if a_norm in allowed:
                    kept_pairs.append((a_raw, h_raw))
                else:
                    removed.append(a_norm)

            if removed:
                log_entries.append(
                    LogEntry(
                        marker=marker,
                        observed_alleles=obs_norm,
                        allowed_alleles=sorted(allowed),
                        removed_dropins=removed,
                        policy="removed",
                    )
                )

        # clear allele/height fields
        for allele_col, height_col, _idx in pairs:
            mix.at[i, allele_col] = ""
            mix.at[i, height_col] = ""

        # left-shift kept
        for j, (a_out, h_out) in enumerate(kept_pairs, start=1):
            allele_col = f"Allele {j}"
            height_col = f"Height {j}"
            if allele_col in mix.columns and height_col in mix.columns:
                mix.at[i, allele_col] = a_out
                mix.at[i, height_col] = h_out

    # NEW: drop rows and reindex
    if rows_to_drop:
        mix = mix.drop(index=rows_to_drop).reset_index(drop=True)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    mix.to_csv(out_txt, sep="\t", index=False)

    if log_txt is not None:
        write_log_txt(
            log_entries=log_entries,
            log_txt=log_txt,
            mixture_path=mixture_txt,
            donor_ids=donor_ids,
        )

# =========================
# BATCH RUN (YOUR STRUCTURE)
# =========================

def mixture_filename(replicate_number: int, dataset_number: int, mixture_type: str, n_contributors: int) -> str:
    """
    Official naming convention:
      <replicate>_<dataset><mixtureType><nContributors>.txt
    e.g. 1_3A2.txt
    """
    return f"{replicate_number}_{dataset_number}{mixture_type}{n_contributors}.txt"


if __name__ == "__main__":
    if RUN_MODE != "batch":
        raise ValueError("This script is configured for RUN_MODE='batch'. Set RUN_MODE='single' if needed.")

    print(f"Mixture dir: {MIXTURE_DIR}")
    print(f"Output mixtures: {OUT_MIXTURE_DIR}")
    print(f"Output logs: {OUT_LOG_DIR}\n")

    n_total = 0
    n_missing = 0

    for mixture_type in MIXTURE_TYPES:
        for dataset_number in DATASETS:
            for replicate_number in REPLICATES:
                for n_contributors in N_CONTRIBUTORS_LIST:

                    fname = mixture_filename(replicate_number, dataset_number, mixture_type, n_contributors)
                    mixture_path = MIXTURE_DIR / fname

                    if not mixture_path.exists():
                        n_missing += 1
                        continue

                    donor_ids = donors_for_mixture(dataset_number, n_contributors)

                    out_mixture = OUT_MIXTURE_DIR / f"{mixture_path.stem}.txt"
                    out_log_txt = OUT_LOG_DIR / f"{mixture_path.stem}_dropin_log.txt"

                    print(f"Processing {fname} | donors={'+'.join(donor_ids)}")

                    remove_dropins_one_mixture(
                        mixture_txt=mixture_path,
                        donor_ids=donor_ids,
                        out_txt=out_mixture,
                        log_txt=out_log_txt,
                    )

                    n_total += 1

    print(f"\nDone. Processed {n_total} mixtures. Skipped {n_missing} missing files.")
