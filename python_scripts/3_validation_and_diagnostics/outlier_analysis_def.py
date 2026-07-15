# -*- coding: utf-8 -*-
"""
outlier_analysis.py

Clean baseline version.

- Allele normalization (e.g. 19 and 19.0 -> "19")
- Plot per-outlier peaks and donor alleles
- Compute info-box fields + counts and add them as columns to outlier_df
- Cause logic:
    * Combine missing alleles present in BOTH major & minor: one line "major/minor ... Genuine dropout"
    * Specific causes for missing minor-only alleles override generic "Peak below threshold"
    * Minor generic cause depends on mixture type:
        - B/D/E -> Peak below threshold
        - A/C/unknown -> Genuine dropout
"""

from pathlib import Path
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

le_or_ge = 'le'
set_delta_threshold = -3

# ----------------------------
# PATHS
# ----------------------------
version = 'v1'
BASE = Path(r"C:\Users\jortk\Documents\inputs\old_inputs\input_HT_2p5p\output_HT_2p5p_runB")
CSV_PATH = BASE / f"results_{version}" / f"outlier_df_{version}.csv" 
#"outlier_loci_log10LRminusmu_le_minus2.csv"

#CSV_PATH = r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code\outlier_loci_log10LRminusmu_le_minus_2.csv"

OUTLIER_PLOTS_DIR = BASE / "outlier_peak_plots"
#OUTLIER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# COLORS
# ----------------------------
COL_PEAK  = "tab:blue"
COL_MAJOR = "tab:green"
COL_MINOR = "tab:brown"


# ----------------------------
# MIXTURE TYPE → PROPORTION
# ----------------------------
MIXTURE_TYPE_PROPORTIONS = {
    "A": (300, 150),
    "B": (300, 30),
    "C": (150, 150),
    "D": (150, 30),
    "E": (600, 30),
}


# ----------------------------
# NORMALIZATION
# ----------------------------
SPECIAL_ALLELES = {"Ø", "X", "Y"}

def normalize_allele(a):
    """Canonical allele string."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None

    s = str(a).strip()
    if s in {"", "nan", "NaN", "None"}:
        return None
    if s in SPECIAL_ALLELES:
        return s

    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s

    if d == d.to_integral_value():
        return str(int(d))

    s2 = format(d.normalize(), "f")
    if "." in s2:
        s2 = s2.rstrip("0").rstrip(".")
    return s2


def _to_float_or_nan(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _allele_sort_key(a):
    """Sort numeric alleles numerically; keep Ø/X/Y at end."""
    a = normalize_allele(a)
    if a is None:
        return (2, 0.0, "")
    if a in SPECIAL_ALLELES:
        return (1, 0.0, a)
    try:
        return (0, float(a), a)
    except Exception:
        return (1, 0.0, a)


def format_pair(prefix: str, r: pd.Series):
    a1 = normalize_allele(r.get(f"{prefix}_a1", None))
    a2 = normalize_allele(r.get(f"{prefix}_a2", None))
    if a1 is None and a2 is None:
        return "NA"
    if a2 is None:
        return f"({a1}, NA)"
    if a1 is None:
        return f"(NA, {a2})"
    return f"({a1}, {a2})"


def format_mixture_proportion(r: pd.Series):
    mtype = str(r.get("mixture_type", "")).strip().upper()
    if mtype in MIXTURE_TYPE_PROPORTIONS:
        maj_h, min_h = MIXTURE_TYPE_PROPORTIONS[mtype]
        return f"Mixture type {mtype} {maj_h}:{min_h}"
    if mtype:
        return f"Mixture type {mtype} (unknown)"
    return "Mixture type NA"


# ----------------------------
# DATA EXTRACTORS
# ----------------------------
# def extract_raw_peaks_from_outlier_row(r: pd.Series) -> pd.DataFrame:
#     """
#     Returns long df with columns: allele, height, peak_idx
#     Supports either raw_Allele i/raw_Height i or Allele i/Height i.
#     """
#     def get_col(name):
#         if f"raw_{name}" in r.index:
#             return r[f"raw_{name}"]
#         if name in r.index:
#             return r[name]
#         return None

#     peaks = []
#     for i in range(1, 11):
#         a = normalize_allele(get_col(f"Allele {i}"))
#         h = _to_float_or_nan(get_col(f"Height {i}"))
#         if a is not None and np.isfinite(h):
#             peaks.append({"allele": a, "height": h, "peak_idx": i})

#     if not peaks:
#         return pd.DataFrame(columns=["allele", "height", "peak_idx"])

#     df = pd.DataFrame(peaks)
#     df = df.sort_values(by="allele", key=lambda s: s.map(_allele_sort_key)).reset_index(drop=True)
#     return df

import re

def extract_raw_peaks_from_outlier_row(r: pd.Series) -> pd.DataFrame:
    """
    Returns long df with columns: allele, height, peak_idx

    Robust: detects all available peak indices automatically (Allele i/Height i
    or raw_Allele i/raw_Height i), so it won't miss Allele 6, 7, ... etc.
    """
    def get_col(name):
        if f"raw_{name}" in r.index:
            return r[f"raw_{name}"]
        if name in r.index:
            return r[name]
        return None

    # Find all indices i for which an Allele i column exists (raw_ or not)
    idxs = set()
    pat = re.compile(r"^(raw_)?Allele\s+(\d+)$")
    for col in r.index:
        m = pat.match(str(col))
        if m:
            idxs.add(int(m.group(2)))

    if not idxs:
        return pd.DataFrame(columns=["allele", "height", "peak_idx"])

    peaks = []
    for i in sorted(idxs):
        a = normalize_allele(get_col(f"Allele {i}"))
        h = _to_float_or_nan(get_col(f"Height {i}"))
        if a is not None and np.isfinite(h):
            peaks.append({"allele": a, "height": h, "peak_idx": i})

    if not peaks:
        return pd.DataFrame(columns=["allele", "height", "peak_idx"])

    df = pd.DataFrame(peaks)
    df = df.sort_values(by="allele", key=lambda s: s.map(_allele_sort_key)).reset_index(drop=True)
    return df



def donor_alleles_from_row(r: pd.Series):
    maj = [normalize_allele(r.get("major_a1", None)), normalize_allele(r.get("major_a2", None))]
    minr = [normalize_allele(r.get("minor_a1", None)), normalize_allele(r.get("minor_a2", None))]
    maj = [a for a in maj if a is not None]
    minr = [a for a in minr if a is not None]
    return maj, minr


# ----------------------------
# STUTTER HELPERS
# ----------------------------
def find_pm1_peaks_not_donor(peak_alleles, donor_alleles, step):
    peak_set = set(map(normalize_allele, peak_alleles))
    donor_set = set(a for a in map(normalize_allele, donor_alleles) if a is not None)

    out = set()
    step_dec = Decimal(str(step))

    for a in donor_set:
        if a in SPECIAL_ALLELES:
            continue
        try:
            a_dec = Decimal(str(a))
        except Exception:
            continue

        candidate = normalize_allele(str(a_dec + step_dec))
        if candidate in peak_set and candidate not in donor_set:
            out.add(candidate)

    return sorted(out, key=_allele_sort_key)


# def find_half_step_peaks_not_donor(peak_alleles, donor_alleles, direction):
#     #ONLY MEANT TO WORK FOR SE33!
def find_half_step_peaks_not_donor(peak_alleles, donor_alleles, direction, marker):
    """
    Detect half-step (±0.5 repeat / ±0.2 allele) stutters.
    ONLY intended to work for SE33.
    """

    # if marker != "SE33":
    #     return []

    peak_set = set(map(normalize_allele, peak_alleles))
    donor_set = set(a for a in map(normalize_allele, donor_alleles) if a is not None)

    out = set()

    for a in donor_set:
        if a in SPECIAL_ALLELES:
            continue
        try:
            a_dec = Decimal(str(a))
        except Exception:
            continue

        integer = a_dec.to_integral_value(rounding="ROUND_FLOOR")
        frac = a_dec - integer

        if frac == Decimal("0.2"):
            candidate = normalize_allele(str(integer + 1)) if direction == +1 else normalize_allele(str(integer))
        # elif frac == Decimal("0"):
        #     candidate = normalize_allele(str(integer + Decimal("0.2")))
        elif frac == Decimal("0"):
            if direction == -1:
                candidate = normalize_allele(str(integer - 1 + Decimal("0.2")))  # 19 -> 18.2
            elif direction == +1:
                candidate = normalize_allele(str(integer + Decimal("0.2")))      # 19 -> 19.2
            else:
                continue
        else:
            continue

        if candidate in peak_set and candidate not in donor_set:
            out.add(candidate)

    return sorted(out, key=_allele_sort_key)


# ----------------------------
# SPECIFIC CAUSE HELPERS (return cause_text, explained_minor_set)
# ----------------------------
def infer_cause_missing_minor_as_backward_stutter(missing_minor, major_alleles, peak_alleles):
    peak_set = set(map(normalize_allele, peak_alleles))
    major_set = set(map(normalize_allele, major_alleles))

    for m in missing_minor:
        m_norm = normalize_allele(m)
        if m_norm is None or m_norm in SPECIAL_ALLELES:
            continue

        try:
            m_dec = Decimal(str(m_norm))
        except Exception:
            continue

        for M in major_set:
            if M is None or M in SPECIAL_ALLELES or M not in peak_set:
                continue
            try:
                M_dec = Decimal(str(M))
            except Exception:
                continue

            if m_dec == M_dec - Decimal("1"):
                txt = (
                    f"Likely cause minor allele {m_norm} not observed:\n"
                    f"Minor allele peak {m_norm} seen as\n"
                    f"-1 backward stutter from major allele\n"
                    f"peak {normalize_allele(M)} and thus wrongfully removed"
                )
                return txt, {m_norm}

    return None, set()


def infer_cause_minor_hidden_under_major_peak(missing_minor, major_alleles, peak_alleles):
    peak_set = set(map(normalize_allele, peak_alleles))
    major_set = set(map(normalize_allele, major_alleles))

    for m in missing_minor:
        m_norm = normalize_allele(m)
        if m_norm is None or m_norm in SPECIAL_ALLELES:
            continue

        try:
            m_dec = Decimal(str(m_norm))
        except Exception:
            continue

        for M in major_set:
            if M is None or M in SPECIAL_ALLELES or M not in peak_set:
                continue
            try:
                M_dec = Decimal(str(M))
            except Exception:
                continue

            if abs(m_dec - M_dec) == Decimal("0.7") or abs(m_dec - M_dec) == Decimal("0.1"): #16-16.1, 16-15.3
                txt = (
                    f"Likely cause minor allele {m_norm} not observed:\n"
                    f"Minor allele peak {m_norm} hidden underneath\n"
                    f"major allele peak {normalize_allele(M)}"
                )
                return txt, {m_norm}

    return None, set()


# ----------------------------
# CORE METRICS (single source of truth)
# ----------------------------
def compute_outlier_metrics(r: pd.Series) -> dict:
    peak_df = extract_raw_peaks_from_outlier_row(r)
    #print(peak_df)
    peak_alleles = peak_df["allele"].astype(str).map(normalize_allele).tolist() if not peak_df.empty else []

    maj_alleles, min_alleles = donor_alleles_from_row(r)
    donor_alleles = maj_alleles + min_alleles

    # lists
    minus1  = find_pm1_peaks_not_donor(peak_alleles, donor_alleles, step=-1)
    plus1   = find_pm1_peaks_not_donor(peak_alleles, donor_alleles, step=+1)
    marker = str(r.get("marker", "")).strip()

    minus05 = find_half_step_peaks_not_donor(
        peak_alleles,
        donor_alleles,
        direction=-1,
        marker=marker
    )
    plus05 = find_half_step_peaks_not_donor(
        peak_alleles,
        donor_alleles,
        direction=+1,
        marker=marker
    )
    # minus05 = find_half_step_peaks_not_donor(
    #     peak_alleles,
    #     donor_alleles,
    #     direction=-1,
    #     marker=peak_df["marker"]
    # )
    # plus05 = find_half_step_peaks_not_donor(
    #     peak_alleles,
    #     donor_alleles,
    #     direction=+1,
    #     marker=peak_df["marker"]
    # )
    #minus05 = find_half_step_peaks_not_donor(peak_alleles, donor_alleles, direction=-1)
    #plus05  = find_half_step_peaks_not_donor(peak_alleles, donor_alleles, direction=+1)

    # stutter_dropin = sorted(
    #     set(peak_alleles) - (set(maj_alleles) | set(min_alleles)),
    #     key=_allele_sort_key
    # )
    # -----------------------------------------
    # Drop-in split: stutter-derived vs genuine
    # -----------------------------------------
    dropin_total = sorted(
        set(peak_alleles) - (set(maj_alleles) | set(min_alleles)),
        key=_allele_sort_key
    )
    
    stutter_set = set(minus1) | set(plus1) | set(minus05) | set(plus05)
    
    dropin_stutter = sorted(set(dropin_total) & stutter_set, key=_allele_sort_key)
    genuine_dropin = sorted(set(dropin_total) - stutter_set, key=_allele_sort_key)


    shared = sorted(set(maj_alleles).intersection(set(min_alleles)), key=_allele_sort_key)
    missing_major = sorted(set(maj_alleles) - set(peak_alleles), key=_allele_sort_key)
    missing_minor = sorted(set(min_alleles) - set(peak_alleles), key=_allele_sort_key)

    # ---- cause logic with overlap handling ----
    cause_lines = []
    explained_minor = set()

    missing_major_norm = [a for a in map(normalize_allele, missing_major) if a is not None]
    missing_minor_norm = [a for a in map(normalize_allele, missing_minor) if a is not None]

    missing_both = sorted(set(missing_major_norm).intersection(set(missing_minor_norm)), key=_allele_sort_key)
    if missing_both:
        both_str = ", ".join(missing_both)
        cause_lines.append(
            f"Likely cause major/minor allele {both_str} not observed:\n"
            f"Genuine dropout"
        )

    missing_major_only = sorted(set(missing_major_norm) - set(missing_both), key=_allele_sort_key)
    missing_minor_only = sorted(set(missing_minor_norm) - set(missing_both), key=_allele_sort_key)

    # specific causes apply only to minor-only
    cause_hidden, explained_h = infer_cause_minor_hidden_under_major_peak(
        missing_minor_only, maj_alleles, peak_alleles
    )
    cause_stutter, explained_s = infer_cause_missing_minor_as_backward_stutter(
        missing_minor_only, maj_alleles, peak_alleles
    )

    has_hidden = int(cause_hidden is not None)
    has_wrong_stutter_removal = int(cause_stutter is not None)

    special_cause_labels = []
    if has_hidden:
        special_cause_labels.append("minor hidden under major peak")
    if has_wrong_stutter_removal:
        special_cause_labels.append("minor removed as -1 stutter")

    if cause_hidden is not None:
        cause_lines.append(cause_hidden)
        explained_minor |= explained_h
    if cause_stutter is not None:
        cause_lines.append(cause_stutter)
        explained_minor |= explained_s

    # generic major-only
    if missing_major_only:
        major_str = ", ".join(missing_major_only)
        cause_lines.append(
            f"Likely cause major allele {major_str} not observed:\n"
            f"Genuine dropout"
        )

    # generic minor-only (only those not specifically explained)
    remaining_minor_only = [m for m in missing_minor_only if normalize_allele(m) not in explained_minor]
    if remaining_minor_only:
        mtype = str(r.get("mixture_type", "")).strip().upper()
        minor_cause = "Peak below threshold" if mtype in {"B", "D", "E"} else "Genuine dropout"
        minor_str = ", ".join(sorted(set(remaining_minor_only), key=_allele_sort_key))
        cause_lines.append(
            f"Likely cause minor allele {minor_str} not observed:\n"
            f"{minor_cause}"
        )

    cause_text = "\n".join(cause_lines) if cause_lines else "None"

    # counts: genuine dropout vs under threshold (aligned with the logic above)
    # - all major missing (including overlap) counted as genuine dropout (but overlap should count once)
    # - for minor-only: those remaining are split by mixture type; specifically explained are NOT counted in either
    n_genuine_dropout = 0
    n_peak_under_threshold = 0

    # count unique missing alleles, not double counting overlap
    n_genuine_dropout += len(set(missing_both))            # overlap: one genuine dropout
    n_genuine_dropout += len(set(missing_major_only))      # major-only: genuine dropout

    mtype = str(r.get("mixture_type", "")).strip().upper()
    for _ in set(remaining_minor_only):
        if mtype in {"B", "D", "E"}:
            n_peak_under_threshold += 1
        else:
            n_genuine_dropout += 1

    # info-box strings
    major_pair_txt = format_pair("major", r)
    minor_pair_txt = format_pair("minor", r)
    mixture_prop_txt = format_mixture_proportion(r)

    shared_txt = ", ".join(shared) if shared else "None"
    missing_major_txt = ", ".join(missing_major) if missing_major else "None"
    missing_minor_txt = ", ".join(missing_minor) if missing_minor else "None"
    #stutter_dropin_txt = ", ".join(stutter_dropin) if stutter_dropin else "None"
    dropin_total_txt   = ", ".join(dropin_total)   if dropin_total   else "None"
    dropin_stutter_txt = ", ".join(dropin_stutter) if dropin_stutter else "None"
    genuine_dropin_txt = ", ".join(genuine_dropin) if genuine_dropin else "None"


    minus1_txt  = ", ".join(minus1)  if minus1  else "None"
    plus1_txt   = ", ".join(plus1)   if plus1   else "None"
    minus05_txt = ", ".join(minus05) if minus05 else "None"
    plus05_txt  = ", ".join(plus05)  if plus05  else "None"

    return {
        # used for plotting
        "peak_df": peak_df,
        "peak_alleles": peak_alleles,
        "maj_alleles": maj_alleles,
        "min_alleles": min_alleles,

        # info box strings
        "infobox_mixture_prop": mixture_prop_txt,
        "infobox_major_pair": major_pair_txt,
        "infobox_minor_pair": minor_pair_txt,
        "infobox_shared": shared_txt,
        "infobox_missing_major": missing_major_txt,
        "infobox_missing_minor": missing_minor_txt,
        "infobox_cause": cause_text,
        #"infobox_stutter_dropin": stutter_dropin_txt,
        "infobox_dropin_total": dropin_total_txt,
        "infobox_dropin_stutter": dropin_stutter_txt,
        "infobox_genuine_dropin": genuine_dropin_txt,
        "infobox_minus1": minus1_txt,
        "infobox_minus05": minus05_txt,
        "infobox_plus05": plus05_txt,
        "infobox_plus1": plus1_txt,

        # counts
        "n_dropout_major": len(set(missing_major_norm)),
        "n_dropout_minor": len(set(missing_minor_norm)),
        "n_dropout_total": int(len(set(missing_major_norm) | set(missing_minor_norm))),
        "n_genuine_dropout": int(n_genuine_dropout),
        "n_peak_under_threshold": int(n_peak_under_threshold),
        

        #"n_dropin": int(len(stutter_dropin)),
        # drop-in counts
        "n_dropin_total": int(len(dropin_total)),
        "n_dropin_stutter": int(len(dropin_stutter)),
        "n_genuine_dropin": int(len(genuine_dropin)),
        "n_minus1_backward": int(len(minus1)),
        "n_plus1_forward": int(len(plus1)),
        "n_minus05_backward": int(len(minus05)),
        "n_plus05_forward": int(len(plus05)),
        "n_backward_total": int(len(minus1) + len(minus05)),
        "n_forward_total": int(len(plus1) + len(plus05)),
        "n_stutter_total": int(len(minus1) + len(minus05) + len(plus05) + len(plus1)),
        "dropin_check" : int(len(dropin_total))==int(len(dropin_stutter))+int(len(genuine_dropin)),


        # special causes
        "has_cause_hidden_under_major": int(has_hidden),
        "has_cause_stutter_wrong_removal": int(has_wrong_stutter_removal),
        "n_special_causes": int(has_hidden + has_wrong_stutter_removal),
        "special_causes": ", ".join(special_cause_labels) if special_cause_labels else "None",
        "dropout_check1" : int(len(set(missing_major_norm) | set(missing_minor_norm)))==len(set(missing_major_norm))+len(set(missing_minor_norm)),
        "dropout_check2" : int(len(set(missing_major_norm) | set(missing_minor_norm)))==(
            int(n_genuine_dropout)+int(n_peak_under_threshold)+int(has_hidden)+int(has_wrong_stutter_removal))
        
    }


# ----------------------------
# PLOTTING
# ----------------------------
def plot_outlier_row(r: pd.Series, save_dir: Path, use_top_markers=True):
    mixture = str(r.get("mixture", "UNKNOWN_MIX"))
    marker  = str(r.get("marker", "UNKNOWN_MARKER"))
    contr   = str(r.get("contributor", ""))
    role    = str(r.get("donor_role", ""))
    delta   = r.get("delta", np.nan)

    m = compute_outlier_metrics(r)
    peak_df = m["peak_df"]
    if peak_df.empty:
        print(f"[WARNING] No raw peak data for {mixture} {marker}. Skipping.")
        return

    peak_alleles = m["peak_alleles"]
    maj_alleles = m["maj_alleles"]
    min_alleles = m["min_alleles"]

    # x-axis labels
    x_labels = sorted(set(peak_alleles) | set(maj_alleles) | set(min_alleles), key=_allele_sort_key)
    allele_to_idx = {a: i for i, a in enumerate(x_labels)}

    x_pos = np.array([allele_to_idx[a] for a in peak_alleles], dtype=float)
    heights = peak_df["height"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.vlines(x_pos, 0, heights, linewidth=1.6, alpha=0.85, color=COL_PEAK)
    ax.scatter(x_pos, heights, s=55, zorder=3, color=COL_PEAK)

    for xi, h in zip(x_pos, heights):
        ax.text(xi, h * 1.02, f"{int(round(h))}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Allele")
    ax.set_ylabel("Peak height")

    # delta_txt = f"delta={float(delta):.2f}" if np.isfinite(delta) else "delta=NA"
    # ax.set_title(f"{mixture} | {marker} | {contr} ({role}) ") #" | {delta_txt}")

    # y_top = float(max(heights)) * 1.20
    # ax.set_ylim(0, y_top * 1.05)
    
    detection_threshold = 140 if marker == "D2S1338" else None

    max_y_for_scale = max(float(max(heights)), detection_threshold or 0)
    y_top = max_y_for_scale * 1.20
    ax.set_ylim(0, y_top * 1.05)
    
    if detection_threshold is not None:
        ax.axhline(
            detection_threshold,
            linestyle="--",
            linewidth=1.2,
            color="red",
            alpha=0.8,
            label="Detection threshold (140 RFU)",
            zorder=2,
        )
    
    

    def allele_to_x(a):
        a = normalize_allele(a)
        if a is None:
            return None
        return allele_to_idx.get(a, None)

    def draw_donor(prefix: str, label: str, color: str, marker_style: str,
                  x_offset: float, y_frac: float):
        a1 = normalize_allele(r.get(f"{prefix}_a1", None))
        a2 = normalize_allele(r.get(f"{prefix}_a2", None))

        xs = []
        for a in (a1, a2):
            x = allele_to_x(a)
            if x is not None:
                xs.append(x)
        if not xs:
            return False

        xs = sorted(set(xs))

        if use_top_markers:
            y = y_top * y_frac
            ax.scatter([x + x_offset for x in xs], [y] * len(xs),
                       s=80, marker=marker_style, color=color, zorder=6, label=label)
        return True

    drew_any = False
    drew_any |= draw_donor("major", "Major donor alleles", COL_MAJOR, "^", x_offset=-0.12, y_frac=0.94)
    drew_any |= draw_donor("minor", "Minor donor alleles", COL_MINOR, "s", x_offset=+0.12, y_frac=0.94)

    # info lines (fixed)
    info_lines = [
        m["infobox_mixture_prop"],
        f"Major allele pair {m['infobox_major_pair']}",
        f"Minor allele pair {m['infobox_minor_pair']}",
        f"Shared alleles: {m['infobox_shared']}",
        f"Major not observed: {m['infobox_missing_major']}",
        f"Minor not observed: {m['infobox_missing_minor']}",
        m["infobox_cause"],
        f"Stutter/drop-in not removed: {m['infobox_dropin_total']}",
        f"-1 backward stutter: {m['infobox_minus1']}",
        f"-0.5 backward stutter: {m['infobox_minus05']}",
        f"+0.5 forward stutter: {m['infobox_plus05']}",
        f"+1 forward stutter: {m['infobox_plus1']}",
    ]

    legend = None
    if drew_any:
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                           borderaxespad=0.0, frameon=True)

    fig.tight_layout() #rect=[0, 0, 0.80, 1])

    # if info_lines:
    #     fig.canvas.draw()
    #     if legend is not None:
    #         bbox = legend.get_window_extent(fig.canvas.get_renderer())
    #         bbox_fig = bbox.transformed(fig.transFigure.inverted())
    #         x0 = bbox_fig.x0
    #         y0 = bbox_fig.y0 - 0.02
    #     else:
    #         x0, y0 = 0.82, 0.98

    #     fig.text(
    #         x0, y0,
    #         "\n".join(info_lines),
    #         ha="left", va="top",
    #         fontsize=9,
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.95)
    #     )

    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.08)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)

    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    outlier_df = pd.read_csv(CSV_PATH)
    #outlier_df = outlier_df[:5]
    if le_or_ge == 'le':
        outlier_df = outlier_df[outlier_df["delta"]<=set_delta_threshold]
    elif le_or_ge == 'ge':
        outlier_df = outlier_df[outlier_df["delta"]>=set_delta_threshold]
    else:
        print("ERROR")
    # # Add metrics as columns
    # metrics_df = outlier_df.apply(lambda r: pd.Series(compute_outlier_metrics(r)), axis=1)

    # # Drop non-scalar columns before concatenating (peak_df/list columns)
    # non_scalar_cols = {"peak_df", "peak_alleles", "maj_alleles", "min_alleles"}
    # metrics_scalar = metrics_df.drop(columns=[c for c in metrics_df.columns if c in non_scalar_cols], errors="ignore")

    # outlier_df = pd.concat([outlier_df, metrics_scalar], axis=1)

    # print("Added columns:", list(metrics_scalar.columns))
    # print(outlier_df.head())
    
    # Add ONLY the requested metric columns
    keep_cols = [
        "n_genuine_dropout",
        "n_peak_under_threshold",
        "has_cause_hidden_under_major",
        "has_cause_stutter_wrong_removal",
        "n_dropout_major",
        "n_dropout_minor",
        "n_dropout_total",
        "dropout_check1",
        "dropout_check2",
        #"n_dropin",
        "n_genuine_dropin",
        "n_dropin_stutter",
        "n_dropin_total",
        "dropin_check",
        # "n_minus1_backward",
        # "n_plus1_forward",
        # "n_minus05_backward",
        # "n_plus05_forward",
        # "n_backward_total",
        # "n_forward_total",
        # "n_stutter_total",
    ]
    
    metrics_df = outlier_df.apply(lambda r: pd.Series(compute_outlier_metrics(r)), axis=1)
    outlier_df = pd.concat([outlier_df, metrics_df[keep_cols]], axis=1)
    
    print("Added columns:", keep_cols)


    #Optional save
    #outlier_df.to_csv(BASE / "outlier_with_metrics_minus0point5.csv", index=False)

    # Plot
    print(f"Plotting {len(outlier_df)} outliers (output dir: {OUTLIER_PLOTS_DIR})")
    for _, row in outlier_df.iterrows():
        plot_outlier_row(row, OUTLIER_PLOTS_DIR, use_top_markers=True)
    return outlier_df

outlier_df = main()

# ----------------------------
# SUMMARY COUNTS
# ----------------------------
sum_cols = [
    'n_genuine_dropout',
    'n_peak_under_threshold',
    'has_cause_hidden_under_major',
    'has_cause_stutter_wrong_removal',
    'n_dropout_major',
    'n_dropout_minor',
    'n_dropout_total',
    'dropout_check1',
    'dropout_check2',
    'n_genuine_dropin',
    'n_dropin_stutter',
    'n_dropin_total',
    'dropin_check',
]

col_sums = outlier_df[sum_cols].sum()
n_zero = (outlier_df["n_dropin_stutter"] == 0).sum()
n_nonzero = (outlier_df["n_dropin_stutter"] != 0).sum()
print("\n===== COLUMN SUMS =====")
for col, val in col_sums.items():
    print(f"{col:35s}: {int(val)}")


# # ----------------------------
# # ZERO vs NON-ZERO n_dropout_total
# # ----------------------------
# n_dropout_zero = (outlier_df["n_dropout_total"] == 0).sum()
# n_dropout_nonzero = (outlier_df["n_dropout_total"] != 0).sum()
# total_rows = len(outlier_df)

# pct_dropout_nonzero = 100.0 * n_dropout_nonzero / total_rows if total_rows else 0.0

# print("\n===== n_dropout_total COUNTS =====")
# print(f"Zero values     : {int(n_dropout_zero)}")
# print(f"Non-zero values : {int(n_dropout_nonzero)}")
# print(f"Total rows      : {int(total_rows)}")
# print(f"Percentage non-zero n_dropout_total : {pct_dropout_nonzero:.2f}%")



#outlier_df_copy = outlier_df[outlier_df['mixture']=='1_3A2']

# col_sums_copy = outlier_df_copy[sum_cols].sum()
# n_zero_copy = (outlier_df_copy["n_dropin_stutter"] == 0).sum()
# n_nonzero_copy = (outlier_df_copy["n_dropin_stutter"] != 0).sum()
# print("\n===== COLUMN SUMS COPY =====")
# for col, val in col_sums_copy.items():
#     print(f"{col:35s}: {int(val)}")
# print("\n===== n_dropin_stutter COUNTS COPY =====")
# print(f"Zero values     : {int(n_zero_copy)}")
# print(f"Non-zero values : {int(n_nonzero_copy)}")
# print(f"Total rows      : {int(n_zero_copy + n_nonzero_copy)}")
# #total_rows = len(outlier_df)
# #pct_nonzero = 100.0 * n_nonzero / total_rows if total_rows else 0.0
# print(f"Percentage non-zero n_dropin_stutter : {100.0 * n_nonzero_copy/(n_zero_copy + n_nonzero_copy):.2f}%")

# # ----------------------------
# # ZERO vs NON-ZERO n_dropout_total (subset)
# # ----------------------------
# n_dropout_zero_copy2 = (outlier_df_copy["n_dropout_total"] == 0).sum()
# n_dropout_nonzero_copy2 = (outlier_df_copy["n_dropout_total"] != 0).sum()

# #pct_dropout_nonzero = 100.0 * n_dropout_nonzero / total_rows if total_rows else 0.0

# print("\n===== n_dropout_total COUNTS (subset) =====")
# print(f"Zero values     : {int(n_dropout_zero_copy2)}")
# print(f"Non-zero values : {int(n_dropout_nonzero_copy2)}")
# print(f"Total rows      : {int(n_dropout_zero_copy2+n_dropout_nonzero_copy2)}")
# print(f"Percentage non-zero n_dropout_total : {100.0 * n_dropout_nonzero_copy2/(n_dropout_zero_copy2 + n_dropout_nonzero_copy2):.2f}%")


outlier_df_copy = outlier_df[outlier_df['delta']<-3]
# ----------------------------
# COMBINED DROPOUT / DROP-IN COUNTS (subset)
# ----------------------------
has_dropout = outlier_df_copy["n_dropout_total"] > 0
has_dropin  = outlier_df_copy["n_dropin_total"] > 0

both_dropout_and_dropin = (has_dropout & has_dropin).sum()
dropout_only            = (has_dropout & ~has_dropin).sum()
dropin_only             = (~has_dropout & has_dropin).sum()
neither_dropout_nor_dropin = (~has_dropout & ~has_dropin).sum()

total_rows = len(outlier_df_copy)

print("\n===== DROPOUT vs DROP-IN COMBINED COUNTS (subset) =====")
print(f"Both dropout AND drop-in     : {int(both_dropout_and_dropin)}")
print(f"Dropout only (no drop-in)    : {int(dropout_only)}")
print(f"Drop-in only (no dropout)    : {int(dropin_only)}")
print(f"Neither dropout nor drop-in : {int(neither_dropout_nor_dropin)}")
print(f"Total markers               : {int(total_rows)}")

print("\n===== DROPOUT vs DROP-IN PERCENTAGES (subset) =====")
print(f"Both dropout AND drop-in     : {100*both_dropout_and_dropin/total_rows:.2f}%")
print(f"Dropout only                : {100*dropout_only/total_rows:.2f}%")
print(f"Drop-in only                : {100*dropin_only/total_rows:.2f}%")
print(f"Neither                     : {100*neither_dropout_nor_dropin/total_rows:.2f}%")

col_sums_copy = outlier_df_copy[sum_cols].sum()
n_zero_copy = (outlier_df_copy["n_dropin_stutter"] == 0).sum()
n_nonzero_copy = (outlier_df_copy["n_dropin_stutter"] != 0).sum()
print("\n===== COLUMN SUMS COPY =====")
for col, val in col_sums_copy.items():
    print(f"{col:35s}: {int(val)}")
print("\n===== n_dropin_stutter COUNTS COPY =====")
print(f"Zero values     : {int(n_zero_copy)}")
print(f"Non-zero values : {int(n_nonzero_copy)}")
print(f"Total rows      : {int(n_zero_copy + n_nonzero_copy)}")
#total_rows = len(outlier_df)
#pct_nonzero = 100.0 * n_nonzero / total_rows if total_rows else 0.0
print(f"Percentage non-zero n_dropin_stutter : {100.0 * n_nonzero_copy/(n_zero_copy + n_nonzero_copy):.2f}%")


# # Plot
# print(f"Plotting {len(outlier_df)} outliers (output dir: {OUTLIER_PLOTS_DIR})")
# for _, row in outlier_df_copy.iterrows():
#     plot_outlier_row(row, OUTLIER_PLOTS_DIR, use_top_markers=True)
    
# if __name__ == "__main__":
#     main()

# ----------------------------
# Clean D2S1338 plot without noise
# ----------------------------

# Peak data
alleles = np.array([17, 20, 25])
heights = np.array([200, 2000, 2200])

# Donor allele positions
major_alleles = [20, 25]
minor_alleles = [17, 25]

# Create figure
fig, ax = plt.subplots(figsize=(9, 4.8))

# Plot peaks as stems
markerline, stemlines, baseline = ax.stem(alleles, heights, basefmt=" ")
plt.setp(stemlines, linewidth=1.5)
plt.setp(markerline, markersize=8)

# Add peak height labels
for x, y in zip(alleles, heights):
    ax.text(x, y + 40, f"{int(y)}", ha="center", va="bottom", fontsize=10)

# Plot donor allele markers near the top
top_y = max(heights) * 1.12

ax.scatter(
    major_alleles,
    [top_y] * len(major_alleles),
    marker="^",
    s=80,
    label="Major donor alleles"
)

ax.scatter(
    minor_alleles,
    [top_y] * len(minor_alleles),
    marker="s",
    s=80,
    label="Minor donor alleles"
)

# Axes and layout
ax.set_xlabel("Allele", fontsize=12)
ax.set_ylabel("Peak height", fontsize=12)
ax.set_xticks(alleles)
ax.set_xlim(16, 26)
ax.set_ylim(0, top_y * 1.08)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()

plt.show()


# ----------------------------
# SAME STYLE clean plot
# ----------------------------

# Colors matching your script
COL_PEAK  = "tab:blue"
COL_MAJOR = "tab:green"
COL_MINOR = "tab:brown"

# Data for the clean profile
peak_alleles = np.array([17, 20, 25], dtype=float)
peak_heights = np.array([200, 2000, 2200], dtype=float)

major_alleles = [20, 25]
minor_alleles = [17, 25]

# Create figure
fig, ax = plt.subplots(figsize=(8, 4.8))

# # Match the light grey plotting background look
# ax.set_facecolor("#EAEAF2")
# fig.patch.set_facecolor("#EAEAF2")

# Plot peaks in the same style
for x, y in zip(peak_alleles, peak_heights):
    ax.vlines(x, 0, y, color=COL_PEAK, linewidth=1.5, zorder=2)
    ax.scatter(x, y, color=COL_PEAK, s=55, zorder=3)
    ax.text(x, y + 35, f"{int(y)}", ha="center", va="bottom", fontsize=10)

# Set y-limits first so donor markers can be placed near the top
y_top = float(max(peak_heights)) * 1.20
ax.set_ylim(0, y_top * 1.05)

# Donor allele markers near the top, like in your figure
donor_y = y_top * 0.96

ax.scatter(
    major_alleles,
    [donor_y] * len(major_alleles),
    marker="^",
    s=90,
    color=COL_MAJOR,
    label="Major donor alleles",
    zorder=4
)

ax.scatter(
    minor_alleles,
    [donor_y] * len(minor_alleles),
    marker="s",
    s=80,
    color=COL_MINOR,
    label="Minor donor alleles",
    zorder=4
)

# Axes formatting
ax.set_xlabel("Allele", fontsize=12)
ax.set_ylabel("Peak height", fontsize=12)

ax.set_xticks([17, 20, 25])
ax.set_xlim(15.5, 26.5)

# Grid similar to your current figure
ax.grid(True, which="major", axis="both", color="#D8D8D8", linewidth=0.8, alpha=0.8)

# Legend on the right, same style
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

fig.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# SAME STYLE clean plot
# ----------------------------

COL_PEAK  = "tab:blue"
COL_MAJOR = "tab:green"
COL_MINOR = "tab:brown"

peak_alleles = np.array([17, 18, 19, 20], dtype=float)
peak_heights = np.array([1000,1000, 2000, 2000], dtype=float)

major_alleles = [19, 20]
minor_alleles = [17, 18]

fig, ax = plt.subplots(figsize=(8, 4.8))

# # Match background style
# ax.set_facecolor("#EAEAF2")
# fig.patch.set_facecolor("#EAEAF2")

# Plot peaks
for x, y in zip(peak_alleles, peak_heights):
    ax.vlines(x, 0, y, color=COL_PEAK, linewidth=1.5, zorder=2)
    ax.scatter(x, y, color=COL_PEAK, s=55, zorder=3)
    ax.text(x, y + 35, f"{int(y)}", ha="center", va="bottom", fontsize=10)

# Y-limits
y_top = float(max(peak_heights)) * 1.3
ax.set_ylim(0, y_top * 1.05)

# Donor marker height
donor_y = y_top * 0.96

# Horizontal offset for shared alleles
x_offset = 0.18
shared = set(major_alleles).intersection(minor_alleles)

major_x = []
for a in major_alleles:
    if a in shared:
        major_x.append(a - x_offset)   # shift major left if shared
    else:
        major_x.append(a)

minor_x = []
for a in minor_alleles:
    if a in shared:
        minor_x.append(a + x_offset)   # shift minor right if shared
    else:
        minor_x.append(a)

# Plot donor markers
ax.scatter(
    major_x,
    [donor_y] * len(major_x),
    marker="^",
    s=90,
    color=COL_MAJOR,
    label="Major donor alleles",
    zorder=4
)

ax.scatter(
    minor_x,
    [donor_y] * len(minor_x),
    marker="s",
    s=80,
    color=COL_MINOR,
    label="Minor donor alleles",
    zorder=4
)

# Axes formatting
ax.set_xlabel("Allele", fontsize=12)
ax.set_ylabel("Peak height", fontsize=12)
ax.set_xticks([17, 18, 19,20])
ax.set_xlim(15.5, 21.5)

detection_threshold = 140 

max_y_for_scale = max(float(max(heights)), detection_threshold or 0)
y_top = max_y_for_scale * 1.20
ax.set_ylim(0, y_top * 1.05)

if detection_threshold is not None:
    ax.axhline(
        detection_threshold,
        linestyle="--",
        linewidth=1.2,
        color="red",
        alpha=0.8,
        label="Detection threshold (140 RFU)",
        zorder=2,
    )

# Grid style
ax.grid(True, which="major", axis="both", color="#D8D8D8", linewidth=0.8, alpha=0.8)

# Legend
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

fig.tight_layout()
plt.show()