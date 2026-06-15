

# -*- coding: utf-8 -*-
"""
Fill missing alleles across triplicate mixture files.

For each group like:
    1_1A2.txt, 2_1A2.txt, 3_1A2.txt

if an allele appears in one replicate and is missing in another,
the missing replicate gets that allele with height 0.
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# ==========================================================
# SETTINGS
# ==========================================================
mixture_folder = Path(
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\mixtures_2p"
)

output_folder = Path(
    r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\inputs\input_2p_Bayes\mixtures_2p_filled"
)

# ==========================================================
# HELPERS
# ==========================================================
def read_file(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def detect_allele_columns(df):
    allele_cols = [c for c in df.columns if c.startswith("Allele")]
    height_cols = [c for c in df.columns if c.startswith("Height")]
    allele_cols = sorted(allele_cols, key=lambda x: int(x.split()[-1]))
    height_cols = sorted(height_cols, key=lambda x: int(x.split()[-1]))
    return allele_cols, height_cols


def row_to_dict(row, allele_cols, height_cols):
    d = {}

    for a_col, h_col in zip(allele_cols, height_cols):
        allele = str(row[a_col]).strip()
        height = str(row[h_col]).strip()

        if allele != "":
            d[allele] = height if height != "" else "0"

    return d


def rebuild_row(sample_name, marker, allele_dict, max_slots):
    row = {
        "Sample Name": sample_name,
        "Marker": marker
    }

    def allele_sort_key(x):
        try:
            return float(x)
        except:
            return x

    alleles = sorted(allele_dict.keys(), key=allele_sort_key)

    for i in range(max_slots):
        a_col = f"Allele {i+1}"
        h_col = f"Height {i+1}"

        if i < len(alleles):
            allele = alleles[i]
            row[a_col] = allele
            row[h_col] = allele_dict[allele]
        else:
            row[a_col] = ""
            row[h_col] = ""

    return row


def get_rework_mixture(stem: str) -> str:
    return stem.split("_", 1)[1]


def get_replicate_number(stem: str) -> int:
    return int(stem.split("_", 1)[0])


# ==========================================================
# GROUP FILES BY REWORK MIXTURE
# ==========================================================
all_files = list(mixture_folder.glob("*.txt"))

groups = defaultdict(list)
for path in all_files:
    groups[get_rework_mixture(path.stem)].append(path)

output_folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# PROCESS EACH GROUP
# ==========================================================
for rework_mixture, file_paths in groups.items():
    # ensure order: 1_, 2_, 3_
    file_paths = sorted(file_paths, key=lambda p: get_replicate_number(p.stem))

    if len(file_paths) != 3:
        print(f"Skipping {rework_mixture}: expected 3 files, found {len(file_paths)}")
        continue

    print(f"Processing {rework_mixture}: {[p.name for p in file_paths]}")

    dfs = [read_file(path) for path in file_paths]

    allele_cols, height_cols = detect_allele_columns(dfs[0])
    max_slots = len(allele_cols)

    markers = dfs[0]["Marker"].tolist()

    out_dfs = [[] for _ in range(3)]

    for marker in markers:
        rows = []
        for df in dfs:
            r = df[df["Marker"] == marker].iloc[0]
            rows.append(r)

        sample_names = [r["Sample Name"] for r in rows]

        allele_maps = [
            row_to_dict(r, allele_cols, height_cols)
            for r in rows
        ]

        all_alleles = set()
        for d in allele_maps:
            all_alleles.update(d.keys())

        for d in allele_maps:
            for a in all_alleles:
                if a not in d:
                    d[a] = "0"

        for i in range(3):
            new_row = rebuild_row(
                sample_name=sample_names[i],
                marker=marker,
                allele_dict=allele_maps[i],
                max_slots=max_slots
            )
            out_dfs[i].append(new_row)

    # save with original filenames
    for rows, input_path in zip(out_dfs, file_paths):
        out_df = pd.DataFrame(rows)
        out_file = output_folder / input_path.name
        out_df.to_csv(out_file, sep="\t", index=False)

print("Done.")
print("Files written to:", output_folder)