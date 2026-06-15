# -*- coding: utf-8 -*
import pandas as pd

# df = pd.read_csv(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\Dataset PP6FC Mixtures\Donoren\1A.csv", sep = ';')
# df.to_csv(r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources\1A.txt", sep=",", index=False)

#import pandas as pd

# paths
reference_file = r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources\reference_file.txt"
#input_txt = r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources\1A.txt"
input_csv = r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\Dataset PP6FC Mixtures\Donoren\2G.csv"
output_txt = r"C:\Users\jortk\OneDrive\OneDrive Documenten\Thesis\Code_new\resources\2G_reference_format_ordered.txt"

# read files
ref = pd.read_csv(reference_file)
df = pd.read_csv(input_csv, sep = ';')

# extract reference sample name (first row)
reference_sample_name = ref.loc[0, "Sample Name"]

# marker order from reference
markers = ref["Marker"]

# index 1B data by marker
df = df.set_index("Marker")

rows = []

for marker in markers:
    if marker in df.index:
        allele1 = df.loc[marker, "Allele1"]
        allele2 = df.loc[marker, "Allele2"]
    else:
        allele1 = ""
        allele2 = ""

    rows.append({
        "Sample Name": reference_sample_name,
        "Marker": marker,
        "Allele 1": allele1,
        "Height 1": 1000 if allele1 != "" else "",
        "Allele 2": allele2,
        "Height 2": 1000 if allele2 != "" else ""
    })

out = pd.DataFrame(rows)
# clean allele formatting: 13.0 -> 13, keep 18.3 as is
for col in ["Allele 1", "Allele 2"]:
    out[col] = out[col].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else x
    )

# write output
out.to_csv(output_txt, index=False)
