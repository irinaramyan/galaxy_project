# load labels
import pandas as pd
df = pd.read_csv("/Users/irinaaramyan/Desktop/galaxy_time_machine/training_solutions_rev1.csv")

# convert vote fractions into classes
def get_label(row):
    if row["Class1.1"] > 0.5: return "elliptical"
    if row["Class1.2"] > 0.5: return "spiral"
    if row["Class6.1"] > 0.5: return "merging"
    return "irregular"

df["label"] = df.apply(get_label, axis=1) # applies the functions on all the rows
label_map = {"elliptical": 0, "spiral": 1, "merging": 2, "irregular": 3}
df["label"] = df["label"].map(label_map)
df = df[["GalaxyID", "label"]]