# %%

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import krippendorff
import os
import time

# %%
# CONFIG
FIG_FOLDER = "../figs/beta_results/"
os.makedirs(FIG_FOLDER, exist_ok=True)
results_log_path = "../results/beta_results/results_log.txt"
os.makedirs(os.path.dirname(results_log_path), exist_ok=True)

ts = time.strftime("%Y%m%d-%H%M%S")
results_log = open(results_log_path, "a")  # append mode
results_log.write(f"\n=== Run {ts} ===\n")

# %%
df = pd.read_csv("../data/beta_data/2026-01-08_betadata.csv")

# add dummy user_id column from user_name
df["user_id"] = df["user_name"].astype("category").cat.codes
df["user_id"] = df["user_id"].apply(lambda x: f"annotator_{x}")

df.head()
# %%
# get task from dict in annotations column
# make sure that it is not string
def maybe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df["annotations"] = df["annotations"].apply(maybe_eval)
df["value"] = df["annotations"].apply(lambda x: x[0]["value"])
df.tail()

# %%
# subset senses data
senses_df = df[df["workflow_name"] == "Senses Survey"]
# and imageability + concreteness
img_conc = df[df["workflow_name"] != "Senses Survey"]
print(f"Senses data shape: {senses_df.shape}")
print(f"Imageability + Concreteness data shape: {img_conc.shape}")

# %%
# ---- Imageability + Concreteness data cleaning ----

# we remove any additional formatting on "value" column () around numbers, any text etc
def extract_number(x):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(x))
    if match:
        return float(match.group())
    return None

img_conc["value"] = img_conc["value"].apply(extract_number)

# subsetting again
img = img_conc[img_conc["workflow_name"] == "Imageability"]
conc = img_conc[img_conc["workflow_name"] == "Concreteness"]
print(f"Imageability data shape: {img.shape}")
print(f"Concreteness data shape: {conc.shape}")

# %%
dfs = {"imageability": img, "concreteness": conc}
storage = {}

# now we want each subject to be a row, and columns to be annotators
# loop over both dataframes
for key, data in dfs.items():
    pivot_df = data.pivot_table(index=["subject_ids"], columns="user_id", values="value").reset_index().rename_axis(None, axis=1)
    # and we add a mean
    pivot_df[f"{key}_mean"] = pivot_df.drop(columns=["subject_ids"]).mean(axis=1, skipna=True)
    pivot_df["n_annotators"] = pivot_df.drop(columns=["subject_ids"]).count(axis=1)
    storage[key] = pivot_df

    # plot number of annotators per subject
    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid")
    sns.histplot(pivot_df[f"n_annotators"], kde=False, bins=range(1, pivot_df[f"n_annotators"].max() + 2))
    plt.title(f"Number of Annotators per Subject for {key.capitalize()}")
    plt.xlabel("Number of Annotators")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_{key}_n_annotators_distribution.png"))
    plt.show()

# %% 
for key in storage:
    print(f"{key.capitalize()} pivot shape: {storage[key].shape}")
    data = storage[key]
    plt.figure(figsize=(10, 6))
    sns.histplot(data[f"{key}_mean"], kde=True)
    plt.title(f"{key.capitalize()} Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_{key}_ratings_distribution.png"))
    plt.show()


# %%
# alright, let's see how much annotators agree with each other
threshold = 5  # minimum number of annotators to filter on

for key in storage:
    data = storage[key]
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]
    matrix = data[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='interval')
    print(f"Krippendorff's alpha for {key}: {alpha:.4f}")
    # log to results file
    results_log.write(f"Krippendorff's alpha for {key}: {alpha:.4f}\n")
    # add filtered scores
    filtered_data = data[data["n_annotators"] >= threshold]
    filtered_matrix = filtered_data[annotator_cols].to_numpy().T
    filtered_alpha = krippendorff.alpha(reliability_data=filtered_matrix, level_of_measurement='interval')
    print(f"Krippendorff's alpha for {key} (n_annotators >= {threshold}): {filtered_alpha:.4f}")
    results_log.write(f"Krippendorff's alpha for {key} (n_annotators >= 5): {filtered_alpha:.4f}\n")

results_log.close()

# %%
