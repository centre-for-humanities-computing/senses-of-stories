# %%

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import krippendorff
import os
import time
import itertools
import numpy as np

# %%
# CONFIG
FIG_FOLDER = "../figs/beta_results/simulate_scales/"
os.makedirs(FIG_FOLDER, exist_ok=True)
results_log_path = "../results/beta_results/simulate_log.txt"
os.makedirs(os.path.dirname(results_log_path), exist_ok=True)

ts = time.strftime("%Y%m%d-%H")
results_log = open(results_log_path, "a")  # append mode
results_log.write(f"\n=== Run {ts} ===\n")

level_of_measurement = 'ordinal'  # or interval for krippendorff's alpha
results_log.write(f"Level of measurement for Krippendorff's alpha: {level_of_measurement}\n")

# %%
df = pd.read_csv("../data/beta_data/2026-01-13_the-senses-of-stories-classifications.csv")#../data/beta_data/2026-01-08_betadata.csv")

# drop everything created before 2025-11-26 13:15:19 UTC
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
cutoff_date = pd.to_datetime("2025-11-26 13:15:19", utc=True)
df = df[df["created_at"] >= cutoff_date]

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
# get imageability + concreteness
img_conc = df[df["workflow_name"] != "Senses Survey"]
print(f"Imageability + Concreteness data shape: {img_conc.shape}")

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

# remove unsure responses (value == 0)
img = img[img["value"] != 0]
conc = conc[conc["value"] != 0]

# %%
dfs = {"imageability": img, "concreteness": conc}
storage = {}

results_log.write("\n")

# now we want each subject to be a row, and columns to be annotators
# loop over both dataframes
for key, data in dfs.items():
    pivot_df = data.pivot_table(index=["subject_ids"], columns="user_id", values="value").reset_index().rename_axis(None, axis=1)
    # and we add a mean
    pivot_df[f"{key}_mean"] = pivot_df.drop(columns=["subject_ids"]).mean(axis=1, skipna=True)
    # and sd
    pivot_df[f"{key}_std"] = pivot_df.drop(columns=["subject_ids"]).std(axis=1, skipna=True)
    pivot_df["n_annotators"] = pivot_df.drop(columns=["subject_ids"]).count(axis=1)
    storage[key] = pivot_df

    # plot number of annotators per subject
    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid")
    sns.histplot(pivot_df[f"n_annotators"], kde=False, bins=range(1, pivot_df[f"n_annotators"].max() + 2))
    plt.title(f"Number of Annotators per Subject for {key.capitalize()}")
    plt.xlabel("Number of Annotators")
    plt.ylabel("Frequency")
    plt.show()

# %%
threshold = 5  # minimum number of annotators to filter on

# log average SD
for key in storage:
    data = storage[key]
    # plot distribution of mean ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(data[f"{key}_mean"], kde=True)
    plt.title(f"{key.capitalize()} Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()
    avg_std = data[f"{key}_std"].mean()
    print(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")
    results_log.write(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")


    # add space in log
    results_log.write(f"--- {key.upper()} ---\n")
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]
    matrix = data[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for {key}: {alpha:.4f}")
    # log to results
    results_log.write(f"alpha for {key}: {alpha:.4f}\n")

    # add filtered scores
    filtered_data = data[data["n_annotators"] >= threshold]
    filtered_matrix = filtered_data[annotator_cols].to_numpy().T
    filtered_alpha = krippendorff.alpha(reliability_data=filtered_matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for {key} (n_annotators >= {threshold}): {filtered_alpha:.4f}")
    results_log.write(f"Krippendorff's alpha for {key} (n_annotators >= {threshold}): {filtered_alpha:.4f}\n")

# %%

storage_5pt = {}
for key, data in storage.items():
    data_5pt = data.copy()
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]
    
    # map 7->5
    map_7_to_5 = {1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:5}
    for col in annotator_cols:
        data_5pt[col] = data_5pt[col].map(map_7_to_5)
    
    # recompute mean and std based on new 5-point values
    data_5pt[f"{key}_mean"] = data_5pt[annotator_cols].mean(axis=1)
    data_5pt[f"{key}_std"] = data_5pt[annotator_cols].std(axis=1)
    
    storage_5pt[key] = data_5pt

results_log.write("\n")
results_log.write(f"--- Forced to 5-point scale ---\n")

# log average SD
for key in storage_5pt:
    data = storage_5pt[key]
    # plot distribution of mean ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(data[f"{key}_mean"], kde=True)
    plt.title(f"{key.capitalize()} Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()
    avg_std = data[f"{key}_std"].mean()
    print(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")
    results_log.write(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")


    # add space in log
    results_log.write(f"--- {key.upper()} ---\n")
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]
    matrix = data[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level_of_measurement)
    print(f"alpha for {key}: {alpha:.4f}")
    # log to results
    results_log.write(f"alpha for {key}: {alpha:.4f}\n")

    # add filtered scores
    filtered_data = data[data["n_annotators"] >= threshold]
    filtered_matrix = filtered_data[annotator_cols].to_numpy().T
    filtered_alpha = krippendorff.alpha(reliability_data=filtered_matrix, level_of_measurement=level_of_measurement)
    print(f"when (n_annotators >= {threshold}): {filtered_alpha:.4f}")
    results_log.write(f"when (n_annotators >= {threshold}): {filtered_alpha:.4f}\n")

results_log.close()

# %%
