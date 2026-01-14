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
FIG_FOLDER = "../figs/beta_results/"
os.makedirs(FIG_FOLDER, exist_ok=True)
results_log_path = "../results/beta_results/results_log.txt"
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
# subset senses data
senses_df = df[df["workflow_name"] == "Senses Survey"]
# and imageability + concreteness
img_conc = df[df["workflow_name"] != "Senses Survey"]
print(f"Senses data shape: {senses_df.shape}")
print(f"Imageability + Concreteness data shape: {img_conc.shape}")

# %%
# ---- Imageability + Concreteness data cleaning ----

results_log.write("\n--- Imageability + Concreteness Data Analysis ---\n")

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
results_log.write(f"Imageability data shape: {img.shape}\n")
results_log.write(f"Concreteness data shape: {conc.shape}\n")


# %%
# log how many "unsure" ratings there are. Unsure is == 0
unsure_img = img[img["value"] == 0].shape[0]
unsure_conc = conc[conc["value"] == 0].shape[0]
print(f"Number of 'unsure' ratings in Imageability: {unsure_img}")
print(f"Number of 'unsure' ratings in Concreteness: {unsure_conc}")
results_log.write(f"Number of 'unsure' ratings in Imageability: {unsure_img}\n")
results_log.write(f"Number of 'unsure' ratings in Concreteness: {unsure_conc}\n")

# then remove them
img = img[img["value"] != 0]
conc = conc[conc["value"] != 0]

# %%
dfs = {"imageability": img, "concreteness": conc}
storage = {}

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
    plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_{key}_n_annotators_distribution.png"))
    plt.show()

# %%
# log average SD
for key in storage:
    avg_std = storage[key][f"{key}_std"].mean()
    print(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")
    results_log.write(f"Average standard deviation/subject for {key}: {avg_std:.4f}\n")

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

# add space in log
results_log.write("\n--- Krippendorff's alpha results ---\n")

for key in storage:
    data = storage[key]
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]
    matrix = data[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for {key}: {alpha:.4f}")
    # log to results file
    results_log.write(f"Krippendorff's alpha for {key}: {alpha:.4f}\n")
    # add filtered scores
    filtered_data = data[data["n_annotators"] >= threshold]
    filtered_matrix = filtered_data[annotator_cols].to_numpy().T
    filtered_alpha = krippendorff.alpha(reliability_data=filtered_matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for {key} (n_annotators >= {threshold}): {filtered_alpha:.4f}")
    results_log.write(f"Krippendorff's alpha for {key} (n_annotators >= {threshold}): {filtered_alpha:.4f}\n")



# %%

# ---- Senses data cleaning ----
results_log.write("\n--- Senses Data Analysis ---\n")

# explode the list of annotations
long_df = senses_df.explode("value")

# unpack the dicts
def extract_score(x):
    answers = x.get("answers", {})
    if not answers:
        return None
    return next(iter(answers.values()))

long_df["modality"] = long_df["value"].apply(lambda x: x.get("choice"))
long_df["score"] = long_df["value"].apply(extract_score)

# log how many "NONE" modalities there are
none_count = long_df[long_df["modality"] == "NONE"].shape[0]
print(f"Number of 'NONE' modality annotations: {none_count}")
results_log.write(f"Number of 'NONE' modality annotations: {none_count}\n")
# and remove "NONE" modalities
long_df = long_df[long_df["modality"] != "NONE"]

# keep only what you need
long_df = long_df[["subject_ids", "user_id", "modality", "score"]].reset_index(drop=True)
# remove everything that is not a number from score
long_df["score"] = long_df["score"].apply(extract_number)

# lets see the amount of annotations per modality and subject
# add a count column
long_df["count"] = 1
annotation_counts = long_df.groupby(["subject_ids", "modality"])["count"].sum().reset_index()
# make a bar plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=annotation_counts, x="modality", y="count", ci=None)
plt.xlabel("Modality")
plt.ylabel("Number of Annotations")
# rotate x labels
plt.xticks(rotation=45)
plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_senses_annotation_counts.png"))
plt.show()

# let's also see the mean scores per modality using boxplots
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.boxplot(data=long_df, x="modality", y="score")
plt.xlabel("Modality")
plt.ylabel("Scores")
# rotate x labels
plt.xticks(rotation=45)
plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_senses_scores_boxplot.png"))
plt.show()

# and again with distribution plots
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.violinplot(data=long_df, x="modality", y="score", inner="quartile")
plt.xlabel("Modality")
plt.ylabel("Scores")
# rotate x labels
plt.xticks(rotation=45)
plt.savefig(os.path.join(FIG_FOLDER, f"{ts}_senses_scores_violinplot.png"))
plt.show()

# and log mean SD per modality
for modality in long_df["modality"].unique():
    modality_df = long_df[long_df["modality"] == modality]
    sd = modality_df["score"].std()
    print(f"Standard deviation of scores for modality {modality}: {sd:.4f}")
    results_log.write(f"Standard deviation of scores for modality {modality}: {sd:.4f}\n")
# %%
# now, for all subjects ids with count > 1, we want the krippendorff's alpha per modality
filtered_long_df = long_df.merge(annotation_counts, on=["subject_ids", "modality"])
filtered_long_df = filtered_long_df[filtered_long_df["count_y"] > 1]
# and remove any rows with NaN scores
filtered_long_df = filtered_long_df.dropna(subset=["score"])
print(f"Filtered senses data shape: {filtered_long_df.shape}")
print(f"original senses data shape: {long_df.shape}")

results_log.write("\n--- Krippendorff's alpha results ---\n")

modalities = filtered_long_df["modality"].unique()
for modality in modalities:
    modality_df = filtered_long_df[filtered_long_df["modality"] == modality]
    pivot_df = modality_df.pivot_table(index=["subject_ids"], columns="user_id", values="score").reset_index().rename_axis(None, axis=1)
    annotator_cols = [col for col in pivot_df.columns if col.startswith("annotator_")]
    matrix = pivot_df[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for modality {modality}: {alpha:.4f}")
    results_log.write(f"Krippendorff's alpha for modality {modality}: {alpha:.4f}\n")

results_log.write("\n--- Krippendorff's alpha results (filtered by annotator count) ---\n")
threshold = 3  # minimum number of annotators to filter on
filtered_long_df_high_annotators = filtered_long_df.loc[filtered_long_df["count_y"] >= threshold]
# print number of unique subject_ids and modalities after filtering
results_log.write(f"Number of unique subject_ids after filtering: {filtered_long_df_high_annotators['subject_ids'].nunique()}\n")
results_log.write(f"Number of unique modalities after filtering: {filtered_long_df_high_annotators['modality'].nunique()}\n")
for modality in modalities:
    modality_df = filtered_long_df_high_annotators[filtered_long_df_high_annotators["modality"] == modality]
    pivot_df = modality_df.pivot_table(index=["subject_ids"], columns="user_id", values="score").reset_index().rename_axis(None, axis=1)
    if pivot_df.shape[0] < 2 or len(annotator_cols) < 2:
        print(f"Skipping modality {modality}: not enough subjects or annotators")
        continue
    annotator_cols = [col for col in pivot_df.columns if col.startswith("annotator_")]
    matrix = pivot_df[annotator_cols].to_numpy().T  # Transpose to have annotators as rows
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level_of_measurement)
    print(f"Krippendorff's alpha for modality {modality} (n_annotators >= {threshold}): {alpha:.4f}")
    results_log.write(f"Krippendorff's alpha for modality {modality} (n_annotators >= {threshold}): {alpha:.4f}\n")

results_log.close()

# %%

# Paivio (1968)(doi:10.1037/h0025327) split raters into two balanced subgroups, averaged ratings within each subgroup, and then correlated these means; high correlations (r ≈ 0.94) indicated strong inter-rater reliability.
# implement a Paivio-style inter-group reliability check

threshold = 0  # only include items with at least this many ratings
n_iterations = 50
tmp = []

for key in ["imageability", "concreteness"]:
    data = storage[key]
    annotator_cols = [col for col in data.columns if col.startswith("annotator_")]

    # Keep items with at least 'threshold' annotators
    filtered_data = data[data["n_annotators"] >= threshold].copy()
    print(f"Number of items {key}: {filtered_data.shape[0]}")
    
    # Get the list of annotators
    annotators = filtered_data[annotator_cols].columns.tolist()
    print(f"Number of annotators for {key}: {len(annotators)}")

    for i in range(n_iterations):
        # Randomly shuffle annotators and split into two groups
        np.random.seed(42)  # reproducibility
        np.random.shuffle(annotators)
        mid = len(annotators) // 2
        group1 = annotators[:mid]
        group2 = annotators[mid:]
        
        # Compute mean per item in each group
        mean1 = filtered_data[group1].mean(axis=1)
        mean2 = filtered_data[group2].mean(axis=1)
        
        # Correlate the two sets of means
        r = mean1.corr(mean2, method='pearson')
        tmp.append({"iteration": i, "key": key, "r": r})
# convert to dataframe
irr_df = pd.DataFrame(tmp)
# summarize results
for key in ["imageability", "concreteness"]:
    subset = irr_df[irr_df["key"] == key]
    r = subset["r"].mean()
    print(f"Paivio-style IRR {key}: r = {r:.4f}")

# %%
