# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import spacy
import pandas as pd
from scipy.stats import shapiro
from tqdm import tqdm
import ftfy
tqdm.pandas()

# get path to current file's parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# set current directory to parent directory
os.chdir(parent_dir)

from src.utils import get_dict_scores

# %%
path = "/Users/au324704/Downloads/2025-10-28_data_from_andrew.csv"

# check the utf-8 problems
df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
print("Total rows before removing replacement chars:", len(df))

replacement_char = "�"
df = df[~df["text"].str.contains(replacement_char, na=False)]
print("Total rows after removing replacement chars:", len(df))
df.head()
# %%
# Compute scores
dict_root = "../resources/dictionaries/"
concreteness_dict = f"{dict_root}concreteness_brysbaert.json"
imageability_dict = f"{dict_root}/mrc_psychol_dict.json"
sensory_dict = f"{dict_root}sensorimotor_norms_dict.json"

sentences = df["text"].tolist()

nlp = spacy.load("en_core_web_sm")

## Concreteness ##
concreteness_total, concreteness_normalized, concreteness_avg = get_dict_scores(
    sentences,
    dict_path=concreteness_dict,
    token_attr='lemma_',
    nlp=nlp)
# add to dataframe
df['concreteness_sum'] = concreteness_total
df['concreteness_normalized'] = concreteness_normalized
df['concreteness_avg_matched'] = concreteness_avg

## Imageability ##
imageability_total, imageability_normalized, imageability_avg = get_dict_scores(
    sentences,
    dict_path=imageability_dict,
    score_key='imag',
    token_attr='lemma_',
    nlp=nlp)
# add to dataframe
df['imageability_sum'] = imageability_total
df['imageability_normalized'] = imageability_normalized
df['imageability_avg_matched'] = imageability_avg

## Sensory modalities ##
# Define sensory modalities and their corresponding keys
modalities = {
    "visuality": "Visual.mean",
    "olfaction": "Olfactory.mean",
    "gustation": "Gustatory.mean",
    "tactile": "Haptic.mean",
    "auditory": "Auditory.mean",
    "interoception": "Interoceptive.mean"}

# Loop through each sensory modality
for modality, score_key in modalities.items():
    total, normalized, averages = get_dict_scores(
        sentences,
        dict_path=sensory_dict,
        score_key=score_key,
        token_attr="lemma_",
        nlp=nlp)
    
    df[f"{modality}_sum"] = total
    df[f"{modality}_normalized"] = normalized
    df[f"{modality}_avg_matched"] = averages

df.head()

# %%
# cleaning
print(len(df))

substrings = ["ISBN", "ebook", "ebooks", "published by", "was published", 
                   "copyright", "all rights reserved", "quoted in", "Press,",
                   "university press", "london:", "gutenberg", 'thank you to',
                   "www.", ".com", ".org", ".net", "edition", "encoding",
                   "visit our website", "project gutenberg", "digital edition",
                   "this etext", "this ebook", "ebook edition", "online edition", "title:"]
# remove all rows w "ISBN"
for substr in substrings:
    df = df[~df["text"].str.contains(substr, case=False, na=False)]
    print(len(df))

# remove rows with regex (p. any digits followed by space/close paren)
regex_patterns = [r'p\.\s*\d+', r'p\)\s*', r'\(p\.\s*\d+', r'chapter\s*\d+']

for pattern in regex_patterns:
    df = df[~df["text"].str.contains(pattern, case=False, na=False, regex=True)]
    print(len(df))
# %%
# fix encoding issues with ftfy
df['text'] = df['text'].apply(ftfy.fix_text)

# fix mojibake
def fix_mojibake(text):
    text = str(text)
    # undo mis-decoding from Latin-1 to UTF-8
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        fixed = text
    return fixed

# Apply with progress bar
df['text'] = df['text'].progress_apply(fix_mojibake)

# %%
# further cleaning
# add a space before ( if not already there
df['text'] = df['text'].apply(lambda x: re.sub(r'(?<!\s)\(', ' (', x))
# remove sentences starting with comma
df = df[~df['text'].str.match(r'^\s*,')]
# remove any * characters
df['text'] = df['text'].str.replace('*', '', regex=False)
# remove rows ending with ft.
df = df[~df['text'].str.endswith('ft.')]
print("Total rows after cleaning:", len(df))
# %%
# now a normalized concreteness threhold above
threshold = 1.05
df = df[df['concreteness_normalized'] > threshold]
print("Total rows after concreteness thresholding:", len(df))

# %%
# save data
# lowercase colnames
df.columns = [col.lower() for col in df.columns]
df.to_csv("../data/org_data_processing/data/DATA_senses_sentences_scored.csv", index=False)

# %%
# reopen it
df = pd.read_csv("../data/org_data_processing/data/DATA_senses_sentences_scored.csv", encoding='utf-8')
df.columns = [col.lower() for col in df.columns]
df.head()
# %%
# clean up the genre column, keep only the first 2 words
df['genre_short'] = df['genre'].astype(str).apply(lambda x: ' '.join(x.split()[:2]))

cols = ['author_gender', 'category', 'genre_short', 'language']
for col in cols:
    print(f"Value counts for {col}:")
    vc = df[col].value_counts()
    print(vc)
    print(vc.keys().tolist())
    print("\n")

# %%
# plot genre counts
vc_above_50 = df['genre_short'].value_counts()[df['genre_short'].value_counts() > 100]
vc_notna = vc_above_50.drop(index='nan')
plt.figure(figsize=(10, 5))
sns.barplot(x=vc_notna.index, y=vc_notna.values)
plt.xticks(rotation=90)
plt.title("Genre Counts (Above 50)")
plt.savefig("../figs/genre_counts_above_100.png")
plt.show()
# %%

plt.figure(figsize=(6, 3))
sns.histplot(df['pubdate'], kde=True)
plt.savefig("../figs/pubdate_distribution.png")
plt.show()
# %%
norm_cols = [col for col in df.columns if '_normalized' in col]
total_cols = [col for col in df.columns if '_sum' in col]
avg_cols = [col for col in df.columns if '_avg_matched' in col]

cols_to_use = norm_cols  # or total_cols
n_cols = len(cols_to_use)
colors = sns.color_palette("husl", n_cols)

# Define 2 rows
n_rows = 4
n_cols_subplot = math.ceil(n_cols / n_rows)  # columns per row

fic_subset = df.loc[df['category'] == 'FIC']
nonfic_subset = df.loc[df['category'] == 'NON']

# subplots
fig, axes = plt.subplots(n_rows, n_cols_subplot, figsize=(5 * n_cols_subplot, 3 * n_rows))
sns.set_style("whitegrid")
axes = axes.flatten()  # for easy indexing
for i, col in enumerate(cols_to_use):
    sns.histplot(fic_subset[col].fillna(0), kde=True, ax=axes[i], color=colors[i])
    sns.histplot(nonfic_subset[col].fillna(0), kde=True, ax=axes[i], color='grey', alpha=0.5)
    axes[i].set_title(col)

    # are these normally distributed?
    stat, p = shapiro(df[col].sample(100, random_state=42))
    print(col, stat, p)
    if p > 0.05:
        normality = "normally distributed"
    else:
        normality = "not normally distributed"
    axes[i].set_xlabel(f"{col.split('_')[0]} ({normality})")

# Hide any extra axes if there are more subplots than columns
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("../figs/senses_distributions_by_category.pdf")
plt.show()

# %%
# lets extract 100 sentences with least and highest concreteness & imageability
n_samples = 50

cols = ['concreteness_normalized', 'imageability_normalized', 'visuality_normalized', 'gustation_normalized', 'tactile_normalized', 'auditory_normalized', 'olfaction_normalized', 'interoception_normalized']

dfs =[]
for col in cols:
    sorted_df = df.sort_values(by=col)
    low_samples = sorted_df.head(n_samples)
    low_samples['type'] = 'low_' + col
    high_samples = sorted_df.tail(n_samples)
    high_samples['type'] = 'high_' + col
    samples_df = pd.concat([low_samples, high_samples], ignore_index=True)
    samples_df = samples_df[['fileid','text', 'type', 'category', 'genre', col]]
    # make sure col is float
    samples_df[col] = samples_df[col].astype(float)
    dfs.append(samples_df)

final_samples_df = pd.concat(dfs, ignore_index=True)
final_samples_df = final_samples_df[['fileid','text', 'type', 'category', 'genre'] + cols]
final_samples_df.to_csv("data/sentence_samples_extreme_senses.csv", index=False, encoding='utf-8')

final_samples_df.head()


# %%