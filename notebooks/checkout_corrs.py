# %%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# set current directory to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
# %%
# use all the keys in the dictionaries
path_conc = "data/resources/dictionaries/concreteness_brysbaert.json"
with open(path_conc, 'r') as f:
    conc_lexicon = json.load(f)

path_imag = "data/resources/dictionaries/mrc_psychol_dict.json"
with open(path_imag, 'r') as f:
    imag_lexicon = json.load(f)
imag_lexicon = {k: v['imag'] for k, v in imag_lexicon.items()} # fix lexicon

path_sensory = "data/resources/dictionaries/sensorimotor_norms_dict.json"
with open(path_sensory, 'r') as f:
    sensory_lexicon = json.load(f)
audit_lexicon = {k: v['Auditory.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
gust_lexicon = {k: v['Gustatory.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
haptic_lexicon = {k: v['Haptic.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
olfact_lexicon = {k: v['Olfactory.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
intero_lexicon = {k: v['Interoceptive.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
vis_lexicon = {k: v['Visual.mean'] for k, v in sensory_lexicon.items()} # fix lexicon
vis_lexicon

# %%
# since imageability is the smallest lexicon, we take all words that are in it
common_words = set(imag_lexicon.keys())

# make df
corr_df = pd.DataFrame({
    'word': list(common_words),
    'concreteness': [conc_lexicon.get(word, np.nan) for word in common_words],
    'imageability': [imag_lexicon.get(word, np.nan) for word in common_words],
    'auditory': [audit_lexicon.get(word, np.nan) for word in common_words],
    'gustatory': [gust_lexicon.get(word, np.nan) for word in common_words],
    'haptic': [haptic_lexicon.get(word, np.nan) for word in common_words],
    'olfactory': [olfact_lexicon.get(word, np.nan) for word in common_words],
    'interoceptive': [intero_lexicon.get(word, np.nan) for word in common_words],
    'visual': [vis_lexicon.get(word, np.nan) for word in common_words],
})

# remove rows with NaN values
corr_df = corr_df.dropna()
print(len(corr_df), "words with complete data across all lexicons")
print(len(common_words), "total common words")

corr_df

# %%

# CHECK THE ENTROPY/NORM OF EMBEDDINGS compared to these scores & to stylistics????
from src.utils import calculate_entropy
from src.utils import clip_embeds

embeds = clip_embeds(corr_df['word'].tolist())
entry_entropy = [calculate_entropy(embedding) for embedding in embeds]

# add to dataframe
corr_df['embedding_entropy'] = entry_entropy
# %%
# get reload
from importlib import reload
import src.utils
reload(src.utils)

# %%
import os

os.path.exists("/Users/au324704/Desktop/EmotionCLIP/MixCLIP.pth")


# %%
from src.utils import emotionclip_embeds

# try out
texts = ["This picture conveys a sense of awe", "This image suggests fear"]
embeddings = emotionclip_embeds(texts)
# %%
# add to dataframe
corr_df['embedding_entropy'] = entry_entropy

# heatmap again
plt.figure(figsize=(7, 7))
sns.heatmap(corr_df.corr(), annot=True, cbar=False, fmt=".2f")
plt.title(f"Common lemmas (n={len(corr_df)}) across all dictionaries + embedding entropy")
plt.show()
# %%
# save to csv
output_path = "results/common_lexicon_correlations.csv"
corr_df.to_csv(output_path, index=True)
# %%
corr_df.reset_index(inplace=True)

# %%
