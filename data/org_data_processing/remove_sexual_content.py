# %%
import pandas as pd
from transformers import pipeline
import time

# %%
# timestamp
ts = time.strftime("%Y-%m-%d_%H-%M")
print("Timestamp:", ts)

# reopen it
df = pd.read_csv("DATA_senses_sentences_scored.csv")
df.head()
# %%

pipe = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sensitive-multilabel",
    top_k=None,
    truncation=True  # ensures long texts don’t break
)

texts = df["text"].tolist()

# batch prediction
all_preds = pipe(texts, batch_size=32)  # adjust batch size depending on GPU/CPU

# extract 'sex' label scores
sex_scores, swear_scores = [], []

for preds in all_preds:
    # if nested list, unpack
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
        preds = preds[0]

    # if preds is now a list of dicts
    if isinstance(preds, list) and all(isinstance(x, dict) for x in preds):
        sex_score = next((x["score"] for x in preds if x["label"] == "sex"), 0.0)
        swear_score = next((x["score"] for x in preds if x["label"] == "profanity"), 0.0)
    else:
        sex_score = 0.0  # fallback
        swear_score = 0.0  # fallback

    sex_scores.append(sex_score)
    swear_scores.append(swear_score)


df["pred_sensitive"] = sex_scores
df["pred_profanity"] = swear_scores

# %%
# save
df.to_csv(f"{ts}_DATA_senses_sentences_scored_labelled.csv", index=False, encoding='utf-8')
df.head()

# %%
# lets see how this works
sex_threshold = 0.8
profanity_threshold = 0.8

for text in df[df["pred_sensitive"] > sex_threshold]["text"].tolist():
    print(text)
    print("Score:", df[df["text"] == text]["pred_sensitive"].values[0])
    print("-----")

# %%
for text in df[df["pred_profanity"] > profanity_threshold]["text"].tolist():
    print(text)
    print("Score:", df[df["text"] == text]["pred_profanity"].values[0])
    print("-----")
# %%
print(len(df[df["pred_sensitive"] > sex_threshold]))
len(df[df["pred_profanity"] > profanity_threshold])
# %%
polished = df[df["pred_sensitive"] <= sex_threshold]
polished = polished[polished["pred_profanity"] <= profanity_threshold]

polished = polished[~polished["text"].str.contains("Press,", case=False, na=False)]

# %%
print("Total rows after removing sexual/profanity content:", len(polished))
polished.to_csv(f"{ts}_senses_stories_final_set.csv", index=False, encoding='utf-8')
# %%


# reopen it
df = pd.read_csv(f"2025-11-02_15-38_senses_stories_final_set.csv", encoding='utf-8')
df.head()
# %%

# just a litte more cleaning

# remove any * characters
df['text'] = df['text'].str.replace('*', '', regex=False)
# also multiple
df['text'] = df['text'].str.replace('**', '', regex=False)
# remove rows ending with ft.
df = df[~df['text'].str.endswith('ft.')]
print("Total rows after cleaning:", len(df))

# %%
def fix_mojibake(text):
    text = str(text)
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # handle already-decoded wrong characters
    fixes = {
        " ¬∞ ": "ó",   # or "ò" depending on the source
        "√º": "ú",
        "√±": "ñ",
        "√≥": "ó",
        "√©": "é",
        "√§": "ç",
        "√®": "î",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    return text

df['text'] = df['text'].apply(fix_mojibake)
# %%
df
# %%
text = "Wickedly yours, Margaret Where are the mountains of M ¬∞ rona?"
print(fix_mojibake(fix_mojibake(text)))
# %%
# and remove three *** 
df['text'] = df['text'].str.replace('*** ', '', regex=False)
# remove stuff ending with : or ;
df = df[~df['text'].str.endswith((':', ';'))]
# %%
import time
ts = time.strftime("%Y-%m-%d_%H-%M")
print(len(df))
df.to_csv(f"{ts}_senses_stories_cleaned.csv", index=False, encoding='utf-8')

# %%