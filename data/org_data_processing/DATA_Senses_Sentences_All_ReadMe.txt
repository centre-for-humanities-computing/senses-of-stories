ReadMe

Each row contains a single sentence sampled from a work of fiction or nonfiction, with associated metadata. 

Sentences were:
Between 10 and 40 tokens long
Filtered to remove dialogue

Sampling Strategy by Subset:
C20: 2 sentences per book
CONLIT: 5 sentences per book
WORLDLIT: 15 sentences per book
FANFIC: 1 sentence per story

Sentences by Category (Class2):
C20: 18,178
CON_FIC: 9,670
CON_NON: 4,100
FANFIC: 5,268
WORLDLIT: 5,055

Sources:
C20: https://textual-optics-lab.uchicago.edu/restricted
WORLDLIT: https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.248
CONLIT: https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.88
FANFIC: https://www.fanfiction.net [stories sampled from a single year: 2016. Only stories in English, between 1000 and 5000 tokens in length. Removed all Mature stories. 179 total "canons", max 200 stories per canon.]

Key Columns:
[Not all columns are populated for all subsets.]

fileid: Unique file identifier
text: The sampled sentence
class: Original category (C20, CONLIT, etc.)
class2: Expanded category (CON_FIC, CON_NON, etc.)
sentence_length: total words of sampled sentence
category, Genre, Genre2: Subclassifications (esp. for CONLIT)
pubdate, Language, Translation, PubHouse: Publication info
author_first, Author_Last, Author_Gender, Author_Nationality
country: Country (WORLDLIT only: CA, IN, NG, SA)
reviews, Favs, Follows, Community.Name: Fanfiction metadata
goodreads_avg, total_ratings, goodreads_URL: Goodreads data
token_count: Number of tokens in the original source book
total_characters: Number of characters in the original book
avg_sentence_length, avg_word_length: Book-level stats
event_count, speed_avg, speed_min, circuitousness, volume: Narrative pace and structure
tuldava_score: Coherence/readability metric book level
probability1p: Likelihood of first-person narration book level

[added by pascale]:
[imageability/concreteness/auditory/gustatory/haptic/interoceptive/visual/olfactory]_sum: sum of scores in the corresponding dictionary per sentence
[imageability/concreteness/auditory/gustatory/haptic/interoceptive/visual/olfactory]_normalized: sum of scores in dictionary normalized by sentence length
[imageability/concreteness/auditory/gustatory/haptic/interoceptive/visual/olfactory]_avg_matched: sum of scores per sentence averaged over number of words with scores in sentence
pred_sensitive: score in the "sexual" content label of roberta model (https://huggingface.co/cardiffnlp/twitter-roberta-base-sensitive-multilabel)
pred_profanity: score in the "profanity" label of the (above) roberta model


Not all columns are populated for all subsets.

