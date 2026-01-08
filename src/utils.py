import json
import numpy as np
import spacy

def get_dict_scores(sentence_list, dict_path, score_key=None, token_attr='lemma_', nlp=spacy.load("en_core_web_sm")):
    """
    Computes scores (e.g., imageability, visuality, concreteness) for a list of sentences using a given lexicon.

    Args:
        texts (list of str): List of sentences to process.
        dict_path (str): Path to the JSON lexicon file.
        score_key (str): Key in the lexicon for extracting scores (e.g., 'imag', 'Visual.mean').
        token_attr (str): Token attribute to match with the lexicon keys (default: 'lemma_').
        normalize_by_tokens (bool): Whether to normalize scores by the total number of tokens (default: True).
            otherwise, normalize by the number of valid tokens with scores.

    Returns:
        tuple: A tuple of two lists:
            - total_scores (list of float): Total scores for each sentence.
            - normalized_scores (list of float): Normalized scores by sentence length for each sentence.
            - avg_score (float): Average score across all dictionary-matched tokens in sentence.
    """
        
    with open(dict_path, 'r') as f:
        lexicon = json.load(f)
    lexicon = {k.lower(): v for k, v in lexicon.items()} # fix lexicon
    print(f'Loaded lexicon for scoring from {dict_path}, len of lexicon:', len(lexicon))

    piped_sentences = nlp.pipe(sentence_list)

    total_scores, normalized_scores, avg_matched_scores = [], [], []

    for sentence in piped_sentences:
        valid_scores = []

        for token in sentence:
            attr_val = getattr(token, token_attr).lower()
            if attr_val in lexicon: # get score from lexicon
                val = lexicon[attr_val]
                if isinstance(val, dict): # in case of multiple scores
                    # check that score_key is provided
                    if score_key is None:
                        raise ValueError("score_key must be provided when lexicon values are dictionaries.")
                    val = val.get(score_key)
                if val is not None:
                    valid_scores.append(val)

        total_score_sentence = sum(valid_scores)
        normalized_score_sentence = total_score_sentence / len(sentence) if sentence else 0
        avg_matched_score_sentence = total_score_sentence / len(valid_scores) if valid_scores else 0

        # Append scores to lists
        total_scores.append(total_score_sentence)
        normalized_scores.append(normalized_score_sentence)
        avg_matched_scores.append(avg_matched_score_sentence)

    return total_scores, normalized_scores, avg_matched_scores


# clip model embeds

import numpy as np
import os
from scipy.stats import skew, kurtosis
from scipy.stats import entropy

import json

from transformers import CLIPProcessor, CLIPModel
import torch

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

def clip_embeds(words):
    inputs = processor(text=words, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        entry_embeddings = model.get_text_features(**inputs)
    # convert to numpy
    entry_embeddings = entry_embeddings.cpu().numpy()
    return entry_embeddings

# entropies
def prob(embedding):
    abs_embedding = np.abs(embedding)  # Ensure non-negative values
    total = np.sum(abs_embedding)
    return abs_embedding / total if total != 0 else np.ones_like(embedding) / len(embedding)

def calculate_entropy(embedding):
    return entropy(prob(embedding))

def softmax_entropy(embedding, temperature=1.0):
    exp_vals = np.exp(embedding / temperature)
    probs = exp_vals / np.sum(exp_vals)
    return entropy(probs)


# Emotionclip
import sys
sys.path.append("/Users/au324704/Desktop/EmotionCLIP")
from EmotionCLIP import model, tokenizer
import torch

def emotionclip_embeds(texts):
    """
    Get text embeddings from EmotionCLIP for a list of texts.
    """
    # Tokenize text
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Compute embeddings
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    
    # Normalize and return as numpy
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds.cpu().numpy()
