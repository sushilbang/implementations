# -*- coding: utf-8 -*-
"""N_gram.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bOw3F6l9qUYBhjK1JQHJpb_jHI4tXiBy
"""

with open("n_gram_text.txt") as f:
  text = f.read()

"""Clean before use"""

import re

def clean(text):
  # Lowercase the complete text
  text = text.lower()
  # Remove punctuations
  text = re.sub(r'[^\w\s]', '', text)
  # Remove numbers
  text = re.sub(r'\d+', '', text)
  # Remove extra white spaces
  text = re.sub(r'\s+', ' ', text).strip()

  return text
text = clean(text)

"""Tokenization(lame)"""

def tokenize(text):
  return text.split()

tokens = tokenize(text)

"""Generate N grams"""

def generate_n_grams(n, tokens):
  return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

bigram = generate_n_grams(2, tokens)
trigram = generate_n_grams(3, tokens)

"""Calc frq?"""

def freq(grams):
  f = {}
  for gram in grams:
    if gram in f:
      f[gram] += 1
    else :
      f[gram] = 1
  return f

frq_bi = freq(bigram)
frq_tri = freq(trigram)

"""Build transition probs"""

def build_transition_table(ngram_freq, n):
  transition_probs = {}
  for ngram, count in ngram_freq.items():
    context = ngram[:-1]
    next_word = ngram[-1]
    if context not in transition_probs:
      transition_probs[context] = {}
    transition_probs[context][next_word] = transition_probs.get(next_word, 0) + count
  return transition_probs

bi_probs = build_transition_table(frq_bi, 2)
tri_probs = build_transition_table(frq_tri, 3)

"""predicting next word"""

def predict_next_word(context, transition_probs, n):
  context = clean(context)
  context_tokens = tokenize(context)
  context_len = n-1

  context = tuple(context_tokens[-context_len:]) if len(context_tokens) >= context_len else tuple()

  possible = transition_probs.get(context, {})
  return max(possible, key=possible.get) if possible else "no possible next word"

next_word = predict_next_word('use of ngrams', tri_probs, 3)

next_word