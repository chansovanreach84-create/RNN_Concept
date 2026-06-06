from itertools import count
import re
import numpy as np
import pandas as pd


corpus = [
    "The cat sat on the mat.",
    "The dog chased the cat.",
    "The bird flew over the mat.",
    "Cats and dogs are pets.",
    "The mat was soft."
]

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

print("Tokens for first sentence:", clean_and_tokenize(corpus[0]))
total_words = 0
for i in corpus:
  tokens = clean_and_tokenize(i)
  total_words += len(tokens)
print(total_words)
unique_words = sorted(list(set(word for sentence in corpus for word in clean_and_tokenize(sentence))))

vocab = {word: idx for idx, word in enumerate(unique_words)}

print(f"Vocabulary Size: {len(vocab)}")
print("Vocabulary Mapping:", vocab)
bow_matrix = np.zeros((len(corpus), len(vocab)), dtype=int)

for doc_idx, sentence in enumerate(corpus):
    tokens = clean_and_tokenize(sentence)
    for token in tokens:
        if token in vocab:
            word_idx = vocab[token]
            bow_matrix[doc_idx, word_idx] += 1

print("BoW Matrix Shape:", bow_matrix.shape)
print(bow_matrix)
words_per_doc = bow_matrix.sum(axis=1, keepdims=True)
tf_matrix = bow_matrix / words_per_doc

N = len(corpus)
document_frequency = np.sum(bow_matrix > 0, axis=0)
idf = np.log(N / (1 + document_frequency))

tfidf_matrix = tf_matrix * idf

print("TF-IDF Matrix calculation complete.")
tfidf_df = pd.DataFrame(tfidf_matrix, columns=unique_words, index=[f"Doc {i+1}" for i in range(N)])