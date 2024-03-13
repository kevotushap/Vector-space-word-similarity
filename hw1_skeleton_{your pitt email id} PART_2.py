#PART 2

import os
import csv
import re
from collections import defaultdict
import numpy as np

def load_snli_corpus(data_file):
    """Loads text from the SNLI corpus into a format similar to the Shakespeare corpus.

    Args:
        data_file (str): Path to the SNLI corpus file.

    Returns:
        list: A list of tuples, each containing the sentence ID and tokenized sentence.
    """
    snli_data = []

    with open(data_file, newline='', encoding='utf-8') as csvfile:
        snli_reader = csv.DictReader(csvfile, delimiter='\t')
        for row in snli_reader:
            sentence_id = row['sentence1_parse'].split()[1][1:-1]  # Extract sentence ID from parse tree
            sentence = re.sub(r'[^a-zA-Z0-9\s]', '', row['sentence1']).lower().split()  # Tokenize and lowercase sentence
            snli_data.append((sentence_id, sentence))

    return snli_data

def build_term_context_matrix(snli_data, vocab, context_window_size=2):
    """Builds a term-context matrix from the SNLI corpus.

    Args:
        snli_data (list): A list of tuples containing sentence ID and tokenized sentence.
        vocab (list): A list of words in the vocabulary.
        context_window_size (int): Size of the context window.

    Returns:
        numpy.ndarray: Term-context matrix.
    """
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    term_context_matrix = defaultdict(lambda: defaultdict(int))

    for _, sentence in snli_data:
        for i, target_word in enumerate(sentence):
            start_index = max(0, i - context_window_size)
            end_index = min(len(sentence), i + context_window_size + 1)

            for j in range(start_index, end_index):
                if j != i:
                    context_word = sentence[j]
                    if target_word in word_to_index and context_word in word_to_index:
                        term_context_matrix[word_to_index[target_word]][word_to_index[context_word]] += 1

    return np.array([[value for value in row.values()] for row in term_context_matrix.values()])

def compute_ppmi_matrix(term_context_matrix):
    """Computes the Positive Pointwise Mutual Information (PPMI) matrix.

    Args:
        term_context_matrix (numpy.ndarray): Term-context matrix.

    Returns:
        numpy.ndarray: PPMI matrix.
    """
    total_word_count = term_context_matrix.sum()
    total_context_count = term_context_matrix.sum(axis=0)

    # Calculate PPMI
    ppmi_matrix = np.log(term_context_matrix * total_word_count / (total_context_count[:, None] * total_word_count))

    # Set negative values to 0
    ppmi_matrix[ppmi_matrix < 0] = 0

    return ppmi_matrix

# Load SNLI corpus
snli_data = load_snli_corpus('snli_corpus.txt')

# Extract vocabulary from SNLI corpus
vocab = set(word for _, sentence in snli_data for word in sentence)

# Build term-context matrix
term_context_matrix = build_term_context_matrix(snli_data, list(vocab))

# Compute PPMI matrix
ppmi_matrix = compute_ppmi_matrix(term_context_matrix)

# Function to get the top associated context words by PPMI for an identity label
def get_top_associated_words(identity_label, ppmi_matrix, vocab, top_n=10):
    idx = vocab.index(identity_label)
    associated_indices = np.argsort(ppmi_matrix[idx])[::-1][:top_n]
    associated_words = [(vocab[i], ppmi_matrix[idx][i]) for i in associated_indices]
    return associated_words

# Print top associated context words for selected identity labels
selected_identity_labels = ["woman", "man", "girl", "boy"]

for identity_label in selected_identity_labels:
    print(f"\nTop 10 associated context words for '{identity_label}':")
    associated_words = get_top_associated_words(identity_label, ppmi_matrix, list(vocab), top_n=10)
    for word, pmi in associated_words:
        print(f"{word}: {pmi}")


# Function to find document contexts with 1st-order similarity
def find_first_order_similarity(identity_label, associated_word, snli_data):
    documents = []
    for sentence_id, sentence in snli_data:
        if identity_label in sentence and associated_word in sentence:
            documents.append((sentence_id, sentence))
    return documents

# Function to find document contexts with 2nd-order similarity
def find_second_order_similarity(identity_label, associated_word, snli_data):
    documents = []
    for sentence_id, sentence in snli_data:
        if identity_label in sentence:
            for word in sentence:
                if word != identity_label:
                    if word in ppmi_matrix[vocab.index(associated_word)]:
                        documents.append((sentence_id, sentence))
                        break
    return documents

# Define pairs of identity terms and associated words
pairs = [
    ("woman", "wearing"),
    ("man", "father"),
    ("girl", "playing"),
    ("boy", "running")
]

# Find contexts for each pair
for identity_label, associated_word in pairs:
    print(f"\nIdentity Label: {identity_label}, Associated Word: {associated_word}")
    
    # Find document contexts with 1st-order similarity
    print("\n1st-order similarity:")
    first_order_similar_documents = find_first_order_similarity(identity_label, associated_word, snli_data)
    for sentence_id, sentence in first_order_similar_documents[:3]:  # Print first 3 results
        print(f"Sentence ID: {sentence_id}, Sentence: {' '.join(sentence)}")
    
    # Find document contexts with 2nd-order similarity
    print("\n2nd-order similarity:")
    second_order_similar_documents = find_second_order_similarity(identity_label, associated_word, snli_data)
    for sentence_id, sentence in second_order_similar_documents[:3]:  # Print first 3 results
        print(f"Sentence ID: {sentence_id}, Sentence: {' '.join(sentence)}")
        