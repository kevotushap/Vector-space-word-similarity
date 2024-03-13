import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy
import pandas as pd
from collections import defaultdict


def read_in_shakespeare():
    """Reads in the Shakespeare dataset and processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open("shakespeare_plays.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    """A convenience function to get a particular row vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      row_id: an integer row_index for the desired row vector

    Returns:
      1-dimensional numpy array of the row vector
    """
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    """A convenience function to get a particular column vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      col_id: an integer col_index for the desired row vector

    Returns:
      1-dimensional numpy array of the column vector
    """
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
# List of Shakespeare's plays
plays = [
    "Henry IV", "Alls well that ends well", "Loves Labours Lost", "Taming of the Shrew",
    "Antony and Cleopatra", "Coriolanus", "Hamlet", "A Midsummer nights dream",
    "Merry Wives of Windsor", "Romeo and Juliet", "Richard II", "King John",
    "macbeth", "Timon of Athens", "A Winters Tale", "The Tempest", "Henry VI Part 2",
    "As you like it", "Julius Caesar", "A Comedy of Errors", "Henry VIII", 
    "Measure for measure", "Richard III", "Two Gentlemen of Verona", "Henry VI Part 1",
    "Much Ado about nothing", "Henry V", "Troilus and Cressida", "Twelfth Night",
    "Merchant of Venice", "Henry VI Part 3", "Othello", "Cymbeline", "King Lear",
    "Pericles", "Titus Andronicus"
]

# Create an empty DataFrame to store the term-document matrix
term_document_matrix = pd.DataFrame(columns=plays)

# Read the text of each play and count word occurrences
for play in plays:
    with open(f"shakespeare_plays/{play}.txt", "r") as file:
        text = file.read()
        # Tokenize the text (you might need more sophisticated tokenization)
        words = text.split()
        # Count word occurrences
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        # Add word counts to the term-document matrix
        term_document_matrix = term_document_matrix.append(word_counts, ignore_index=True)

# Fill missing values (NaN) with zeros
term_document_matrix.fillna(0, inplace=True)

# Save the term-document matrix to a CSV file
term_document_matrix.to_csv("term_document_matrix.csv", index=False)


    # Load the dataset
    df = pd.read_csv(data_file)
    
    # Initialize an empty dictionary to store word frequencies for each document
    term_doc_matrix = {}
    
    # Iterate over each row (play) in the dataframe
    for index, row in df.iterrows():
        # Extract the play name and text content
        play = row['Play']
        text = row['PlayerLine']
        
        # Tokenize the text and lowercase each word
        words = text.lower().split()
        
        # Iterate over each word in the tokenized text
        for word in words:
            # Increment the frequency count for the current word in the current play
            if word not in term_doc_matrix:
                term_doc_matrix[word] = {play: 1}
            else:
                if play not in term_doc_matrix[word]:
                    term_doc_matrix[word][play] = 1
                else:
                    term_doc_matrix[word][play] += 1
    
    # Convert the dictionary into a pandas DataFrame
    term_doc_matrix_df = pd.DataFrame(term_doc_matrix).fillna(0)
    
    return term_doc_matrix_df

# Example usage:
term_doc_matrix = create_term_document_matrix('shakespeare_plays.csv')
print(term_doc_matrix)

    return None


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuple"""
         
    # Load the dataset
    df = pd.read_csv(data_file)
    
    # Initialize an empty defaultdict to store co-occurrence counts
    term_context_matrix = defaultdict(lambda: defaultdict(int))
    
    # Iterate over each row (play) in the dataframe
    for index, row in df.iterrows():
        # Extract the text content of the play and tokenize it
        text = row['PlayerLine'].lower().split()
        
        # Iterate over each word in the tokenized text
        for i, target_word in enumerate(text):
            # Determine the context window for the target word
            start_index = max(0, i - window_size)
            end_index = min(len(text), i + window_size + 1)
            
            # Iterate over words in the context window
            for j in range(start_index, end_index):
                # Skip the target word itself
                if j != i:
                    # Update co-occurrence count in the term-context matrix
                    context_word = text[j]
                    term_context_matrix[target_word][context_word] += 1
    
    # Convert the defaultdict to a pandas DataFrame
    term_context_matrix_df = pd.DataFrame(term_context_matrix).fillna(0)
    
    return term_context_matrix_df

# Example usage:
term_context_matrix = create_term_context_matrix('shakespeare_plays.csv')
print(term_context_matrix)



  # Function to create TF-IDF matrix
def create_tf_idf_matrix(term_document_matrix):
    """Given the term document matrix, output a tf-idf weighted version.

    See section 6.5 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h."""
    
    
    # Compute term frequency (TF) for each term in each document
    tf = term_document_matrix.div(term_document_matrix.sum(axis=1), axis=0)
    
    # Compute inverse document frequency (IDF) for each term
    idf = np.log(term_document_matrix.shape[0] / term_document_matrix.astype(bool).sum(axis=0))
    
    # Compute TF-IDF matrix
    tf_idf_matrix = tf.mul(idf, axis=1)
    
    return tf_idf_matrix
    
    # Example usage:
    tf_idf_matrix = create_tf_idf_matrix(term_document_matrix)
   

   # Function to create PPMI matrix
def create_ppmi_matrix(term_context_matrix):
    """Given the term context matrix, output a PPMI weighted version.

    See section 6.6 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: Numpy array where each column represents a context word
      and each row, the frequency of a word that occurs with that context word.

    Returns:
      A numpy array with the same dimension as term_context_matrix, where
      A_ij is weighted by PPMI.
    """

 
    # Compute marginal probabilities
    word_prob = term_context_matrix.sum(axis=1) / term_context_matrix.values.sum()
    context_prob = term_context_matrix.sum(axis=0) / term_context_matrix.values.sum()
    
    # Compute pointwise mutual information (PMI)
    pmi = np.log(term_context_matrix.div(np.outer(word_prob, context_prob), axis=0))
    
    # Replace negative PMI values with 0
    pmi[pmi < 0] = 0
    
    # Compute positive PMI (PPMI) by replacing values with max(0, PMI)
    ppmi_matrix = np.maximum(pmi, 0)
    
    return ppmi_matrix

# Example usage:
ppmi_matrix = create_ppmi_matrix(term_context_matrix)
  


def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    # Check for 0 vectors
    if not np.any(vector1) or not np.any(vector2):
        sim = 0

    else:
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)

    return sim


def rank_words(target_word_index, matrix):
    """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
      A length-n list of similarity scores, ordered by decreasing similarity to the
      target word indexed by word_index
    """

    # Extract the target word vector
    target_vector = vector_space[target_word_index]
    
    # Compute cosine similarity between the target word vector and all other word vectors
    similarities = np.dot(vector_space, target_vector) / (np.linalg.norm(vector_space, axis=1) * np.linalg.norm(target_vector))
    
    # Sort the words based on similarity scores in descending order
    ranked_words_indices = np.argsort(similarities)[::-1]
    
    return ranked_words_indices

# Example usage:
# Assuming 'juliet_index' is the index of the word 'juliet' in the vocabulary
# 'term_document_matrix' and 'term_context_matrix' are the vector spaces
# Change the index accordingly based on the word you want to evaluate
juliet_index = 0  # Example index for the word 'juliet'
top_similar_words = rank_words(juliet_index, term_document_matrix)
print("Top 10 similar words to 'juliet' using term-document matrix:")
print(top_similar_words[:10])

top_similar_words = rank_words(juliet_index, term_context_matrix)
print("Top 10 similar words to 'juliet' using term-context matrix:")
print(top_similar_words[:10])

    return [], []


if __name__ == "__main__":
    tuples, document_names, vocab = read_in_shakespeare()

    print("Computing term document matrix...")
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print("Computing tf-idf matrix...")
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)


    print("Computing term context matrix...")
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=4)

    print("Computing PPMI matrix...")
    ppmi_matrix = create_ppmi_matrix(tc_matrix)

    # random_idx = random.randint(0, len(document_names) - 1)

    word = "juliet"
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], td_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))


    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on tf-idf matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tf_idf_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
