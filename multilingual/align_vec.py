# Directly adapted from https://github.com/Babylonpartners/fastText_multilingual/blob/master/align_your_own.ipynb
#

import numpy as np
import sys

from fasttext import FastVector

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)


if __name__ == '__main__':
    lang_pairs = sys.argv[1].split(",")
    for pair in lang_pairs:
        lang1, lang2 = pair.split("-")
        lang1_dictionary = FastVector(vector_file='embed_bkp/%s.embed.bin' % lang1)
        lang2_dictionary = FastVector(vector_file='embed_bkp/%s.embed.bin' % lang2)

        lang1_words = set(lang2_dictionary.word2id.keys())
        lang2_words = set(lang1_dictionary.word2id.keys())
        overlap = list(lang1_words & lang2_words)
        bilingual_dictionary = [(entry, entry) for entry in overlap]

        # form the training matrices
        source_matrix, target_matrix = make_training_matrices(
            lang1_dictionary, lang2_dictionary, bilingual_dictionary)

        # learn and apply the transformation
        transform = learn_transformation(source_matrix, target_matrix)
        lang1_dictionary.apply_transform(transform)
        lang1_dictionary.export('embed_aligned/%s.embed.bin' % lang1)

