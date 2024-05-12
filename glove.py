import compress_pickle as pkl
import numpy as np

def create_word_embeddings():
    word_embeddings = {}
    with open('glove.6B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings

obj = create_word_embeddings()
with open("glove.6B.50d.gz", "wb") as f:
    pkl.dump(obj, f)