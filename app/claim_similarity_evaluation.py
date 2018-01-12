from time import time
import logging
import os

from gensim.similarities import WmdSimilarity

import pickle

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

# Initialize logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

# Initialize Stemmer
stemmer = LancasterStemmer()

# Download NLTK corpus:
#   Download stopwords list.
download('stopwords')
#   Download data for tokenizer.
download('punkt')


def pre_process(sentence):
    # Lowercase the text.
    sentence = sentence.lower()
    # Split into words.
    sentence = word_tokenize(sentence)
    # Remove stopwords.
    stop_words = stopwords.words('english')
    sentence = [w for w in sentence if w not in stop_words]
    # Remove numbers and punctuation
    sentence = [w for w in sentence if w.isalnum()]

    return sentence


def perform_stemming(sentence):
    # stem each word
    words = [stemmer.stem(w) for w in sentence]

    return words


def remove_duplicates(words):
    # remove duplicates
    words = sorted(list(set(words)))

    return words


if not os.path.exists('model/wmd_instance_mk.model'):
    raise ValueError("You need to download trained wmd_instance_mk.model")

start = time()
instance = WmdSimilarity.load('model/wmd_instance_mk.model')
print('Took %.2f seconds to load trained model/wmd_instance_mk.model.' % (time() - start))

with open('model/original_corpus_mk.pkl', 'rb') as f:
    original_corpus = pickle.load(f)


def find_similar_claims(claim):
    # A query is simply a "look-up" in the similarity class.
    print('Finding similar claims')
    start = time()
    sims = instance[claim]
    print('Took %.2f seconds to find similar claims.' % (time() - start))
    print(sims)

    similar_claims = []
    for i in range(10):
        score = sims[i][1]
        claim_txt = original_corpus[sims[i][0]]
        # print('sim = %.4f' % score)
        # print("claim_txt '%s'" % claim_txt)

        similar_claims.append({'similarity_score': str(score), 'similar_claim': claim_txt})

    return similar_claims


