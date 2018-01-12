from time import time
import logging
import os

from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

import pickle

# Initialize logging.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

# Download NLTK corpus:
#   Download stopwords list.
download('stopwords')
#   Download data for tokenizer.
download('punkt')

# Initialize Stemmer
stemmer = LancasterStemmer()

# with open('../data/independent_claims_from_oce.pkl', 'rb') as f:
with open('../data/independent_claims.pkl', 'rb') as f:
    original_corpus = pickle.load(f)


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


def process_claims(original_corpus):
    claim_corpus = []

    start = time()
    counter = 0
    print('Took %.2f seconds to iterate through collected PATI Data.' % (time() - start))
    for sentence in original_corpus:
        counter = counter + 1
        # sentence = sentence.decode('utf-8')
        print(sentence)
        sentence = pre_process(sentence)
        print(sentence)
        sentence = perform_stemming(sentence)
        print(sentence)
        claim_corpus.append(sentence)

    print("Total processed documents: %r " % counter)

    return claim_corpus, original_corpus


# print('Step 4: Process claim text (lowercase, stop words, stemming, etc)')
start = time()
print('Processing Data...')
claim_txt_corpus, original_corpus = process_claims(original_corpus)
print('Took %.2f seconds to load Data.' % (time() - start))

# print('Step 7: Load google vector model to get W2V similarity between words')
print('Loading Google model...')
start = time()
if not os.path.exists('../data/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("You need to download the google news model: https://code.google.com/archive/p/word2vec/")
model = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Took %.2f seconds to load the Google model.' % (time() - start))

# print('Step 8: Now that you have corpus of claims and W2V Google model, build WMD Similarity model')
num_best = 10
start = time()
instance = WmdSimilarity(claim_txt_corpus, model, num_best=num_best)
print('Took %.2f seconds to build WMD Similarity Instance.' % (time() - start))

# print('Step 9: Save the WMD Similarity model and claim list')
instance.save('wmd_instance_mk.model')

with open('original_corpus_mk.pkl', 'wb') as f:
    pickle.dump(original_corpus, f)
