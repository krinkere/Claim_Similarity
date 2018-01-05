from time import time
import os

from app import app
import findspark
import pyspark
from pyspark.sql import SparkSession, HiveContext

from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity

import pandas as pd
# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# Initialize logging.
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

# Download NLTK corpus:
#   Download stopwords list.
download('stopwords')
#   Download data for tokenizer.
download('punkt')

FINDSPARK_INIT = app.config['FINDSPARK_INIT']
PATI_DATA = app.config['PATI_DATA']

findspark.init(FINDSPARK_INIT)
# conf = pyspark.SparkConf().setAppName('claim_similarity')
# sc = pyspark.SparkContext(conf=conf).getOrCreate()
spark = SparkSession.builder.appName('claim_similarity').config("spark.driver.maxResultSize", "4g").enableHiveSupport().getOrCreate()
df_pati_data = spark.read.parquet('hdfs://bdr-itwv-mongo-3.dev.uspto.gov:54310/tmp/PATI_data/data')
df_pati_data = df_pati_data.sample(withReplacement=False, fraction=0.001, seed=123)

print('# of rows: ', df_pati_data.count())
print('PATI Data')
print(df_pati_data.show())

df_pati_clm_data = df_pati_data[df_pati_data['type'] == "CLM"]
print('Claim Only Data')
print(df_pati_clm_data.show())

df_pati_clm_txt = df_pati_clm_data.drop('appId', 'exception', 'ifwNumber', 'mailRoomDate', 'type')
print('Claim Text')
print(df_pati_clm_txt.show())


def process_huge_claims(claim_txt):
    claim_corpus = []
    # corpus, with no pre-processing to retrieve the original documents.
    documents = []
    # convert claim text rdd into list of claims
    claims = claim_txt.toPandas().values
    print('*********************extracted claim text*********************************')
    for claim in claims:
        claim_corpus.append(pre_process(claim))
        documents.append(claim)

    return claim_corpus, documents

""" Load PATI specific data in terms of claims and compare them """
print('Attempting to load PATI sub data...')
start = time()
# df = pd.read_csv("data/pati_data.csv")
# claim_txts = df['CLAIM_TXTS']
# claim_txt_corpus, original_corpus = process_claims(claim_txts)
claim_txt_corpus, original_corpus = process_huge_claims(df_pati_clm_txt)
print('Took %.2f seconds to load PATI data.' % (time() - start))

#tableList = [x["text"] for x in df_pati_clm_data.rdd.collect()]
#print(tableList)

# spark.stop()

print('Attempting to load Google model...')
start = time()
if not os.path.exists('data/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model: https://code.google.com/archive/p/word2vec/")
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Took %.2f seconds to load the Google model.' % (time() - start))


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


# def process_claims(claim_txt):
#     claim_corpus = []
#     # corpus, with no pre-processing to retrieve the original documents.
#     documents = []
#     for index_val, txt_val in claim_txt.iteritems():
#         # print(txt_val)
#         splitup_txt = [s.strip() for s in txt_val.split("***")]
#         for txt in splitup_txt:
#             claim_corpus.append(pre_process(txt))
#             documents.append(txt)
#
#     return claim_corpus, documents


# Initialize WmdSimilarity.
num_best = 10
start = time()
instance = WmdSimilarity(claim_txt_corpus, model, num_best=num_best)
print('Took %.2f seconds to initialize WMD Similarity Instance.' % (time() - start))


def find_similar_claims(claim):
    # A query is simply a "look-up" in the similarity class.
    print('trying to find similar claims')
    sims = instance[claim]
    print('found similar claims')

    similar_claims = []
    for i in range(num_best):
        score = sims[i][1]
        claim_txt = original_corpus[sims[i][0]]
        print('sim = %.4f' % score)
        print(claim_txt)
        print("")

        similar_claims.append({'similarity_score': str(score), 'similar_claim': claim_txt})

    return similar_claims
