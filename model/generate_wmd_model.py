from time import time
import logging
import os

import findspark
from pyspark.sql import SparkSession

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

# mongo3
FINDSPARK_INIT = '/usr/hdp/current/spark2-client/'
# model1
# FINDSPARK_INIT = '/data/krinker/spark/spark-2.2.1-bin-hadoop2.7/'
PATI_DATA = 'hdfs://bdr-itwv-mongo-3.dev.uspto.gov:54310/tmp/PATI_data/data'

findspark.init(FINDSPARK_INIT)
spark = SparkSession.builder.\
    appName('claim_similarity').\
    config("spark.driver.maxResultSize", "64g").\
    enableHiveSupport().\
    getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

spark.sql("set spark.sql.parquet.enableVectorizedReader=false")

df_pati_data = spark.read.parquet(PATI_DATA)

# If testing, you can limit the amount of data to process
# df_pati_data = df_pati_data.sample(withReplacement=False, fraction=0.0001, seed=123)

print('Data size: ', df_pati_data.count())

# print("Step 1: Read PATI Data off %s" % PATI_DATA)
# print('Sample PATI Data')
# print(df_pati_data.show())

df_pati_clm_data = df_pati_data[df_pati_data['type'] == "CLM"]
# print('Step 2: Extract Claim Only Data')
# print(df_pati_clm_data.show())

df_pati_clm_txt = df_pati_clm_data.drop('appId', 'exception', 'ifwNumber', 'mailRoomDate', 'type')
# print('Step 3: Extract Claim Text')
# print(df_pati_clm_txt.show())


def extract_claim1(text):
    all_claims = text.splitlines()
    if len(all_claims) == 0:
        return "No claims were found"

    claim_1 = all_claims[0]
    for single_claim in all_claims:
        single_claim = single_claim.lstrip()
        if single_claim.startswith("1."):
            claim_1 = single_claim
            break

    return claim_1


def process_claims(claim_txt):
    claim_corpus = []
    # corpus, with no pre-processing to retrieve the original documents.
    documents = []

    # print("show old")
    # claim_txt.show(5)

    # print('Step 5: Iterate through claims to extract claim 1 and then apply stemming and pre processing')
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType

    # Extract claim 1
    extract_claim1_udf = udf(extract_claim1, StringType())
    claim_txt = claim_txt.withColumn('claim1', extract_claim1_udf('text'))

    # Pre-process claim 1 - lowecase, tokenize and remove punctuations and stop words
    stopwords_list = stopwords.words('english')

    def remove_stopwords(label, feature_list):
        # Lowercase the text.
        sentence = label.lower()
        # Split into words.
        sentence = word_tokenize(sentence)
        # Remove stopwords.
        sentence = [w for w in sentence if w not in feature_list]
        # Remove numbers and punctuation
        sentence = [w for w in sentence if w.isalnum()]

        return sentence

    def pre_process_udf(stopwords_list):
        return udf(lambda l: remove_stopwords(l, stopwords_list))

    claim_txt = claim_txt.withColumn('claim1_processed', pre_process_udf(stopwords_list)(col("claim1")))

    # print("show pre processed")
    # claim_txt.show(5)

    # Stem re-processed claim 1
    # Initialize Stemmer
    stemmer = LancasterStemmer()

    def perform_stemming(sentence, stemmer):
        # stem each word
        words = [stemmer.stem(w) for w in sentence]

        return words

    def stemmer_udf(stemmer):
        return udf(lambda l: perform_stemming(l, stemmer))

    claim_txt = claim_txt.withColumn('claim1_processed_stemmed', stemmer_udf(stemmer)(col("claim1_processed")))

    # print("show stemmed")
    # claim_txt.show(5)

    # print('Step 6: Collect PATI data')
    start = time()
    claims_df = claim_txt.rdd.flatMap(lambda x: [(x.claim1, x.claim1_processed_stemmed)]).collect()
    print('Took %.2f seconds to collect PATI Data.' % (time() - start))

    start = time()
    counter = 0
    for claim in claims_df:
        counter = counter + 1
        claim_corpus.append(claim[1])
        documents.append(claim[0])

    print('Took %.2f seconds to iterate through collected PATI Data.' % (time() - start))
    print("Total processed documents: %r " % counter)

    return claim_corpus, documents


# print('Step 4: Process claim text (lowercase, stop words, stemming, etc)')
start = time()
print('Processing PATI Data...')
claim_txt_corpus, original_corpus = process_claims(df_pati_clm_txt)
print('Took %.2f seconds to load PATI Data.' % (time() - start))

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
instance.save('wmd_instance.model')

with open('original_corpus.pkl', 'wb') as f:
    pickle.dump(original_corpus, f)
