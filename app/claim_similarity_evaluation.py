from time import time
import logging
import os

from app import app

import findspark
from pyspark.sql import SparkSession

from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity

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

FINDSPARK_INIT = app.config['FINDSPARK_INIT']
PATI_DATA = app.config['PATI_DATA']

findspark.init(FINDSPARK_INIT)
spark = SparkSession.builder.\
    appName('claim_similarity').\
    config("spark.driver.maxResultSize", "24g").\
    enableHiveSupport().\
    getOrCreate()

spark.sql("set spark.sql.parquet.enableVectorizedReader=false")

df_pati_data = spark.read.parquet(PATI_DATA)

# If testing, you can limit the amount of data to process
# df_pati_data = df_pati_data.sample(withReplacement=False, fraction=0.001, seed=123)

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


def process_claims(claim_txt):
    claim_corpus = []
    # corpus, with no pre-processing to retrieve the original documents.
    documents = []

    # print('Step 5: Collect PATI data')
    claims_df = spark.createDataFrame(claim_txt.toPandas()).collect()
    # print("Print first 2 results of collected data")
    # print(claims_df[0:2])

    # print('Step 6: Iterate through collected data to extract claim 1 and then apply stemming and pre processing')
    counter = 0
    for claim in claims_df:
        counter = counter + 1
        # print(">>>>> Claim text '%s'" % claim.text)
        all_claims = claim.text.splitlines()
        claim_1 = all_claims[0]
        for single_claim in all_claims:
            # print(">>>>> >>>>> Analyzing '%s'" % single_claim)
            single_claim = single_claim.lstrip()
            if single_claim.startswith("1."):
                claim_1 = single_claim
                # print(">>>>> >>>>> >>>>> Extracted claim 1 text '%s'" % claim_1)
                # print("##### ##### ##### found possible match, exit the loop")
                break

        # print("$$$$$ $$$$$ add found match to the corpus '%s'" % claim_1)
        claim_corpus.append(perform_stemming(pre_process(claim_1)))
        documents.append(claim_1)

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
if not os.path.exists('data/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model: https://code.google.com/archive/p/word2vec/")
model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Took %.2f seconds to load the Google model.' % (time() - start))

# print('Step 8: Now that you have corpus of claims and W2V Google model, build WMD Similarity model')
num_best = 10
start = time()
instance = WmdSimilarity(claim_txt_corpus, model, num_best=num_best)
print('Took %.2f seconds to build WMD Similarity Instance.' % (time() - start))

# print('Step 9: Save the WMD Similarity model')
instance.save('model/wmd_instance.model')


def find_similar_claims(claim):
    # A query is simply a "look-up" in the similarity class.
    print('Finding similar claims')
    start = time()
    sims = instance[claim]
    print('Took %.2f seconds to find similar claims.' % (time() - start))
    print(sims)

    similar_claims = []
    for i in range(num_best):
        score = sims[i][1]
        claim_txt = original_corpus[sims[i][0]]
        # print('sim = %.4f' % score)
        # print("claim_txt '%s'" % claim_txt)

        similar_claims.append({'similarity_score': str(score), 'similar_claim': claim_txt})

    return similar_claims
