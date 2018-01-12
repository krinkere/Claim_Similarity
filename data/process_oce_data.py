import pandas as pd
import zipfile
import findspark
from pyspark.sql import SparkSession
import os
import csv
from time import time
from pyspark.sql.types import StringType, StructType, StructField, BooleanType, IntegerType


# FINDSPARK_INIT = '/data/krinker/spark/spark-2.2.1-bin-hadoop2.7/'
FINDSPARK_INIT = '/usr/hdp/current/spark2-client/'


zf = zipfile.ZipFile('patent_claims_fulltext.csv.zip')
csv_files = zf.infolist()

findspark.init(FINDSPARK_INIT)
spark = SparkSession.builder.\
    appName('claim_similarity').\
    config("spark.driver.maxResultSize", "64g").\
    enableHiveSupport().\
    getOrCreate()

# patent numbers for AU 1747
# au_1747_claims = "patlistAU_1747.csv"
# au1747_pattents = []
# with open(au_1747_claims, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # skip header row
#     for row in reader:
#         pat_no = int(row[0])
#         print("Processing %s" % pat_no)
#         au1747_pattents.append(pat_no)
#
# print("collected %r patents" % len(au1747_pattents))
independent_claims = []

with zipfile.ZipFile('patent_claims_fulltext.csv.zip') as z:
    for filename in z.namelist():
        print(filename)
        if not os.path.isdir(filename):
            # read the file
            with z.open(filename) as f:
                f.readline()  # skip the first line
                for line in f:
                    # print(line)
                    # print("***")
                    pat_data = line.split(b',')
                    pat_no = pat_data[0]
                    claim_no = int(pat_data[1])
                    claim_txt = pat_data[2]
                    if claim_no == 1:
                        print(claim_txt)
                        independent_claims.append(claim_txt)
                    else:
                        continue


print("Number of claims %r" % len(independent_claims))
import pickle
with open('independent_claims_from_oce.pkl', 'wb') as f:
    pickle.dump(independent_claims, f)

# schemaString = "pat_no claim_no claim_txt dependencies ind_flg appl_id"
# fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
# schema = StructType(fields)

# or
#schema = StructType([
#    StructField("pat_no", IntegerType()),
#    StructField("claim_no", IntegerType()),
#    StructField("claim_txt", StringType()),
#    StructField("dependencies", StringType()),
#    StructField("ind_flg", IntegerType()),
#    StructField("appl_id", StringType())    
#])


# for csv_file in csv_files:
#     print(csv_file.filename)
#     process_file = zf.open(csv_file.filename)
#     print("opened file")
#     start = time()
#     df = spark.read.csv(process_file, header=True, mode="DROPMALFORMED",schema=schema)
#     print('Took %.2f seconds to read data.' % (time() - start))
#     # df = pd.read_csv(zf.open(csv_file.filename))
#     df.show()
#
# # https://stackoverflow.com/questions/40003021/how-to-elegantly-create-a-pyspark-dataframe-from-a-csv-file-and-convert-it-to-a
# for csv_file in csv_files:
#     from pyspark.sql.types import *
#     PATI_DATA_FILE = sc.textFile(csv_file.filename)
#     # Extract the header line
#     header = PATI_DATA_FILE.first()
#
#     # Assuming that all the columns are string, let's create a new StructField for each column
#     fields = [StructField(field_name, StringType(), True) for field_name in header]
#
#     schema = StructType(fields)
#     # We have the remove the header from the textfile rdd
#
#     # Extracting the header (first line) from the RDD
#     dataHeader = PATI_DATA_FILE.filter(lambda x: "pat_no" in x)
#
#     # Extract the data without headers. We can make use of the `subtract` function
#     dataNoHeader = PATI_DATA_FILE.subtract(dataHeader)
#
#     pati_temp_rdd = dataNoHeader.mapPartitions(lambda x: csv.reader(x, delimiter=","))
#
#     pati_df = sqlContext.createDataFrame(pati_temp_rdd, schema)
#     pati_df.registerTempTable("pati_oec")
#
#     from pyspark.sql import SQLContext
#     sqlContext  =  SQLContext(sc)
#     sqlContext.sql('SELECT Id from numeric')
#
#     pandas_df = pati_df.limit(5).toPandas()
#
#     pati_df.select("pat_no")
#
