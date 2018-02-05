import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    #FINDSPARK_INIT = os.environ.get('FINDSPARK_INIT') or '/usr/hdp/current/spark2-client/'
    FINDSPARK_INIT = os.environ.get('FINDSPARK_INIT') or 'C:/spark/spark-2.2.0-bin-hadoop2.7/'
    PATI_DATA = os.environ.get('PATI_DATA') or 'hdfs://bdr-itwv-mongo-3.dev.uspto.gov:54310/tmp/PATI_data/data'

