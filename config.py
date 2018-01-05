import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    FINDSPARK_INIT = os.environ.get('FINDSPARK_INIT') or '/usr/hdp/current/spark2-client/'
    PATI_DATA = os.environ.get('PATI_DATA') or 'hdfs://bdr-itwv-mongo-3.dev.uspto.gov:54310/tmp/PATI_data/data'

