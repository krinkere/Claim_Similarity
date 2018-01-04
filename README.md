# create virtual environment
conda create -n claim_similarity python=3.6

# activate it
activate claim_similarity

# install following packages
conda install gensim
conda install pandas
conda install nltk
conda install flask
conda install flask-wtf
conda install boto3
conda install pyemd

# set variable to specify flask app that needs to be run
export FLASK_APP=claim_similarity.py
or on Windows
set FLASK_APP=claim_similarity.py

# run the application
flask run 

# Datasets are available from
https://www.uspto.gov/learning-and-resources/electronic-data-products/patent-claims-research-dataset
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
or 
https://code.google.com/archive/p/word2vec/