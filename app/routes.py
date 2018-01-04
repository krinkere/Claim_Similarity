from flask import render_template, flash, redirect, url_for, session
from app import app
from app.forms import ClaimForm
from app.claim_similarity_evaluation import pre_process, perform_stemming, remove_duplicates, find_similar_claims


@app.route('/')
@app.route('/index')
def index():
    # session['claim_results'] = []
    session['original_claim'] = ""
    session['pre_processed_claim'] = ""
    session['stemmed_claim'] = ""
    session['no_duplicates_claim'] = ""
    return render_template('index.html', title='Welcome to USPTO Claim Similarity Finder')


@app.route('/claim', methods=['GET', 'POST'])
def claim():
    form = ClaimForm()
    if form.validate_on_submit():
        original_claim = form.claim.data
        if original_claim is None:
            flash('Provide claim data')
            return redirect(url_for('claim'))

        # claim_1_orig = " A light diffuser comprising a transparent substrate and a plurality of lenses forming a complex lens structure on a first surface of the  substrate  wherein the plurality of lenses comprise a non-homogenous material."
        # claim_2_orig = "The light diffuser according to claim 1  wherein the non- homogenous material includes a majority component and a minority component."
        # claim_3_orig = "It contains a mix of theory and examples, but the focus is on the code, since we believe that the best way to learn something in this field is through looking at examples."

        pre_processed_claim = pre_process(original_claim)
        stemmed_claim = perform_stemming(pre_processed_claim)
        no_duplicates_claim = remove_duplicates(stemmed_claim)
        similar_claims = find_similar_claims(no_duplicates_claim)

        # claim_results = [
        #     {
        #         'similarity_score': 'pre_processed_claim',
        #         'similar_claim': pre_processed_claim
        #     },
        #     {
        #         'similarity_score': 'claim_data',
        #         'similar_claim': original_claim
        #     }
        # ]

        session['original_claim'] = original_claim
        session['pre_processed_claim'] = pre_processed_claim
        session['stemmed_claim'] = stemmed_claim
        session['no_duplicates_claim'] = no_duplicates_claim
        session['similar_claims'] = similar_claims

        # session['claim_results'] = claim_results
        return redirect(url_for('result'))

    # claim_results = session.get('claim_results')
    # original_claim = session.get('original_claim')
    # pre_processed_claim = session.get('pre_processed_claim')
    # stemmed_claim = session.get('stemmed_claim')

    return render_template('claim.html',  title='Claim', form=form)


@app.route('/result')
def result():
    claim_results = session.get('claim_results')
    original_claim = session.get('original_claim')
    pre_processed_claim = session.get('pre_processed_claim')
    stemmed_claim = session.get('stemmed_claim')
    no_duplicates_claim = session['no_duplicates_claim']
    similar_claims = session['similar_claims']
    return render_template('result.html',
                           claim_results=claim_results,
                           original_claim=original_claim,
                           pre_processed_claim=pre_processed_claim,
                           stemmed_claim=stemmed_claim,
                           no_duplicates_claim=no_duplicates_claim,
                           similar_claims=similar_claims
                           )
