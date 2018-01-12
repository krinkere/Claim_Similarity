from flask import render_template, flash, redirect, url_for, session
from app import app
from app.forms import ClaimForm
from app.claim_similarity_evaluation import pre_process, perform_stemming, remove_duplicates, find_similar_claims
from flask.json import jsonify


@app.route('/')
@app.route('/index')
def index():
    session['original_claim'] = ""
    session['pre_processed_claim'] = ""
    session['stemmed_claim'] = ""
    session['no_duplicates_claim'] = ""

    return render_template('index.html', title='Welcome to USPTO Claim Similarity Finder')


@app.route('/api/v1.0/sim_claims/<string:claim_text>', methods=['GET'])
def api(claim_text):
    pre_processed_claim = pre_process(claim_text)
    stemmed_claim = perform_stemming(pre_processed_claim)
    no_duplicates_claim = remove_duplicates(stemmed_claim)
    similar_claims = find_similar_claims(no_duplicates_claim)

    return jsonify({'sim_claims': similar_claims})


@app.route('/claim', methods=['GET', 'POST'])
def claim():
    form = ClaimForm()
    if form.validate_on_submit():
        original_claim = form.claim.data
        if original_claim is None:
            flash('Provide claim data')
            return redirect(url_for('claim'))

        pre_processed_claim = pre_process(original_claim)
        stemmed_claim = perform_stemming(pre_processed_claim)
        no_duplicates_claim = remove_duplicates(stemmed_claim)
        similar_claims = find_similar_claims(no_duplicates_claim)

        session['original_claim'] = original_claim
        session['pre_processed_claim'] = pre_processed_claim
        session['stemmed_claim'] = stemmed_claim
        session['no_duplicates_claim'] = no_duplicates_claim
        session['similar_claims'] = similar_claims

        return render_template('result.html',
                           original_claim=original_claim,
                           pre_processed_claim=pre_processed_claim,
                           stemmed_claim=stemmed_claim,
                           no_duplicates_claim=no_duplicates_claim,
                           similar_claims=similar_claims
                           )
        # return redirect(url_for('result'))

    return render_template('claim.html',  title='Claim', form=form)


@app.route('/result')
def result():
    claim_results = session.get('claim_results')
    print(claim_results)
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
