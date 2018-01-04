from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired


class ClaimForm(FlaskForm):
    claim = TextAreaField('Claim', validators=[DataRequired()])
    submit = SubmitField('Submit')
