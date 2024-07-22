import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models and scalers
@st.cache_resource(show_spinner="Loading model")
def load_model():
    model = joblib.load("/Users/sot/final_model.pk1")
    return model

@st.cache_resource(show_spinner="Loading scaler")
def load_scaler():
    scaler = joblib.load("/Users/sot/final_scaler.pk1")
    return scaler

@st.cache_resource(show_spinner="Loading column names")
def load_col():
    col_name = joblib.load("/Users/sot/col_names.pk1")
    return col_name

# Prediction function
@st.cache_data(show_spinner="Predicting...")
def make_prediction(_model, _scaler, _col_name, X_pred):
    cat_df = pd.DataFrame([[education, employmenttype, maritalstatus, hasmortgage, hasdependent, loanpurpose, hascosigner]],
                          columns=['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'])
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[age, income, loan_amount, credit_score, monthemployed, numofcl, interest, loanterm, dtiratio]],
                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                   'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'])
    combined_df = pd.concat([num_df, cat_encoded], axis=1)
    combined_df = combined_df.reindex(columns=_col_name, fill_value=0)

    loan = combined_df.values
    num_scaled = _scaler.transform(loan[:, :24])
    loan = np.hstack([num_scaled, loan[:, 24:]])
    
    prediction_proba = _model.predict_proba(loan)[0]
    prediction = _model.predict(loan)[0]
    prob_percentage = prediction_proba[prediction] * 100

    if prediction == 1:
        return f'(1) -> This client will not be able to pay back the loan ({prob_percentage:.2f}% probability)'
    else:
        return f'(0) -> This client would pay back the loan ({prob_percentage:.2f}% probability)'

# Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Loan Eligibility Checker", layout="centered", initial_sidebar_state="collapsed")
    st.title('Loan Eligibility Checker')

    st.write("## Fill out the information below to check your loan eligibility")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        income = st.number_input('Income', min_value=0, max_value=1000000000, value=0, step=1)
        age = st.slider('Age', min_value=18, max_value=100, value=18, step=1)
        hasmortgage = st.checkbox('Has Mortgage')
        interest = st.slider('Interest Rate (%)', min_value=1, max_value=100, value=2, step=1)
        education = st.selectbox('Education', ["B - Bachelor's", 'P - PhD', "M - Master's", 'H - High School'])
        employmenttype = st.selectbox('Employment Type', ['F - Full-time', 'U - Unemployed', 'S - Self-employed', 'P - Part-time'])

    with col2:
        loan_amount = st.number_input('Loan Amount', min_value=0, max_value=1000000, value=0, step=1)
        credit_score = st.slider('Credit Score', min_value=100, max_value=1000, value=100, step=1)
        hascosigner = st.checkbox('Has Cosigner')
        loanterm = st.slider('Loan Term', min_value=1, max_value=100, value=1, step=1)
        maritalstatus = st.selectbox('Marital Status', ['M - Married', 'D - Divorced', 'S - Single'])
        
    with col3:
        monthemployed = st.number_input('Months Employed', min_value=0, max_value=200, value=1, step=1)
        dtiratio = st.slider('DTI Ratio', min_value=0.000, max_value=30.000, value=0.100, step=0.001)
        hasdependent = st.checkbox('Has Dependent')
        numofcl = st.slider('Number of Credit Lines', min_value=0, max_value=25, value=0, step=1)
        loanpurpose = st.selectbox('Loan Purpose', ['A - Auto', 'B - Business', 'E - Education', 'H - Home', 'O - Other'])

    st.divider()
    pred_btn = st.button('Check Eligibility', type='primary')
    if pred_btn:
        model = load_model()
        scaler = load_scaler()
        col_name = load_col()

        X_pred = [
            education, employmenttype, maritalstatus, 
            'yes' if hasmortgage else 'no', 'yes' if hasdependent else 'no', 
            loanpurpose, 'yes' if hascosigner else 'no',
            age, income, loan_amount, credit_score, monthemployed, numofcl, interest, loanterm, dtiratio
        ]
        pred = make_prediction(model, scaler, col_name, X_pred)
        st.write(pred)
