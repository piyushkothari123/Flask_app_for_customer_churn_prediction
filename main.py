# import import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
with open('my_model.pkl', 'rb') as mo:
    model = pickle.load(mo)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])


def predict():
    classes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges','MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL','InternetService_Fiber optic', 'InternetService_No','OnlineSecurity_No', 'OnlineSecurity_No internet service','OnlineSecurity_Yes', 'OnlineBackup_No','OnlineBackup_No internet service', 'OnlineBackup_Yes','DeviceProtection_No', 'DeviceProtection_No internet service','DeviceProtection_Yes', 'TechSupport_No','TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No','StreamingTV_No internet service', 'StreamingTV_Yes','StreamingMovies_No', 'StreamingMovies_No internet service','StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year','Contract_Two year', 'PaymentMethod_Bank transfer','PaymentMethod_Credit card', 'PaymentMethod_Electronic check','PaymentMethod_Mailed check']
    to_ints = [float(request.form.get(class_, 0)) if request.form.get(class_, 0) != '' else 0.0 for class_ in classes]
    predictions = model.predict([to_ints])
    if (predictions[0] == 0):
        prediction_text="Customer is not churning"
    else:
        prediction_text="Customer is churning"
        print(prediction_text)
    return render_template('index.html',prediction_text=f'After applying all the observation the customer seems to be {prediction_text}')



if __name__=='__main__':
     app.run(debug=True)