import pandas as pd
from sklearn.externals import joblib
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = joblib.load('./models/credit_model.pkl')
features_names = [
            'bill_amt_1',
            'bill_amt_2',
            'bill_amt_3',
            'bill_amt_4',
            'bill_amt_5',
            'bill_amt_6',
            'limit_balance',
            'degree',
            'pay_0',
            'pay_2',
            'pay_3',
            'pay_4',
            'pay_5',
            'pay_6',
            'pay_amt_1',
            'pay_amt_2',
            'pay_amt_3',
            'pay_amt_4',
            'pay_amt_5',
            'pay_amt_6',
            'sex',
            'marital_status',
            'age'
        ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    features = pd.DataFrame(data=[features[:23]], columns=features_names)
    features[features.columns[0:7]] = features[features.columns[0:7]].astype(int)
    features[features.columns[8:]] = features[features.columns[8:]].astype(int)
    prediction = model.predict_proba(features)[0][1]

    return render_template('index.html', prediction_text="Probability of next month's default on payment: {} %".format(round(prediction*100, 2)))

if __name__ == "__main__":
    app.run()
