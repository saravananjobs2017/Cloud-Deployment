from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

clf = joblib.load(open('nlp_model.pkl','rb'))
cv = pickle.load(open('tranform1.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method =='POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction=my_prediction)


if __name__=='__main__':
    app.run(debug=True)