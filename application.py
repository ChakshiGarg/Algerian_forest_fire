import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template,jsonify
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('ridge.pkl','rb'))
scaler_model = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('data.html')
@app.route('/data',methods = ["GET",'POST'])
def data():
    if request.method == 'POST':
       Temperature = float(request.form.get('Temperature'))
       RH = float(request.form.get('RH'))
       Ws = float(request.form.get('Ws'))
       Rain = float(request.form.get('Rain'))
       FFMC = float(request.form.get('FFMC'))
       DMC = float(request.form.get('DMC'))
       ISI = float(request.form.get('ISI'))
       Classes = float(request.form.get('Classes'))
       Region = float(request.form.get('Region'))

       new_data_scaled = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
       result = ridge_model.predict(new_data_scaled)

       return render_template('predict.html', results = result[0])
      
    else:
        return render_template('predict.html')


if __name__  == "__main__":
    app.run(host = "0.0.0.0")