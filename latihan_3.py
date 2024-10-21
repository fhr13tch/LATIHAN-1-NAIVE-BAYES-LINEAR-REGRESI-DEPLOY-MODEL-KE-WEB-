from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
with open("model/hasil_pelatihan_model.pkl", "rb") as model_file:
    ml_model = joblib.load(model_file)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])

            
            pred_args = np.array([RnD_Spend, Admin_Spend, Market_Spend]).reshape(1, -1)
            model_prediction = ml_model.predict(pred_args)
            model_prediction = round(float(model_prediction), 2)

            return render_template('predict.html', prediction=model_prediction)
        except ValueError:
            return "Please ensure all values are entered correctly."
    
    return render_template('predict.html', prediction=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0')