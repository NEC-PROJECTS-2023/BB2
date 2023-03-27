import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('about.html')

@app.route('/formsg')
def formsg():
    return render_template('formsg.html')

@app.route('/predict', methods= ["POST", "GET"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    result = model.predict(features)
    return render_template('submit.html', result = result)

if __name__  ==  '__main__':
    app.run(debug=True)