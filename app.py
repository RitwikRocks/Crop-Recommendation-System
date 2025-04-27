from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
models = {
    'Random Forest': pickle.load(open('models/random_forest.pkl', 'rb')),
    'KNN':          pickle.load(open('models/knn.pkl',          'rb')),
    'SVM Linear':  pickle.load(open('models/svm_linear.pkl',    'rb')),
    'SVM NonLinear': pickle.load(open('models/svm_nonlinear.pkl','rb')),
    'Naive Bayes': pickle.load(open('models/naive_bayes.pkl',   'rb')),
}

scaler          = pickle.load(open('models/scaler.pkl',          'rb'))
season_encoder  = pickle.load(open('models/encoder.pkl',         'rb'))
label_encoder   = pickle.load(open('models/label_encoder.pkl',  'rb'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['GET'])
def home():
    # Just render the form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1) Collect inputs
    temperature        = float(request.form['temperature'])
    humidity           = float(request.form['humidity'])
    ph                 = float(request.form['ph'])
    water_availability = float(request.form['water_availability'])
    season             = request.form['season']
    model_choice       = request.form['model_choice']

    # 2) Encode & scale
    season_transformed = season_encoder.transform([season])[0]
    X_raw = [temperature, humidity, ph, water_availability]
    X_scaled = scaler.transform([X_raw])[0]
    X_final = np.append(X_scaled, season_transformed).reshape(1, -1)

    # 3) Predict
    model = models.get(model_choice, models['Random Forest'])
    pred_encoded = model.predict(X_final)
    prediction = label_encoder.inverse_transform(pred_encoded)[0]

    # 4) Render the result page
    return render_template('result.html', prediction=prediction)

     # 4) Render the About this Project Page

if __name__ == '__main__':
    app.run(debug=True)
