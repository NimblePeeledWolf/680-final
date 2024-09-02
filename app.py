from flask import Flask, request, render_template
import pickle
import numpy as np
from ucimlrepo import fetch_ucirepo 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

maternal_health_risk = fetch_ucirepo(id=863)
x = maternal_health_risk.data.features 
y = maternal_health_risk.data.targets 
y = y.values.ravel()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y = y_encoded
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Load the saved model and label encoder
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        features = [
            float(request.form['age']),
            float(request.form['systolicbp']),
            float(request.form['diastolicbp']),
            float(request.form['bs']),
            float(request.form['bodytemp']),
            float(request.form['heartrate'])
        ]

        # Make a prediction
        prediction_encoded = model.predict([features])
        prediction_categorical = label_encoder.inverse_transform(prediction_encoded)

        return render_template('index.html', prediction=prediction_categorical[0])

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
