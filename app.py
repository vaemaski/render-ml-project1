from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
# Load the trained model and scaler
try:
    with open("modelpyt.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    print("Model and scaler loaded successfully!")
except EOFError:
    print("Error: The pickle file is empty or corrupted.")
except FileNotFoundError:
    print("Error: File not found. Check the file path.")


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

    
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = np.asarray(int_features).reshape(1, -1)

    # Standardize the input data using the pre-fitted scaler
    std_data = scaler.transform(final_features)

    # Use the loaded model to predict
    prediction = model.predict(std_data)

    # Prepare the output
    if prediction[0] == 0:
        output = 'Not diabetic'
    else:
        output = 'Diabetic'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
