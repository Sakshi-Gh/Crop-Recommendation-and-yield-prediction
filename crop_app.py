from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
app = Flask(__name__)

# Load the decision tree model for crop recommendation
with open("model.pkl", "rb") as file:
    dt_tuned_model = pickle.load(file)

# Load lstm model for crop yeild
from tensorflow.keras.models import load_model
loaded_lstm_model = load_model("lstm_model.h5")

label_encoder_crop = LabelEncoder()
label_encoder_state = LabelEncoder()
scaler_area = StandardScaler()
scaler_rainfall = StandardScaler()
scaler_fertilizer = StandardScaler()

yield_data = pd.read_csv("data/crop_yield.csv")
label_encoder_crop.fit(yield_data['Crop'])
label_encoder_state.fit(yield_data['State'])
scaler_area.fit(yield_data[['Area']])
scaler_rainfall.fit(yield_data[['Annual_Rainfall']])
scaler_fertilizer.fit(yield_data[['Fertilizer']])

@app.route("/")
def home():
    return render_template("crop_recommendation.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Retrieve form data
    try:
        N = float(request.form.get("nitrogen"))
        P = float(request.form.get("phosphorus"))
        K = float(request.form.get("potassium"))
        pH = float(request.form.get("ph_level"))
        rainfall = float(request.form.get("rainfall"))
        temperature = float(request.form.get("temperature"))

        # Prepare the input as a DataFrame
        input_data = pd.DataFrame([[N, P, K, pH, rainfall, temperature]],
                                  columns=["N", "P", "K", "pH", "rainfall", "temperature"])

        # Make prediction
        predicted_crop = dt_tuned_model.predict(input_data)[0]
        recommendation_results = f"Recommended crop: {predicted_crop}"
    except Exception as e:
        recommendation_results = f"Error in prediction: {e}"

    return render_template("crop_recommendation.html", recommendation_results=recommendation_results)

@app.route("/yield", methods=["POST"])
def yield_prediction():
    # Collect input data from the form
    crop = request.form["crop"]
    state = request.form["state"]
    area = float(request.form["area"])
    rainfall = float(request.form["annual_rainfall"])
    fertilizer = float(request.form["fertilizer"])
    
    # Encode and scale inputs
    crop_encoded = label_encoder_crop.transform([crop])[0]
    state_encoded = label_encoder_state.transform([state])[0]
    area_scaled = scaler_area.transform([[area]])[0][0]
    rainfall_scaled = scaler_rainfall.transform([[rainfall]])[0][0]
    fertilizer_scaled = scaler_fertilizer.transform([[fertilizer]])[0][0]
    
    # Create input array
    input_features = np.array([[crop_encoded, state_encoded, area_scaled, rainfall_scaled, fertilizer_scaled]])
    input_features_reshaped = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))
    
    # Predict yield
    yield_prediction = loaded_lstm_model.predict(input_features_reshaped)[0][0]
    yield_results=f"Predicted Yield: {yield_prediction*1000:.2f} kg/ha."

    # Return result to the form
    return render_template("crop_yield_prediction.html", yield_results=yield_results)

if __name__ == "__main__":
    app.run(debug=True)
