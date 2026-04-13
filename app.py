from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Load trained model
model = joblib.load("models/aqi_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ✅ Home route
@app.route("/")
def home():
    return "AQI Backend Running ✅"


# ✅ AQI Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [
            data["pm2_5"],
            data["pm10"],
            data["no2"],
            data["so2"],
            data["co"],
            data["o3"]
        ]

        # Scale input
        scaled = scaler.transform([features])
        aqi = model.predict(scaled)[0]

        # 🎯 AQI logic
        if aqi <= 50:
            alert = "Good"
            health = "Air quality is satisfactory"
            advice = "Enjoy outdoor activities 😊"
        elif aqi <= 100:
            alert = "Moderate"
            health = "Acceptable air quality"
            advice = "Sensitive people take care 🙂"
        elif aqi <= 200:
            alert = "Unhealthy"
            health = "Breathing discomfort possible"
            advice = "Wear mask 😷"
        else:
            alert = "Very Unhealthy"
            health = "Serious health effects"
            advice = "Stay indoors 🚨"

        return jsonify({
            "aqi": round(aqi, 2),
            "alert": alert,
            "health": health,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ✅ Chatbot Route
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message", "").lower()

        # 🎯 Simple chatbot logic
        if "aqi" in user_msg:
            reply = "AQI (Air Quality Index) tells how clean or polluted the air is."
        
        elif "safe" in user_msg:
            reply = "AQI below 50 is considered safe for everyone."
        
        elif "pollution" in user_msg:
            reply = "Air pollution is caused by vehicles, industries, and burning fuels."
        
        elif "mask" in user_msg:
            reply = "Wearing a mask helps reduce inhalation of harmful particles."
        
        elif "hello" in user_msg or "hi" in user_msg:
            reply = "Hello! Ask me anything about air quality 😊"
        
        else:
            reply = "I can help you with AQI, pollution, and health advice!"

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": "Error: " + str(e)})


# ✅ Run server
if __name__ == "__main__":
    app.run(debug=True, port=5002)