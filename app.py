from flask_cors import CORS
CORS(app)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import joblib
import os

from anomaly import log_anomaly
from alerts import alert_system
from health import health_advice
from email_alert import send_email_alert
from db import engine

#  AI CHATBOT
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ==============================
#  API KEYS
# ==============================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
#  LOAD MODELS
# ==============================
aqi_model = joblib.load("models/aqi_model.pkl")
anomaly_model = joblib.load("models/anomaly.pkl")
health_model = joblib.load("models/health.pkl")
scaler = joblib.load("models/scaler.pkl")

# ==============================
#  WEATHER FUNCTION
# ==============================
def get_weather(city="Delhi"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()

        return {
            "humidity": res["main"]["humidity"],
            "wind": res["wind"]["speed"],
            "temperature": res["main"]["temp"]
        }
    except:
        return {"humidity": 50, "wind": 2, "temperature": 30}


# ==============================
#  SAVE TO DB
# ==============================
def save_to_db(data, aqi, weather):
    df = pd.DataFrame([{
        "pm25": data['pm25'],
        "pm10": data['pm10'],
        "co": data['co'],
        "humidity": weather['humidity'],
        "wind": weather['wind'],
        "temperature": weather['temperature'],
        "aqi": aqi
    }])
    df.to_sql("air_quality_data", engine, if_exists="append", index=False)


# ==============================
# HOME
# ==============================
@app.route('/')
def home():
    return " Smart AQI AI API Running"


# ==============================
#  ANALYZE
# ==============================
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        city = data.get("city", "Delhi")

        weather = get_weather(city)

        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        co = float(data['co'])

        humidity = weather["humidity"]
        wind = weather["wind"]
        temperature = weather["temperature"]

        features = [[pm25, pm10, co, humidity, wind, temperature]]

        #  AQI
        aqi = aqi_model.predict(features)[0]
        aqi = round(float(aqi), 2)

        #  Anomaly
        anomaly_pred = anomaly_model.predict(features)[0]
        anomaly = "Yes" if anomaly_pred == -1 else "No"

        #  Health
        aqi_scaled = scaler.transform([[aqi]])
        health_pred = health_model.predict(aqi_scaled)[0]

        health_map = {
            0: "Good",
            1: "Moderate",
            2: "Unhealthy",
            3: "Hazardous"
        }
        health = health_map.get(health_pred, "Unknown")

        alert = alert_system(aqi)
        advice = health_advice(aqi)

        log_anomaly(data, anomaly)
        save_to_db(data, aqi, weather)

        if aqi > 250:
            send_email_alert(aqi, alert)

        return jsonify({
            "AQI": aqi,
            "Anomaly": anomaly,
            "Health": health,
            "Alert": alert,
            "Advice": advice,
            "Humidity": humidity,
            "Wind": wind,
            "Temperature": temperature,
            "City": city
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get("query")
        aqi = data.get("aqi", 0)

        prompt = f"""
        You are an expert Air Quality Assistant.

        Current AQI: {aqi}

        User: {query}

        Give short, helpful, practical advice.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AQI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        return jsonify({
            "response": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({"response": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)