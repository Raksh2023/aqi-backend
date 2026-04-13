import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("🚀 Starting model training...")

# Load dataset
file_path = "/Users/rb123/Downloads/airqualitydataset.csv/city_day.csv"
df = pd.read_csv(file_path)

print("✅ File loaded successfully")

# Convert column names to lowercase
df.columns = df.columns.str.lower()
print("Columns:", df.columns)

# Features and target
features = ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3']
target = 'aqi'

# Clean data
df = df[features + [target]].dropna()
print("✅ Data cleaned")

# Split
X = df[features]
y = df[target]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Data scaled")

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, y)
print("✅ Model trained")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/aqi_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("🎉 DONE! Everything working perfectly")
